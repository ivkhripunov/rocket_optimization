"""
TargetOrbit — спецификация целевой орбиты.

apply_target_orbit_maptor() добавляет event-constraints (граничные условия)
на последнюю фазу Maptor-задачи через rv2oe с smooth Heaviside —
те же формулы, что в рабочем Maptor-примере.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import casadi as ca

from src.maptor.constants import EARTH_MU


# =============================================================================
# Вспомогательные функции (CasADi-символьные)
# =============================================================================

def _cross(a, b):
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _smooth_h(x, a=0.1):
    """Сглаженная функция Хевисайда (smooth Heaviside) через tanh."""
    return 0.5 * (1.0 + ca.tanh(x / a))


def rv2oe(rv, vv, mu=EARTH_MU, a_eps=0.1):
    """
    Декартовы координаты → орбитальные элементы (CasADi-символьно).
    Возвращает ca.MX(6): [a, e, i, Ω, ω, ν]

    Квадранты Ω, ω, ν определяются через _smooth_h — производная
    существует везде, нет разрывов в оптимизации.
    """
    eps = 1e-12

    K = ca.vertcat(0.0, 0.0, 1.0)
    hv = _cross(rv, vv)
    nv = _cross(K, hv)

    h2 = ca.fmax(_dot(hv, hv), eps)
    v2 = ca.fmax(_dot(vv, vv), eps)
    r = ca.sqrt(ca.fmax(_dot(rv, rv), eps))
    n = ca.sqrt(ca.fmax(_dot(nv, nv), eps))

    rv_dot_vv = _dot(rv, vv)
    ev = (1.0 / mu) * ((v2 - mu / r) * rv - rv_dot_vv * vv)

    p = h2 / mu
    e = ca.sqrt(ca.fmax(_dot(ev, ev), eps))
    a_oe = p / ca.fmax(1.0 - e * e, eps)
    i_oe = ca.acos(ca.fmax(ca.fmin(hv[2] / ca.sqrt(h2), 1.0 - eps), -1.0 + eps))

    # RAAN
    cos_Om = ca.fmax(ca.fmin(nv[0] / n, 1.0 - eps), -1.0 + eps)
    Om_raw = ca.acos(cos_Om)
    Om = _smooth_h(nv[1], a_eps) * Om_raw + _smooth_h(-nv[1], a_eps) * (2 * ca.pi - Om_raw)

    # Аргумент перицентра
    nv_dot_ev = _dot(nv, ev)
    cos_om = ca.fmax(ca.fmin(nv_dot_ev / (n * e), 1.0 - eps), -1.0 + eps)
    om_raw = ca.acos(cos_om)
    om = _smooth_h(ev[2], a_eps) * om_raw + _smooth_h(-ev[2], a_eps) * (2 * ca.pi - om_raw)

    # Истинная аномалия
    ev_dot_rv = _dot(ev, rv)
    cos_nu = ca.fmax(ca.fmin(ev_dot_rv / (e * r), 1.0 - eps), -1.0 + eps)
    nu_raw = ca.acos(cos_nu)
    nu = _smooth_h(rv_dot_vv, a_eps) * nu_raw + _smooth_h(-rv_dot_vv, a_eps) * (2 * ca.pi - nu_raw)

    return ca.vertcat(a_oe, e, i_oe, Om, om, nu)


def oe2rv(a, e, inc, raan, aop, nu, mu=EARTH_MU):
    """Орбитальные элементы → ECI (r, v). Numpy, только для initial guess."""
    p = a * (1.0 - e ** 2)
    r = p / (1.0 + e * np.cos(nu))

    rv_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    vv_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    co, so = np.cos(aop), np.sin(aop)
    ci, si = np.cos(inc), np.sin(inc)

    R = np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ])
    return R @ rv_pf, R @ vv_pf


# =============================================================================
# TargetOrbit
# =============================================================================

@dataclass
class TargetOrbit:
    """Целевая орбита. Не заданные элементы (None) не ограничиваются."""

    # ── Значения ──────────────────────────────────────────────────────────────
    a: Optional[float] = None  # м
    e: Optional[float] = None
    inc_deg: Optional[float] = None  # °
    arg_periapsis_deg: Optional[float] = None  # °
    raan_deg: Optional[float] = None  # °

    # ── Диапазоны (lo_offset, hi_offset) относительно значения ───────────────
    a_bounds: Optional[tuple] = None  # м
    e_bounds: Optional[tuple] = None
    inc_bounds_deg: Optional[tuple] = None  # °
    arg_periapsis_bounds_deg: Optional[tuple] = None  # °
    raan_bounds_deg: Optional[tuple] = None  # °

    def scaled(self, factor: float) -> 'TargetOrbit':
        """Вернуть копию с bounds × factor (для гомотопии)."""
        from dataclasses import replace

        def _s(b):
            return (b[0] * factor, b[1] * factor) if b is not None else None

        return replace(self,
                       a_bounds=_s(self.a_bounds),
                       e_bounds=_s(self.e_bounds),
                       inc_bounds_deg=_s(self.inc_bounds_deg),
                       arg_periapsis_bounds_deg=_s(self.arg_periapsis_bounds_deg),
                       raan_bounds_deg=_s(self.raan_bounds_deg),
                       )

    def target_rv(self, nu: float = 0.0):
        """Вычислить (r, v) для целевой орбиты при истинной аномалии nu."""
        if self.a is None:
            raise ValueError("a не задана — нельзя вычислить r, v")
        return oe2rv(
            a=self.a,
            e=self.e or 0.0,
            inc=np.deg2rad(self.inc_deg) if self.inc_deg else 0.0,
            raan=np.deg2rad(self.raan_deg) if self.raan_deg else 0.0,
            aop=np.deg2rad(self.arg_periapsis_deg) if self.arg_periapsis_deg else 0.0,
            nu=nu,
        )

    def __repr__(self):
        parts = []
        if self.a is not None:
            lo, hi = self.a_bounds or (0, 0)
            parts.append(f'a={self.a / 1e3:.0f}[{lo / 1e3:+.0f},{hi / 1e3:+.0f}] км')
        if self.e is not None:
            lo, hi = self.e_bounds or (0, 0)
            parts.append(f'e={self.e:.3f}[{lo:+.3f},{hi:+.3f}]')
        if self.inc_deg is not None:
            lo, hi = self.inc_bounds_deg or (0, 0)
            parts.append(f'i={self.inc_deg:.1f}[{lo:+.1f},{hi:+.1f}]°')
        if self.raan_deg is not None:
            lo, hi = self.raan_bounds_deg or (0, 0)
            parts.append(f'Ω={self.raan_deg:.1f}[{lo:+.1f},{hi:+.1f}]°')
        if self.arg_periapsis_deg is not None:
            lo, hi = self.arg_periapsis_bounds_deg or (0, 0)
            parts.append(f'ω={self.arg_periapsis_deg:.1f}[{lo:+.1f},{hi:+.1f}]°')
        return f'TargetOrbit({", ".join(parts)})'


# =============================================================================
# Применение ограничений к Maptor-фазе
# =============================================================================

def apply_target_orbit_maptor(phase, r_final_s, v_final_s, orbit: TargetOrbit,
                              r_scale: float, v_scale: float):
    """
    Добавить event-constraints целевой орбиты на финальную фазу.

    r_final_s, v_final_s — МАСШТАБИРОВАННЫЕ финальные состояния (CasADi-символы).
    """
    r_phys = ca.vertcat(*r_final_s) * r_scale
    v_phys = ca.vertcat(*v_final_s) * v_scale
    oe = rv2oe(r_phys, v_phys)  # [a, e, i, Ω, ω, ν]

    constraints = []

    if orbit.a is not None:
        lo, hi = orbit.a_bounds
        # Maptor поддерживает только equality ==; для inequality используем два:
        # orbit.a + lo <= oe[0] <= orbit.a + hi
        # → добавляем через равенство с slack или просто используем равенство
        # Если bounds симметричны и малы — equality ≈ OK
        # Иначе используем два inequality (Maptor поддерживает >= и <=)
        constraints.append(oe[0] >= orbit.a + lo)
        constraints.append(oe[0] <= orbit.a + hi)

    if orbit.e is not None:
        lo, hi = orbit.e_bounds
        constraints.append(oe[1] >= max(orbit.e + lo, 0.0))
        constraints.append(oe[1] <= orbit.e + hi)

    if orbit.inc_deg is not None:
        inc_rad = np.deg2rad(orbit.inc_deg)
        lo_rad = np.deg2rad(orbit.inc_bounds_deg[0])
        hi_rad = np.deg2rad(orbit.inc_bounds_deg[1])
        constraints.append(oe[2] >= max(inc_rad + lo_rad, 0.0))
        constraints.append(oe[2] <= inc_rad + hi_rad)

    if orbit.raan_deg is not None:
        raan_rad = np.deg2rad(orbit.raan_deg)
        lo_rad = np.deg2rad(orbit.raan_bounds_deg[0])
        hi_rad = np.deg2rad(orbit.raan_bounds_deg[1])
        constraints.append(oe[3] >= raan_rad + lo_rad)
        constraints.append(oe[3] <= raan_rad + hi_rad)

    if orbit.arg_periapsis_deg is not None:
        w_rad = np.deg2rad(orbit.arg_periapsis_deg)
        lo_rad = np.deg2rad(orbit.arg_periapsis_bounds_deg[0])
        hi_rad = np.deg2rad(orbit.arg_periapsis_bounds_deg[1])
        constraints.append(oe[4] >= w_rad + lo_rad)
        constraints.append(oe[4] <= w_rad + hi_rad)

    if constraints:
        phase.event_constraints(*constraints)
