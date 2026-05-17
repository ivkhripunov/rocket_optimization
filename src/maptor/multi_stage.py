"""
run_multi_stage() — многофазная задача оптимизации траектории в Maptor.

Линковка масс между фазами:
    m_start(B) = m_final(A) - mass_drop_after_A

mass_drop — это ФИКСИРОВАННАЯ аппаратная масса (корпуса сбрасываемых
ускорителей, обтекателей). Она НЕ зависит от m_dry и m_propellant фаз,
даже если эти параметры оптимизируются.

При оптимизации m_dry(A):
    m_dry(A) — нижняя граница массы во время фазы A (path-constraint m>=m_dry_s).
    Сброс после A остаётся фиксированным.
    Следовательно: m_start(B) = m_final(A) - fixed_drop,
    а m_final(A) сам по себе теперь зависит от оптимизируемого m_dry_s.
"""

from typing import List

import numpy as np
import maptor as mtor

from src.maptor.constants import (
    EARTH_MU, EARTH_RAD, EARTH_OMEGA,
    R_SCALE, V_SCALE, M_SCALE,
)
from src.maptor.phase_config import PhaseConfig
from src.maptor.target_orbit import TargetOrbit, apply_target_orbit_maptor, oe2rv
from src.maptor.stage_phase import build_maptor_phase, set_phase_guess


def run_multi_stage(
        phases: List[PhaseConfig],
        mass_drops: List[float],
        launch_lat_deg: float,
        launch_lon_deg: float,
        launch_alt: float = 0.0,
        objective: str = 'max_final_mass',
        target_orbit: TargetOrbit | None = None,
        error_tol: float = 1e-4,
        max_refine_iter: int = 10,
        min_poly_degree: int = 3,
        max_poly_degree: int = 8,
        ipopt_options: dict | None = None,
        problem_name: str = 'multistage',
):
    """
    Создать и решить многофазную задачу оптимизации траектории.

    Parameters
    ----------
    phases        : список PhaseConfig (по одному на фазу)
    mass_drops    : список масс сброса (кг), len == len(phases).
                    mass_drops[i] = масса, сбрасываемая ПОСЛЕ фазы i.
                    Это ФИКСИРОВАННЫЕ аппаратные параметры (сухие массы
                    сбрасываемых компонентов), независимые от оптимизируемых
                    m_dry, m_propellant фаз.
    objective     : 'max_final_mass' | 'min_initial_mass'
    target_orbit  : целевая орбита (None → без орбитальных ограничений)
    error_tol     : допуск адаптивной сетки
    ipopt_options : переопределить настройки IPOPT
    """
    assert len(mass_drops) == len(phases), \
        f'len(mass_drops)={len(mass_drops)} != len(phases)={len(phases)}'

    for cfg in phases:
        cfg.validate()

    # =========================================================
    # Начальные условия (стартовая площадка)
    # =========================================================
    lat0 = np.deg2rad(launch_lat_deg)
    lon0 = np.deg2rad(launch_lon_deg)

    x0 = (EARTH_RAD + launch_alt) * np.cos(lat0) * np.cos(lon0)
    y0 = (EARTH_RAD + launch_alt) * np.cos(lat0) * np.sin(lon0)
    z0 = (EARTH_RAD + launch_alt) * np.sin(lat0)
    r0_phys = np.array([x0, y0, z0])

    omega_np = np.array([[0, -EARTH_OMEGA, 0],
                         [EARTH_OMEGA, 0, 0],
                         [0, 0, 0]])
    v0_phys = omega_np @ r0_phys

    r0_s = [r0_phys[i] / R_SCALE for i in range(3)]
    v0_s = [v0_phys[i] / V_SCALE for i in range(3)]

    # =========================================================
    # Целевая точка для guess последней фазы
    # =========================================================
    if target_orbit is not None and target_orbit.a is not None:
        rf_phys, vf_phys = target_orbit.target_rv(nu=0.0)
    else:
        zenith = r0_phys / np.linalg.norm(r0_phys)
        rf_phys = zenith * (EARTH_RAD + 300_000.0)
        east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
        vf_phys = np.sqrt(EARTH_MU / np.linalg.norm(rf_phys)) * east

    rf_s = [rf_phys[i] / R_SCALE for i in range(3)]
    vf_s = [vf_phys[i] / V_SCALE for i in range(3)]

    # =========================================================
    # Сборка задачи
    # =========================================================
    problem = mtor.Problem(problem_name)

    t_start = 0.0
    prev_result = None
    all_results = []

    for i, (cfg, drop_kg) in enumerate(zip(phases, mass_drops)):
        result = build_maptor_phase(
            problem=problem,
            config=cfg,
            t_start=t_start,
            prev_states=prev_result,
            mass_drop_kg=drop_kg,
            r0_s=r0_s if i == 0 else None,
            v0_s=v0_s if i == 0 else None,
        )
        all_results.append(result)
        t_start = result['t_end']
        prev_result = result

    # =========================================================
    # Орбитальные ограничения (на последней фазе)
    # =========================================================
    last = all_results[-1]
    if target_orbit is not None:
        apply_target_orbit_maptor(
            phase=last['phase'],
            r_final_s=[s.final for s in last['r_s']],
            v_final_s=[s.final for s in last['v_s']],
            orbit=target_orbit,
            r_scale=R_SCALE,
            v_scale=V_SCALE,
        )

    # =========================================================
    # Целевая функция
    # =========================================================
    if objective == 'max_final_mass':
        problem.minimize(-last['m_s'].final)
    elif objective == 'min_initial_mass':
        problem.minimize(all_results[0]['m_s'].initial)
    else:
        raise ValueError(f'Неизвестный objective: {objective}')

    # =========================================================
    # Начальные приближения (как в Maptor-примере)
    # =========================================================
    n_phases = len(phases)

    # Накопление масс для guess (с учётом сбросов)
    m_phase_init = [phases[0].m_initial / M_SCALE]
    for i in range(n_phases - 1):
        m_prev_final = m_phase_init[-1] - phases[i].m_propellant / M_SCALE
        m_next_start = m_prev_final - mass_drops[i] / M_SCALE
        m_phase_init.append(m_next_start)

    for i, (cfg, result) in enumerate(zip(phases, all_results)):
        is_last = (i == n_phases - 1)

        # Фазы 1–N-1: константный guess на стартовых условиях
        # Последняя фаза: константный guess на целевой орбите
        r_const = rf_s if is_last else r0_s
        v_const = vf_s if is_last else v0_s

        m_init_s = m_phase_init[i]
        m_final_s = cfg.m_dry / M_SCALE

        set_phase_guess(result, cfg, r_const, v_const, m_init_s, m_final_s)

    # =========================================================
    # Решение
    # =========================================================
    ipopt_default = {
        'ipopt.max_iter': 2000,
        'ipopt.tol': max(error_tol * 0.1, 1e-7),
        'ipopt.constr_viol_tol': max(error_tol * 0.1, 1e-7),
        'ipopt.linear_solver': 'mumps',
        'ipopt.print_level': 5,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.nlp_scaling_method': 'gradient-based',
    }
    if ipopt_options:
        ipopt_default.update(ipopt_options)

    solution = mtor.solve_adaptive(
        problem,
        error_tolerance=error_tol,
        max_iterations=max_refine_iter,
        min_polynomial_degree=min_poly_degree,
        max_polynomial_degree=max_poly_degree,
        nlp_options=ipopt_default,
    )

    return solution


# =============================================================================
# Вывод результатов
# =============================================================================

def print_results(solution,
                  phases: List[PhaseConfig],
                  mass_drops: List[float]):
    if not solution.status['success']:
        print(f'FAILED: {solution.status["message"]}')
        return

    print('=' * 70)
    print('Результаты оптимизации')
    print('=' * 70)
    obj_mass = -solution.status['objective'] * M_SCALE
    print(f'Финальная масса:   {obj_mass:>12,.2f} кг')
    print()

    for i, (cfg, drop) in enumerate(zip(phases, mass_drops)):
        pid = cfg.phase_id
        try:
            dur = solution.phases[pid]['times']['duration']
        except Exception:
            dur = cfg.duration
        m_i = cfg.m_initial
        m_f = cfg.m_dry
        dv = cfg.Isp * 9.80665 * np.log(m_i / m_f) if m_f > 0 else 0
        print(f'  {cfg.name:10s} [id={pid}]: {dur:7.1f} с | '
              f'm: {m_i:>9,.0f}→{m_f:>9,.0f} кг | '
              f'ΔV≈{dv:>6,.0f} м/с | сброс: {drop:>7,.0f} кг')

    print('=' * 70)
