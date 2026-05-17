from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseConfig:
    name: str
    phase_id: int

    # ── Флаги фиксации ─────────────────────────────────────────────────────
    fix_duration: bool = True
    fix_thrust: bool = True
    fix_throttle: bool = True  # throttle=1.0 или свободный [0,1]
    fix_m_dry: bool = True
    fix_m_propellant: bool = True
    fix_Isp: bool = True

    # ── Значения параметров ────────────────────────────────────────────────
    duration: float = 100.0
    thrust: float = 1.0e6
    throttle: float = 1.0
    m_dry: float = 1_000.0
    m_propellant: float = 10_000.0
    Isp: float = 300.0

    # ── Диапазоны ──────────────────────────────────────────────────────────
    duration_bounds: Optional[tuple] = None  # (t_lo, t_hi) абс. конец, с
    thrust_bounds: Optional[tuple] = None  # (lo, hi), Н
    throttle_bounds: Optional[tuple] = None  # (lo, hi) ∈ [0,1]
    m_dry_bounds: Optional[tuple] = None  # (lo, hi), кг
    m_propellant_bounds: Optional[tuple] = None  # (lo, hi), кг
    Isp_bounds: Optional[tuple] = None  # (lo, hi), с

    # ── Атмосфера ──────────────────────────────────────────────────────────
    use_atmosphere: bool = True
    rho_ref: float = 1.225
    h_scale: float = 7200.0

    # ── Аэродинамика ───────────────────────────────────────────────────────
    CD: float = 0.5
    S: float = 4 * 3.14159265

    # ── Путевые ограничения ────────────────────────────────────────────────
    q_heat_constraint: bool = False
    q_dyn_constraint: bool = False
    g_load_constraint: bool = False
    q_heat_max: float = 5.0e5
    q_dyn_max: float = 120_000.0
    g_load_max: float = 10.0

    # ── Сетка ──────────────────────────────────────────────────────────────
    nodes_per_interval: list = field(default_factory=lambda: [4, 4])

    # ── Read-only свойства ─────────────────────────────────────────────────
    @property
    def m_initial(self) -> float:
        return self.m_dry + self.m_propellant

    @property
    def n_intervals(self) -> int:
        return len(self.nodes_per_interval)

    @property
    def tau_boundaries(self) -> list:
        import numpy as np
        return list(np.linspace(-1.0, 1.0, self.n_intervals + 1))

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes_per_interval)

    def validate(self):
        """Проверить физическую и логическую совместимость."""

        if not self.fix_m_propellant and self.fix_throttle:
            raise ValueError(
                f'{self.name}: fix_m_propellant=False требует fix_throttle=False. '
                f'При throttle=1.0 всё топливо сгорит полностью — '
                f'свободный m_propellant не даёт реальной степени свободы. '
                f'Установите fix_throttle=False, throttle_bounds=(0.0, 1.0).')

        required_bounds = {
            'fix_m_dry': ('m_dry_bounds', self.fix_m_dry, self.m_dry_bounds),
            'fix_thrust': ('thrust_bounds', self.fix_thrust, self.thrust_bounds),
            'fix_Isp': ('Isp_bounds', self.fix_Isp, self.Isp_bounds),
            'fix_m_propellant': ('m_propellant_bounds', self.fix_m_propellant, self.m_propellant_bounds),
            'fix_throttle': ('throttle_bounds', self.fix_throttle, self.throttle_bounds),
        }
        for flag_name, (bounds_name, is_fixed, bounds) in required_bounds.items():
            if not is_fixed and bounds is None:
                raise ValueError(
                    f'{self.name}: {flag_name}=False требует {bounds_name}')