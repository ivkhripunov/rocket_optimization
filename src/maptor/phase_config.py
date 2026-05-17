from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseConfig:
    name:     str
    phase_id: int            # номер фазы в Maptor (1, 2, 3, …)

    # ── Флаги фиксации ─────────────────────────────────────────────────────
    fix_duration:     bool = True
    fix_thrust:       bool = True
    fix_m_dry:        bool = True    # ← сухая масса / минимальная масса фазы
    fix_m_propellant: bool = True
    fix_Isp:          bool = True

    # ── Значения параметров ────────────────────────────────────────────────
    duration:     float = 100.0       # с
    thrust:       float = 1.0e6      # Н
    m_dry:        float = 1_000.0    # кг — минимум массы (нижняя граница)
    m_propellant: float = 10_000.0   # кг — сжигаемое топливо этой фазы
    Isp:          float = 300.0      # с

    # ── Диапазоны (при оптимизации конструкции) ────────────────────────────
    duration_bounds:     Optional[tuple] = None  # (t_lo, t_hi) абс. время конца, с
    thrust_bounds:       Optional[tuple] = None  # (lo, hi), Н
    m_dry_bounds:        Optional[tuple] = None  # (lo, hi), кг
    m_propellant_bounds: Optional[tuple] = None  # (lo, hi), кг
    Isp_bounds:          Optional[tuple] = None  # (lo, hi), с

    # ── Атмосфера ──────────────────────────────────────────────────────────
    use_atmosphere: bool  = True
    rho_ref:        float = 1.225    # кг/м³
    h_scale:        float = 7200.0   # м  (Maptor-значение, не 8440!)

    # ── Аэродинамика ───────────────────────────────────────────────────────
    CD: float = 0.5
    S:  float = 4 * 3.14159265      # м² ≈ 12.57 (Maptor: 4π)

    # ── Путевые ограничения ────────────────────────────────────────────────
    q_heat_constraint: bool  = False
    q_dyn_constraint:  bool  = False
    g_load_constraint: bool  = False
    q_heat_max:        float = 5.0e5
    q_dyn_max:         float = 120_000.0
    g_load_max:        float = 10.0

    # ── Сетка ──────────────────────────────────────────────────────────────
    nodes_per_interval: list = field(default_factory=lambda: [4, 4])

    # ── Производные (read-only) ────────────────────────────────────────────
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
        if not self.fix_m_propellant and self.fix_duration:
            raise ValueError(
                f'{self.name}: fix_m_propellant=False при fix_duration=True '
                f'лишено смысла: расход топлива однозначно задан через '
                f'thrust, Isp, duration.')
        if not self.fix_m_dry and self.m_dry_bounds is None:
            raise ValueError(
                f'{self.name}: fix_m_dry=False требует m_dry_bounds')
        if not self.fix_thrust and self.thrust_bounds is None:
            raise ValueError(
                f'{self.name}: fix_thrust=False требует thrust_bounds')
        if not self.fix_Isp and self.Isp_bounds is None:
            raise ValueError(
                f'{self.name}: fix_Isp=False требует Isp_bounds')
        if not self.fix_m_propellant and self.m_propellant_bounds is None:
            raise ValueError(
                f'{self.name}: fix_m_propellant=False требует m_propellant_bounds')