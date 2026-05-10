from dataclasses import dataclass

@dataclass
class StageConfig:
    name: str = 'phase0'

    # ===== Геометрия носа (для Sutton-Graves) =====
    nose_radius: float = 0.5  # м, радиус кривизны

    # ===== Двигатель =====
    thrust_max: float = 2.1e6  # Н, максимальная тяга
    Isp: float = 265.2  # с, удельный импульс

    # ===== Масса =====
    m_dry: float = 1_000.0  # кг, сухая масса этой ступени
    m_propellant: float = 116_000.0  # кг, топливо в этой ступени

    # ===== Аэродинамика =====
    CD: float = 0.5  # аэродинамический коэффициент
    S: float = 7.069  # м^2, характерная площадь

    # ===== Границы для design-оптимизации =====
    thrust_max_bounds: tuple = (1.0e5, 1.0e7)
    Isp_bounds: tuple = (200.0, 450.0)
    m_dry_bounds: tuple = (100.0, 1.0e5)
    m_propellant_bounds: tuple = (1.0e3, 1.0e6)

    def m_total(self) -> float:
        return self.m_dry + self.m_propellant

    def m_min(self) -> float:
        return self.m_dry
