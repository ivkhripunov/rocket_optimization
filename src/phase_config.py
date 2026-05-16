from dataclasses import dataclass


@dataclass
class PhaseConfig:
    name: str

    # ===== Фикс параметров =====
    fix_duration: bool
    fix_thrust: bool
    fix_throttle: bool
    fix_m_dry: bool
    fix_m_propellant: bool
    fix_Isp: bool

    # ===== Значения параметров =====
    duration: float
    thrust: float
    throttle: float
    m_dry: float
    m_propellant: float
    Isp: float

    # ===== Диапазоны параметров =====
    duration_bounds: tuple
    thrust_bounds: tuple
    throttle_bounds: tuple
    m_dry_bounds: tuple
    m_propellant_bounds: tuple
    Isp_bounds: tuple

    # ===== Модель =====
    use_atmosphere: bool
    q_heat_constraint: bool
    q_dyn_constraint: bool
    g_load_constraint: bool

    # ===== Аэродинамика =====
    nose_radius: float  # м, радиус кривизны
    CD: float  # аэродинамический коэффициент
    S: float  # м^2, характерная площадь

    # ===== Атмосфера =====
    rho_ref: float = 1.225  # кг/м³, плотность на h=0
    h_scale: float = 8.44e3  # м, масштаб высоты

    # ===== Тепловые / аэродинамические нагрузки =====
    q_heat_max: float = 5.0e5  # Вт/м², ограничение теплового потока
    q_dyn_max: float = 120_000.0  # Па, ограничение динамического давления

    # ===== Структурные ограничения =====
    g_load_max: float = 10.0  # перегрузка в g

    # ===== Сетка =====
    num_segments: int = 10
    order: int = 3

    # ===== Уточнение сетки =====
    refine: bool = False
    refine_method: str = 'ph'  # 'hp' или 'ph'
    refine_iter_limit: int = 1
    refine_tol: float = 1.0e-1
    refine_min_order: int = 3
    refine_max_order: int = 3
    refine_smoothness: float = 1.5
