from src.stage_config import StageConfig


class PhaseConfig(StageConfig):
    # ===== Атмосфера =====
    use_atmosphere: bool = True
    rho_ref: float = 1.225  # кг/м³, плотность на h=0
    h_scale: float = 8.44e3  # м, масштаб высоты

    # ===== Сетка =====
    num_segments: int = 15
    order: int = 3

    # ===== Уточнение сетки =====
    refine: bool = True
    refine_method: str = 'ph'  # 'hp' или 'ph'
    refine_iter_limit: int = 1
    refine_tol: float = 1.0e-1
    refine_min_order: int = 3
    refine_max_order: int = 3
    refine_smoothness: float = 1.5
