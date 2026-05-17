"""
Миссия Delta III → ГТО.

Физические параметры совпадают с рабочим Maptor-примером:
    BOOSTER_TOTAL = 19 290 кг  (исходный код отчёта: 19 200 — НЕВЕРНО)
    BOOSTER_DRY   =  2 280 кг  (исходный код отчёта:  2 190 — НЕВЕРНО)
    H_SCALE       =  7 200 м   (исходный код отчёта:  8 440 — НЕВЕРНО)
    A_TARGET      = 24 361 140 м

Целевая орбита (ГТО):
    a    = 24 361 140 м
    e    = 0.7308
    i    = 28.5°
    RAAN = 269.8°
    AOP  = 130.5°
"""

import numpy as np
import maptor as mtor

from src.maptor.constants import G0, EARTH_RAD, M_SCALE
from src.maptor.phase_config import PhaseConfig
from src.maptor.target_orbit import TargetOrbit
from src.maptor.multi_stage import run_multi_stage, print_results

# =========================================================
# Физические характеристики Delta III
# =========================================================
BOOSTER_TOTAL = 19_290.0  # кг  ← Maptor-значение (не 19200!)
BOOSTER_PROP = 17_010.0  # кг
BOOSTER_DRY = BOOSTER_TOTAL - BOOSTER_PROP  # 2 280 кг (не 2190!)
BOOSTER_F = 628_500.0  # Н
BOOSTER_ISP = 284.0  # с

STAGE1_TOTAL = 104_380.0  # кг
STAGE1_PROP = 95_550.0  # кг
STAGE1_DRY = STAGE1_TOTAL - STAGE1_PROP  # 8 830 кг
STAGE1_F = 1_083_100.0  # Н
STAGE1_ISP = 301.7  # с

STAGE2_TOTAL = 19_300.0  # кг
STAGE2_PROP = 16_820.0  # кг
STAGE2_DRY = STAGE2_TOTAL - STAGE2_PROP  # 2 480 кг
STAGE2_F = 110_094.0  # Н
STAGE2_ISP = 462.4  # с

PAYLOAD = 4_164.0  # кг
N_BOOSTERS = 9

# =========================================================
# Временна́я сетка (как в Maptor-примере)
# =========================================================
T1_END = 75.2  # с, конец фазы 1
T2_END = 150.4  # с, конец фазы 2
T3_END = 261.0  # с, конец фазы 3
T4_MAX = 961.0  # с, максимальный конец фазы 4

# =========================================================
# Расходы и Isp (вычисляются через mdot, как в Maptor-примере)
#
# ВАЖНО: Maptor определяет mdot как m_propellant / t_burn,
# а затем Isp = thrust / (G0 * mdot). Для совместимости
# используем те же значения.
# =========================================================
MDOT_SRB = BOOSTER_PROP / T1_END  # 17010/75.2  = 226.2 кг/с
MDOT_FIRST = STAGE1_PROP / T3_END  # 95550/261.0 = 366.1 кг/с
MDOT_SECOND = STAGE2_PROP / 700.0  # 16820/700   = 24.03 кг/с

ISP_SRB = BOOSTER_F / (G0 * MDOT_SRB)  # ≈ 283.3 с
ISP_FIRST = STAGE1_F / (G0 * MDOT_FIRST)  # ≈ 301.7 с
ISP_SECOND = STAGE2_F / (G0 * MDOT_SECOND)  # ≈ 467.1 с

# =========================================================
# Полная стартовая масса
# =========================================================
M_INITIAL = N_BOOSTERS * BOOSTER_TOTAL + STAGE1_TOTAL + STAGE2_TOTAL + PAYLOAD

# =========================================================
# Массы по фазам (точное воспроизведение Maptor-примера)
# =========================================================

# Фаза 1: 6 ускорителей + 1 ступень
F_PH1 = STAGE1_F + 6 * BOOSTER_F
MDOT_PH1 = MDOT_FIRST + 6 * MDOT_SRB
ISP_PH1 = F_PH1 / (G0 * MDOT_PH1)
PROP_PH1 = MDOT_PH1 * T1_END
M_END_PH1 = M_INITIAL - PROP_PH1
M_DROP_1 = 6 * BOOSTER_DRY  # 13 680 кг
M_START_PH2 = M_END_PH1 - M_DROP_1

# Фаза 2: 3 ускорителя + 1 ступень
F_PH2 = STAGE1_F + 3 * BOOSTER_F
MDOT_PH2 = MDOT_FIRST + 3 * MDOT_SRB
ISP_PH2 = F_PH2 / (G0 * MDOT_PH2)
PROP_PH2 = MDOT_PH2 * (T2_END - T1_END)
M_END_PH2 = M_START_PH2 - PROP_PH2
M_DROP_2 = 3 * BOOSTER_DRY  # 6 840 кг
M_START_PH3 = M_END_PH2 - M_DROP_2

# Фаза 3: только 1 ступень
F_PH3 = STAGE1_F
MDOT_PH3 = MDOT_FIRST
ISP_PH3 = ISP_FIRST
PROP_PH3 = MDOT_PH3 * (T3_END - T2_END)
M_END_PH3 = M_START_PH3 - PROP_PH3
M_DROP_3 = STAGE1_DRY  # 8 830 кг
M_START_PH4 = M_END_PH3 - M_DROP_3

# Фаза 4: только 2 ступень
F_PH4 = STAGE2_F
ISP_PH4 = ISP_SECOND
M_DRY_PH4 = STAGE2_DRY + PAYLOAD  # 2480 + 4164 = 6 644 кг


# =========================================================
# Конфигурации фаз
# =========================================================

def make_delta3_phase_configs(
        use_atmosphere: bool = True,
        CD: float = 0.5,
) -> list:
    """
    Создать 4 конфигурации виртуальных фаз Delta III.

    По умолчанию все параметры кроме длительности фазы 4 зафиксированы.
    Для оптимизации конструкции установите fix_*=False с соответствующими *_bounds.

    Примечание по mass_drops:
        M_DROP_1, M_DROP_2, M_DROP_3 — ФИКСИРОВАННЫЕ параметры (сухие массы
        сбрасываемых ускорителей/ступени), независимые от оптимизируемых
        m_dry/m_propellant фаз. Передаются в run_multi_stage отдельно.
    """
    common_atm = dict(
        use_atmosphere=use_atmosphere,
        CD=CD,
        S=4 * np.pi,  # ≈ 12.57 м² (Maptor)
        nodes_per_interval=[4, 4],
    )

    phase1 = PhaseConfig(
        name='phase_1', phase_id=1,
        # Всё зафиксировано — только направление тяги оптимизируется
        fix_duration=True, fix_thrust=True, fix_m_dry=True,
        fix_m_propellant=True, fix_Isp=True,
        duration=T1_END,
        thrust=F_PH1,
        m_dry=M_END_PH1,  # минимальная масса в фазе 1
        m_propellant=PROP_PH1,
        Isp=ISP_PH1,
        **common_atm,
    )

    phase2 = PhaseConfig(
        name='phase_2', phase_id=2,
        fix_duration=True, fix_thrust=True, fix_m_dry=True,
        fix_m_propellant=True, fix_Isp=True,
        duration=T2_END - T1_END,
        thrust=F_PH2,
        m_dry=M_END_PH2,
        m_propellant=PROP_PH2,
        Isp=ISP_PH2,
        **common_atm,
    )

    phase3 = PhaseConfig(
        name='phase_3', phase_id=3,
        fix_duration=True, fix_thrust=True, fix_m_dry=True,
        fix_m_propellant=True, fix_Isp=True,
        duration=T3_END - T2_END,
        thrust=F_PH3,
        m_dry=M_END_PH3,
        m_propellant=PROP_PH3,
        Isp=ISP_PH3,
        **common_atm,
    )

    phase4 = PhaseConfig(
        name='phase_4', phase_id=4,
        fix_duration=False, fix_thrust=True, fix_m_dry=True,
        fix_m_propellant=True, fix_Isp=True,
        duration=700.0,  # guess
        duration_bounds=(T3_END + 1.0, T4_MAX),  # абс. время конца
        thrust=F_PH4,
        m_dry=M_DRY_PH4,
        m_propellant=STAGE2_PROP,
        Isp=ISP_PH4,
        use_atmosphere=False,  # вакуум
        CD=CD,
        S=4 * np.pi,
        nodes_per_interval=[4, 4],
    )

    return [phase1, phase2, phase3, phase4]


# =========================================================
# Запуск с гомотопией
# =========================================================

def run_delta3_gto(
        use_atmosphere: bool = True,
        CD: float = 0.5,
):
    """
    Оптимизировать траекторию Delta III → ГТО.

    Три шага гомотопии:
        Шаг 1: только a и e, широко     — найти нужный энергетический уровень
        Шаг 2: добавить i, сузить       — зафиксировать плоскость орбиты
        Шаг 3: добавить RAAN, финально  — точная ориентация орбиты
    """
    phases = make_delta3_phase_configs(
        use_atmosphere=use_atmosphere,
        CD=CD,
    )
    mass_drops = [M_DROP_1, M_DROP_2, M_DROP_3, 0.0]

    orbit_steps = [
        # Шаг 1: только a и e, широкие допуски
        TargetOrbit(
            a=24_361_140.0, a_bounds=(-5_000_000, +5_000_000),
            e=0.7308, e_bounds=(-0.15, +0.15),
        ),
        # Шаг 2: добавляем наклонение, сужаем
        TargetOrbit(
            a=24_361_140.0, a_bounds=(-1_000_000, +1_000_000),
            e=0.7308, e_bounds=(-0.05, +0.05),
            arg_periapsis_deg=130.5, arg_periapsis_bounds_deg=(-15.0, +15.0),
            inc_deg=28.5, inc_bounds_deg=(-3.0, +3.0),
        ),
        # Шаг 3: добавляем RAAN, финальные допуски
        TargetOrbit(
            a=24_361_140.0, a_bounds=(-200_000, +200_000),
            e=0.7308, e_bounds=(-0.01, +0.01),
            arg_periapsis_deg=130.5, arg_periapsis_bounds_deg=(-5.0, +5.0),
            inc_deg=28.5, inc_bounds_deg=(-0.5, +0.5),
            raan_deg=269.8, raan_bounds_deg=(-5.0, +5.0),
        ),
    ]

    solution = None
    for step, orbit in enumerate(orbit_steps):
        is_last = (step == len(orbit_steps) - 1)

        print(f'\n{"=" * 65}')
        print(f'Шаг {step + 1}/{len(orbit_steps)}: {orbit}')
        print('=' * 65)

        solution = run_multi_stage(
            phases=phases,
            mass_drops=mass_drops,
            launch_lat_deg=28.5,
            launch_lon_deg=0.0,  # как в Maptor-примере
            launch_alt=0.0,
            objective='max_final_mass',
            target_orbit=orbit,
            error_tol=1e-4 if is_last else 1e-3,
            max_refine_iter=20,
            problem_name=f'delta3_step{step}',
        )

        if not solution.status['success']:
            print(f'  ⚠  Шаг {step + 1} не сошёлся: {solution.status["message"]}')
            if step == 0:
                break

    return solution


# =========================================================
# Сводка параметров
# =========================================================

def print_delta3_summary():
    print('=' * 65)
    print('Delta III — параметры ракеты')
    print('=' * 65)
    print(f'Стартовая масса:           {M_INITIAL:>12,.1f} кг')
    print(f'  9 ускорителей × {BOOSTER_TOTAL:.0f}:  {N_BOOSTERS * BOOSTER_TOTAL:>12,.1f} кг')
    print(f'  Ступень 1:               {STAGE1_TOTAL:>12,.1f} кг')
    print(f'  Ступень 2:               {STAGE2_TOTAL:>12,.1f} кг')
    print(f'  Полезная нагрузка:       {PAYLOAD:>12,.1f} кг')
    print()

    data = [
        ('Фаза 1', '6 уск + ст1', F_PH1, ISP_PH1, T1_END,
         M_INITIAL, M_END_PH1, M_DROP_1),
        ('Фаза 2', '3 уск + ст1', F_PH2, ISP_PH2, T2_END - T1_END,
         M_START_PH2, M_END_PH2, M_DROP_2),
        ('Фаза 3', 'ст1', F_PH3, ISP_PH3, T3_END - T2_END,
         M_START_PH3, M_END_PH3, M_DROP_3),
        ('Фаза 4', 'ст2 → ГТО', F_PH4, ISP_PH4, None,
         M_START_PH4, M_DRY_PH4, 0.0),
    ]
    for name, desc, F, Isp, dur, m_i, m_f, drop in data:
        dur_s = f'{dur:.1f} с' if dur is not None else 'FREE'
        print(f'{name} ({desc})')
        print(f'  F={F:>12,.0f} Н   Isp={Isp:.1f} с   t={dur_s}')
        print(f'  m: {m_i:>10,.0f} → {m_f:>10,.0f} кг   drop={drop:>7,.0f} кг')
        print()
    print('=' * 65)


# =========================================================
# Точка входа
# =========================================================

if __name__ == '__main__':
    print_delta3_summary()

    solution = run_delta3_gto(use_atmosphere=True, CD=0.5)

    phases = make_delta3_phase_configs()
    mass_drops = [M_DROP_1, M_DROP_2, M_DROP_3, 0.0]
    print_results(solution, phases, mass_drops)

    if solution.status['success']:
        solution.plot()
