"""
Миссия Delta III → ГТО (геостационарная переходная орбита).

По отчёту, ракета моделируется как 4-фазная виртуальная ступень:
  Фаза 1: 1 ступень + 6 ускорителей  (отделение 6 ускорителей в конце)
  Фаза 2: 1 ступень + 3 ускорителя   (отделение 3 ускорителей в конце)
  Фаза 3: только 1 ступень           (отделение сухой 1 ступени в конце)
  Фаза 4: 2 ступень                  (выведение на ГТО, длительность свободна)

Запуск с мыса Канаверал (28.5° N, −80.5°).

Параметры Delta III (из таблицы 3 отчёта):
  Ускоритель: m_total=19200 кг, m_prop=17010 кг, F=628500 Н, Isp=284 с
  Ступень 1:  m_total=104380 кг, m_prop=95550 кг, F=1083100 Н, Isp=301.7 с
  Ступень 2:  m_total=19300 кг, m_prop=16820 кг, F=110094 Н, Isp=462.4 с
  Полезная нагрузка: 4164 кг

Целевая орбита (ГТО):
  a = 24500 км, e ≈ 0.73, i = 28.5°
"""

import numpy as np
from src.phase_config import PhaseConfig
from src.multi_stage import run_multi_stage
from src.frame_converter import EARTH_RAD
from src.stage_ode import G0
from src.visualize import plot_eci_trajectory_3d, plot_eci_trajectory_zoomed, plot_multi_stage, print_design_results
from pathlib import Path

# =========================================================
# Физические характеристики Delta III
# =========================================================
BOOSTER_TOTAL = 19_200.0
BOOSTER_PROP = 17_010.0
BOOSTER_DRY = BOOSTER_TOTAL - BOOSTER_PROP  # 2 190 кг
BOOSTER_F = 628_500.0
BOOSTER_ISP = 284.0

STAGE1_TOTAL = 104_380.0
STAGE1_PROP = 95_550.0
STAGE1_DRY = STAGE1_TOTAL - STAGE1_PROP  # 8 830 кг
STAGE1_F = 1_083_100.0
STAGE1_ISP = 301.7

STAGE2_TOTAL = 19_300.0
STAGE2_PROP = 16_820.0
STAGE2_DRY = STAGE2_TOTAL - STAGE2_PROP  # 2 480 кг
STAGE2_F = 110_094.0
STAGE2_ISP = 462.4

PAYLOAD = 4_164.0
N_BOOSTERS = 9

# =========================================================
# Производные параметры виртуальных фаз
# =========================================================
mdot_booster = BOOSTER_F / (BOOSTER_ISP * G0)
mdot_stage1 = STAGE1_F / (STAGE1_ISP * G0)
mdot_stage2 = STAGE2_F / (STAGE2_ISP * G0)

# ----- Фаза 1: 6 ускорителей + 1 ступень -----
F_phase1 = STAGE1_F + 6 * BOOSTER_F
mdot_phase1 = mdot_stage1 + 6 * mdot_booster
Isp_phase1 = F_phase1 / (mdot_phase1 * G0)
t_phase1 = BOOSTER_PROP / mdot_booster  # ~75.4 с
prop_phase1 = mdot_phase1 * t_phase1

# ----- Фаза 2: 3 ускорителя + 1 ступень -----
F_phase2 = STAGE1_F + 3 * BOOSTER_F
mdot_phase2 = mdot_stage1 + 3 * mdot_booster
Isp_phase2 = F_phase2 / (mdot_phase2 * G0)
t_phase2 = BOOSTER_PROP / mdot_booster  # ~75.4 с
prop_phase2 = mdot_phase2 * t_phase2

# ----- Фаза 3: только 1 ступень, до выгорания -----
prop_stage1_used_in_12 = mdot_stage1 * (t_phase1 + t_phase2)
prop_stage1_remaining = STAGE1_PROP - prop_stage1_used_in_12
F_phase3 = STAGE1_F
mdot_phase3 = mdot_stage1
Isp_phase3 = STAGE1_ISP
t_phase3 = prop_stage1_remaining / mdot_stage1  # ~110 с
prop_phase3 = prop_stage1_remaining

# ----- Фаза 4: 2 ступень, длительность свободна -----
F_phase4 = STAGE2_F
mdot_phase4 = mdot_stage2
Isp_phase4 = STAGE2_ISP
t_phase4 = STAGE2_PROP / mdot_stage2

# =========================================================
# Стартовая масса = всё на пусковом столе
# =========================================================
M_INITIAL = N_BOOSTERS * BOOSTER_TOTAL + STAGE1_TOTAL + STAGE2_TOTAL + PAYLOAD
# = 9*19200 + 104380 + 19300 + 4164 = 300 644 кг   (совпадает с отчётом)

# Массы по фазам
m_after_phase1 = M_INITIAL - prop_phase1
m_drop_after_1 = 6 * BOOSTER_DRY  # 13 140 кг
m_initial_p2 = m_after_phase1 - m_drop_after_1

m_after_phase2 = m_initial_p2 - prop_phase2
m_drop_after_2 = 3 * BOOSTER_DRY  # 6 570 кг
m_initial_p3 = m_after_phase2 - m_drop_after_2

m_after_phase3 = m_initial_p3 - prop_phase3
m_drop_after_3 = STAGE1_DRY  # 8 830 кг
m_initial_p4 = m_after_phase3 - m_drop_after_3  # ≈ STAGE2_TOTAL + PAYLOAD

m_dry_phase4 = STAGE2_DRY + PAYLOAD  # минимум массы = сухая ст2 + ПН


def make_delta3_phase_configs():
    """4 конфигурации виртуальных ступеней Delta III."""

    phase1 = PhaseConfig(
        name='phase_1',
        fix_duration=True,
        fix_thrust=True,
        fix_throttle=True,
        fix_m_dry=True,
        fix_m_propellant=True,
        fix_Isp=True,

        duration=t_phase1,
        thrust=F_phase1,
        throttle=1.0,
        m_dry=m_after_phase1,
        m_propellant=prop_phase1,
        Isp=Isp_phase1,

        duration_bounds=(None, None),
        thrust_bounds=(None, None),
        throttle_bounds=(None, None),
        m_dry_bounds=(None, None),
        m_propellant_bounds=(None, None),
        Isp_bounds=(None, None),

        use_atmosphere=True,
        q_heat_constraint=False,
        q_dyn_constraint=False,
        g_load_constraint=False,

        nose_radius=1e6,
        CD=0.5,
        S=4 * 3.14,
    )

    phase2 = PhaseConfig(
        name='phase_2',

        fix_duration=True,
        fix_thrust=True,
        fix_throttle=True,
        fix_m_dry=True,
        fix_m_propellant=True,
        fix_Isp=True,

        duration=t_phase2,
        thrust=F_phase2,
        throttle=1.0,
        m_dry=m_after_phase2,
        m_propellant=prop_phase2,
        Isp=Isp_phase2,

        duration_bounds=(None, None),
        thrust_bounds=(None, None),
        throttle_bounds=(None, None),
        m_dry_bounds=(None, None),
        m_propellant_bounds=(None, None),
        Isp_bounds=(None, None),

        use_atmosphere=True,
        q_heat_constraint=False,
        q_dyn_constraint=False,
        g_load_constraint=False,

        nose_radius=1e6,
        CD=0.5,
        S=4 * 3.14,
    )

    phase3 = PhaseConfig(
        name='phase_3',

        fix_duration=True,
        fix_thrust=True,
        fix_throttle=True,
        fix_m_dry=True,
        fix_m_propellant=True,
        fix_Isp=True,

        duration=t_phase3,
        thrust=F_phase3,
        throttle=1.0,
        m_dry=m_after_phase3,
        m_propellant=prop_phase3,
        Isp=Isp_phase3,

        duration_bounds=(None, None),
        thrust_bounds=(None, None),
        throttle_bounds=(None, None),
        m_dry_bounds=(None, None),
        m_propellant_bounds=(None, None),
        Isp_bounds=(None, None),

        use_atmosphere=True,
        q_heat_constraint=False,
        q_dyn_constraint=False,
        g_load_constraint=False,

        nose_radius=1e6,
        CD=0.5,
        S=4 * 3.14,
    )

    phase4 = PhaseConfig(
        name='phase_4',

        fix_duration=False,
        fix_thrust=True,
        fix_throttle=True,
        fix_m_dry=True,
        fix_m_propellant=True,
        fix_Isp=True,

        duration=t_phase4,
        thrust=F_phase4,
        throttle=1.0,
        m_dry=STAGE2_DRY,
        m_propellant=STAGE2_PROP,
        Isp=Isp_phase4,

        duration_bounds=(1., 1.1 * t_phase4),
        thrust_bounds=(None, None),
        throttle_bounds=(None, None),
        m_dry_bounds=(None, None),
        m_propellant_bounds=(None, None),
        Isp_bounds=(None, None),

        use_atmosphere=True,
        q_heat_constraint=False,
        q_dyn_constraint=False,
        g_load_constraint=False,

        nose_radius=1e6,
        CD=0.5,
        S=4 * 3.14,
    )

    return [phase1, phase2, phase3, phase4]


def run_delta3_gto(optimize_design: bool = True):
    phases = make_delta3_phase_configs()

    mass_drops = [
        m_drop_after_1,  # после фазы 1: 6 сухих ускорителей
        m_drop_after_2,  # после фазы 2: 3 сухих ускорителя
        m_drop_after_3,  # после фазы 3: сухая 1 ступень
        0.0,  # после фазы 4: ничего (последняя)
    ]

    a = 24_500_000
    e = 0.73
    i = 28.5
    w = 130.5

    return run_multi_stage(
        phases=phases,
        launch_lat_deg=28.5,
        launch_lon_deg=-80.5,
        launch_alt=0.0,
        objective='max_final_mass',
        target_a=a,
        target_e=e,
        target_inc_deg=i,
        target_arg_periapsis_deg=w,
        optimizer_tol=1.0e-4,
        optimizer_max_iter=1000,
        simulate=True,
    )


def print_delta3_summary():
    print('=' * 70)
    print('Delta III mission summary')
    print('=' * 70)
    print(f'Total initial mass:        {M_INITIAL:>12,.1f} kg')
    print(f'  9 boosters total:        {N_BOOSTERS * BOOSTER_TOTAL:>12,.1f} kg')
    print(f'  Stage 1:                 {STAGE1_TOTAL:>12,.1f} kg')
    print(f'  Stage 2:                 {STAGE2_TOTAL:>12,.1f} kg')
    print(f'  Payload:                 {PAYLOAD:>12,.1f} kg')
    print()

    print('Phase 1 (1 stage + 6 boosters)')
    print(f'  Thrust:       {F_phase1:>12,.0f} N')
    print(f'  Isp:          {Isp_phase1:>12.1f} s')
    print(f'  Duration:     {t_phase1:>12.1f} s')
    print(f'  m_initial:    {M_INITIAL:>12,.1f} kg')
    print(f'  m_final:      {m_after_phase1:>12,.1f} kg')
    print(f'  Drop after:   {m_drop_after_1:>12,.1f} kg (6 boosters dry)')
    print()

    print('Phase 2 (1 stage + 3 boosters)')
    print(f'  Thrust:       {F_phase2:>12,.0f} N')
    print(f'  Isp:          {Isp_phase2:>12.1f} s')
    print(f'  Duration:     {t_phase2:>12.1f} s')
    print(f'  m_initial:    {m_initial_p2:>12,.1f} kg')
    print(f'  m_final:      {m_after_phase2:>12,.1f} kg')
    print(f'  Drop after:   {m_drop_after_2:>12,.1f} kg (3 boosters dry)')
    print()

    print('Phase 3 (1 stage alone, until burnout)')
    print(f'  Duration:     {t_phase3:>12.1f} s')
    print(f'  m_initial:    {m_initial_p3:>12,.1f} kg')
    print(f'  m_final:      {m_after_phase3:>12,.1f} kg')
    print(f'  Drop after:   {m_drop_after_3:>12,.1f} kg (stage 1 dry)')
    print()

    print('Phase 4 (stage 2, free duration → GTO)')
    print(f'  Thrust:       {F_phase4:>12,.0f} N')
    print(f'  Isp:          {Isp_phase4:>12.1f} s')
    print(f'  m_initial:    {m_initial_p4:>12,.1f} kg')
    print(f'  m_propellant: {STAGE2_PROP:>12,.1f} kg')
    print(f'  m_dry (min):  {m_dry_phase4:>12,.1f} kg (stage 2 dry + payload)')
    print('=' * 70)


print_delta3_summary()
p, sol_db, sim_db = run_delta3_gto()
print(f'\nSolution:   {sol_db}')
print(f'Simulation: {sim_db}')

phase_names = ('phase1', 'phase2', 'phase3', 'phase4')

plot_eci_trajectory_3d(Path(sol_db), phase_names=phase_names)

plot_eci_trajectory_zoomed(Path(sol_db), phase_names=phase_names)

plot_multi_stage(sol_db, sim_db, phase_names=phase_names)

phase_configs = make_delta3_phase_configs()
mass_drops = [m_drop_after_1, m_drop_after_2, m_drop_after_3, 0.0]

print_design_results(p, phase_configs, mass_drops)
