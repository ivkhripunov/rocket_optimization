"""
Один модуль ступени ракеты для оптимизации траектории в Dymos.

Архитектура подготовлена к двум расширениям:

1. Многоступенчатая ракета — каждая ступень = одна фаза. Создаётся
   через `build_stage_phase(config, is_first_phase=...)`. Линковка между
   фазами (положение, скорость непрерывны, масса с разрывом на сбросе
   ступени) делается в коде верхнего уровня через `traj.link_phases(...)`.

2. Совместная оптимизация конструкции и траектории — параметры ступени
   (тяга, Isp, сухая масса, масса топлива) могут стать design-переменными
   при `optimize_design=True`.

Состояние:    (rx, ry, rz, vx, vy, vz, m) в инерциальной СК (ECI).
Управления:   (dir_x, dir_y, dir_z, throttle).
Параметры:    thrust_max, Isp, m_dry, m_propellant (могут быть оптимизируемыми).
"""

from dataclasses import dataclass
from typing import Optional

import openmdao.api as om
import dymos as dm
import numpy as np
import jax.numpy as jnp

from src.frame_converter import (
    EARTH_RAD, EARTH_OMEGA,
    geographic_to_cartesian, ecef_to_eci,
)

EARTH_MU = 3.986004418e14    # м³/с², гравитационный параметр Земли
G0 = 9.80665                 # м/с², стандартное ускорение для Isp


# =============================================================================
# Конфигурация ступени
# =============================================================================
@dataclass
class StageConfig:
    """
    Параметры одной ступени ракеты.

    Все массы/тяги — этой конкретной ступени, без учёта вышестоящих ступеней.
    Полную «начальную массу» считает уже код верхнего уровня (сумма данной
    ступени и всех вышестоящих).
    """
    # Имя фазы. Дефолт 'phase0' для совместимости с прежним visualize.py.
    name: str = 'phase0'

    # ===== Двигатель =====
    thrust_max: float = 2.1e6        # Н, максимальная тяга
    Isp: float = 265.2               # с, удельный импульс

    # ===== Масса =====
    m_dry: float = 1_000.0           # кг, сухая масса этой ступени
    m_propellant: float = 116_000.0  # кг, топливо в этой ступени

    # ===== Аэродинамика =====
    CD: float = 0.5
    S: float = 7.069                 # м², характерная площадь

    # ===== Атмосфера =====
    use_atmosphere: bool = False
    rho_ref: float = 1.225           # кг/м³, плотность на h=0
    h_scale: float = 8.44e3          # м, масштаб высоты

    # ===== Сетка =====
    num_segments: int = 15
    order: int = 3

    # ===== Уточнение сетки (grid refinement) =====
    refine: bool = False
    refine_method: str = 'hp'        # 'hp' или 'ph'
    refine_iter_limit: int = 3
    refine_tol: float = 1.0e-4
    refine_min_order: int = 3
    refine_max_order: int = 8
    refine_smoothness: float = 1.5

    # ===== Границы для design-оптимизации =====
    thrust_max_bounds: tuple = (1.0e5, 1.0e7)
    Isp_bounds: tuple = (200.0, 450.0)
    m_dry_bounds: tuple = (100.0, 1.0e5)
    m_propellant_bounds: tuple = (1.0e3, 1.0e6)

    @property
    def m_total(self) -> float:
        """Полная масса этой ступени с топливом (без вышестоящих)."""
        return self.m_dry + self.m_propellant

    @property
    def m_min(self) -> float:
        """Минимальная допустимая масса (после полного сгорания топлива)."""
        return self.m_dry


# =============================================================================
# ODE одной ступени
# =============================================================================
class RocketStageODE(om.JaxExplicitComponent):
    """
    Динамика одной ступени ракеты в инерциальной СК.

    Атмосфера может быть выключена (`use_atmosphere=False`) — для верхних
    ступеней / вакуумных режимов сопротивление и плотность игнорируются.

    Параметры m_dry / m_propellant в саму физику не входят — они нужны
    только на уровне постановки (для констрейнтов и начальной массы).
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare('CD', types=float, default=0.5)
        self.options.declare('S',  types=float, default=7.069)

        self.options.declare('use_atmosphere', types=bool, default=True)
        self.options.declare('rho_ref', types=float, default=1.225)
        self.options.declare('h_scale', types=float, default=8.44e3)

    def setup(self):
        nn = self.options['num_nodes']

        # ---- inputs: состояния ----
        for n in ('rx', 'ry', 'rz'):
            self.add_input(n, val=EARTH_RAD * np.ones(nn), units='m')
        for n in ('vx', 'vy', 'vz'):
            self.add_input(n, val=np.zeros(nn), units='m/s')
        self.add_input('m', val=1.0e5 * np.ones(nn), units='kg')

        # ---- inputs: управления ----
        self.add_input('dir_x', val=np.ones(nn))
        self.add_input('dir_y', val=np.zeros(nn))
        self.add_input('dir_z', val=np.zeros(nn))
        self.add_input('throttle', val=np.ones(nn))

        # ---- inputs: параметры ступени (могут быть design vars) ----
        self.add_input('thrust_max', val=2.1e6 * np.ones(nn), units='N')
        self.add_input('Isp', val=265.2 * np.ones(nn), units='s')

        # ---- outputs: производные состояний ----
        for n in ('rxdot', 'rydot', 'rzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s')
        for n in ('vxdot', 'vydot', 'vzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), units='kg/s')

        # ---- outputs: диагностика ----
        self.add_output('r_mag', val=np.zeros(nn), units='m')
        self.add_output('v_mag', val=np.zeros(nn), units='m/s')
        self.add_output('v_radial', val=np.zeros(nn), units='m/s')
        self.add_output('dir_norm_sq', val=np.ones(nn))
        self.add_output('h', val=np.zeros(nn), units='m')
        self.add_output('thrust_actual', val=np.zeros(nn), units='N')

    def compute_primal(self,
                       rx, ry, rz,
                       vx, vy, vz,
                       m,
                       dir_x, dir_y, dir_z,
                       throttle,
                       thrust_max, Isp):

        CDA            = self.options['CD'] * self.options['S']
        use_atmosphere = self.options['use_atmosphere']
        rho_ref        = self.options['rho_ref']
        h_scale        = self.options['h_scale']

        F_T = thrust_max * throttle

        # ---- нормировка вектора направления ----
        dir_norm = jnp.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z + 1e-12)
        dx = dir_x / dir_norm
        dy = dir_y / dir_norm
        dz = dir_z / dir_norm

        # ---- геоцентрическое расстояние и высота ----
        r = jnp.sqrt(rx * rx + ry * ry + rz * rz)
        h = r - EARTH_RAD

        # ---- центральная гравитация ----
        inv_r3 = 1.0 / (r * r * r)
        a_grav_x = -EARTH_MU * rx * inv_r3
        a_grav_y = -EARTH_MU * ry * inv_r3
        a_grav_z = -EARTH_MU * rz * inv_r3

        # ---- атмосферное сопротивление (опционально) ----
        if use_atmosphere:
            rho = rho_ref * jnp.exp(-h / h_scale)

            v_atm_x = -EARTH_OMEGA * ry
            v_atm_y =  EARTH_OMEGA * rx
            vrx = vx - v_atm_x
            vry = vy - v_atm_y
            vrz = vz
            v_rel = jnp.sqrt(vrx * vrx + vry * vry + vrz * vrz + 1.0)

            a_drag_x = -0.5 * CDA * rho * v_rel * vrx / m
            a_drag_y = -0.5 * CDA * rho * v_rel * vry / m
            a_drag_z = -0.5 * CDA * rho * v_rel * vrz / m
        else:
            a_drag_x = jnp.zeros_like(vx)
            a_drag_y = jnp.zeros_like(vy)
            a_drag_z = jnp.zeros_like(vz)

        # ---- ускорение от тяги ----
        a_thrust_x = (F_T / m) * dx
        a_thrust_y = (F_T / m) * dy
        a_thrust_z = (F_T / m) * dz

        # ---- производные состояний ----
        rxdot = vx
        rydot = vy
        rzdot = vz
        vxdot = a_grav_x + a_thrust_x + a_drag_x
        vydot = a_grav_y + a_thrust_y + a_drag_y
        vzdot = a_grav_z + a_thrust_z + a_drag_z
        mdot  = -F_T / (Isp * G0)

        # ---- диагностика ----
        r_mag         = r
        v_mag         = jnp.sqrt(vx * vx + vy * vy + vz * vz)
        v_radial      = (rx * vx + ry * vy + rz * vz) / r
        dir_norm_sq   = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z
        thrust_actual = F_T

        return (rxdot, rydot, rzdot,
                vxdot, vydot, vzdot,
                mdot,
                r_mag, v_mag, v_radial, dir_norm_sq, h, thrust_actual)


# =============================================================================
# Сборка фазы из конфигурации ступени
# =============================================================================
def build_stage_phase(
    config: StageConfig,
    *,
    is_first_phase: bool = True,
    transcription=None,
    optimize_design: bool = False,
    duration_bounds: tuple = (50.0, 1500.0),
    duration_ref: float = 300.0,
) -> dm.Phase:
    """
    Собрать `dm.Phase` для одной ступени ракеты.

    Создаёт фазу со всей структурой состояний, управлений, параметров и
    «общестепеневых» path-констрейнтов (h ≥ -100, |dir| = 1, m ≥ m_dry).
    Боундари-констрейнты целевой орбиты и связи между фазами добавляет
    вызывающий код.
    """
    if transcription is None:
        transcription = dm.GaussLobatto(
            num_segments=config.num_segments,
            order=config.order,
            compressed=True,
        )

    phase = dm.Phase(
        ode_class=RocketStageODE,
        ode_init_kwargs={
            'CD':              config.CD,
            'S':               config.S,
            'use_atmosphere':  config.use_atmosphere,
            'rho_ref':         config.rho_ref,
            'h_scale':         config.h_scale,
        },
        transcription=transcription,
    )

    # =========================================================
    # Параметры (могут быть design vars при optimize_design=True)
    # =========================================================
    phase.add_parameter(
        'thrust_max', units='N',
        val=config.thrust_max,
        opt=optimize_design,
        lower=config.thrust_max_bounds[0],
        upper=config.thrust_max_bounds[1],
        ref=1.0e6,
    )
    phase.add_parameter(
        'Isp', units='s',
        val=config.Isp,
        opt=optimize_design,
        lower=config.Isp_bounds[0],
        upper=config.Isp_bounds[1],
        ref=300.0,
    )
    # m_dry и m_propellant не нужны в ODE напрямую — их потребляют только
    # path/boundary-constraint выражения. Поэтому targets=[] (не подключаем).
    phase.add_parameter(
        'm_dry', units='kg',
        val=config.m_dry,
        opt=optimize_design,
        lower=config.m_dry_bounds[0],
        upper=config.m_dry_bounds[1],
        ref=1.0e3,
        targets=[],
    )
    phase.add_parameter(
        'm_propellant', units='kg',
        val=config.m_propellant,
        opt=optimize_design,
        lower=config.m_propellant_bounds[0],
        upper=config.m_propellant_bounds[1],
        ref=1.0e5,
        targets=[],
    )

    # =========================================================
    # Время
    # =========================================================
    phase.set_time_options(
        fix_initial=is_first_phase,
        duration_bounds=duration_bounds,
        duration_ref=duration_ref,
        units='s',
    )

    # =========================================================
    # Состояния
    # =========================================================
    for n in ('rx', 'ry', 'rz'):
        phase.add_state(n, rate_source=n + 'dot',
                        fix_initial=is_first_phase,
                        units='m', ref=EARTH_RAD, defect_ref=1.0e5)
    for n in ('vx', 'vy', 'vz'):
        phase.add_state(n, rate_source=n + 'dot',
                        fix_initial=is_first_phase,
                        units='m/s', ref=1.0e3, defect_ref=1.0e3)

    # Масса: разная стратегия в зависимости от того, являются ли
    # m_dry / m_propellant переменными оптимизации.
    if optimize_design:
        # m_dry — переменная, поэтому нижнюю границу нельзя задать константой.
        # Вместо этого: m свободна, ограничение `m ≥ m_dry` через path-constraint.
        # Начальное значение тоже свободно, фиксируется boundary-constraint
        # `m = m_dry + m_propellant` (только для первой фазы — далее идёт
        # линковка с предыдущей).
        phase.add_state('m', rate_source='mdot',
                        fix_initial=False,
                        units='kg', ref=1.0e5, defect_ref=1.0e3)

        phase.add_path_constraint(
            'm_excess = m - m_dry',
            lower=0.0, ref=1.0e3,
        )

        if is_first_phase:
            phase.add_boundary_constraint(
                'm_init_check = m - m_dry - m_propellant',
                loc='initial', equals=0.0, ref=1.0e3,
            )
    else:
        # Простой случай: m_dry — константа, нижняя граница состояния.
        phase.add_state('m', rate_source='mdot',
                        fix_initial=is_first_phase,
                        lower=config.m_min,
                        units='kg', ref=1.0e5, defect_ref=1.0e3)

    # =========================================================
    # Управления
    # =========================================================
    for n in ('dir_x', 'dir_y', 'dir_z'):
        phase.add_control(n, opt=True, lower=-1.0, upper=1.0,
                          continuity=True, rate_continuity=True)
    phase.add_control('throttle', opt=True,
                      lower=0.0, upper=1.0,
                      continuity=True, rate_continuity=True)

    # =========================================================
    # Path-констрейнты ступени
    # =========================================================
    phase.add_path_constraint('dir_norm_sq', equals=1.0, ref=1.0)
    phase.add_path_constraint('h', lower=-100.0)

    # =========================================================
    # Диагностика в timeseries
    # =========================================================
    for n in ('r_mag', 'v_mag', 'v_radial',
              'dir_norm_sq', 'h', 'thrust_actual'):
        phase.add_timeseries_output(n)

    # =========================================================
    # Уточнение сетки (grid refinement)
    # =========================================================
    if config.refine:
        phase.set_refine_options(
            refine=True,
            tol=config.refine_tol,
            min_order=config.refine_min_order,
            max_order=config.refine_max_order,
            smoothness_factor=config.refine_smoothness,
        )

    return phase


# =============================================================================
# Запуск оптимизации SSTO (одноступенчатая ракета)
# =============================================================================
def run_ssto_3d(
    stage: Optional[StageConfig] = None,
    launch_lat_deg: float = 0.0,
    launch_lon_deg: float = 0.0,
    launch_alt: float = 0.0,
    target_alt: float = 200_000.0,
    duration_guess: float = 400.0,
    optimize_design: bool = False,
    optimizer_tol: float = 1e-4,
    optimizer_max_iter: int = 500,
):
    """
    Решить задачу выведения одноступенчатой ракеты на круговую орбиту.
    Цель: максимум финальной массы.
    """
    if stage is None:
        stage = StageConfig()

    # =========================================================
    # Сборка задачи
    # =========================================================
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    phase = build_stage_phase(
        stage,
        is_first_phase=True,
        optimize_design=optimize_design,
        duration_bounds=(150.0, 800.0),
        duration_ref=duration_guess,
    )
    traj.add_phase(stage.name, phase)

    # ---- Целевая орбита (boundary-констрейнты на ПОСЛЕДНЕЙ фазе) ----
    target_radius = EARTH_RAD + target_alt
    target_speed  = float(np.sqrt(EARTH_MU / target_radius))

    phase.add_boundary_constraint('r_mag',    loc='final',
                                  equals=target_radius, ref=target_radius)
    phase.add_boundary_constraint('v_mag',    loc='final',
                                  equals=target_speed, ref=target_speed)
    phase.add_boundary_constraint('v_radial', loc='final',
                                  lower=-10.0, upper=10.0)

    # ---- Цель: максимум финальной массы ----
    phase.add_objective('m', loc='final', ref=-stage.m_total)

    # ---- Driver ----
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['tol']      = optimizer_tol
    p.driver.opt_settings['max_iter'] = optimizer_max_iter
    p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=False)

    # =========================================================
    # Начальные условия и приближения
    # =========================================================
    lat0 = np.deg2rad(launch_lat_deg)
    lon0 = np.deg2rad(launch_lon_deg)

    x0_ecef, y0_ecef, z0_ecef = geographic_to_cartesian(lat0, lon0, launch_alt)
    r0_eci = np.array([x0_ecef, y0_ecef, z0_ecef])    # ECI ≡ ECEF при t=0

    omega_vec = np.array([0.0, 0.0, EARTH_OMEGA])
    v0_eci = np.cross(omega_vec, r0_eci)

    rf_eci   = ecef_to_eci(x0_ecef, y0_ecef, z0_ecef, duration_guess)
    east_eci = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    vf_eci   = target_speed * east_eci

    zenith0 = r0_eci / np.linalg.norm(r0_eci)

    phase.set_time_val(initial=0.0, duration=duration_guess)

    phase.set_state_val('rx', [r0_eci[0], rf_eci[0]])
    phase.set_state_val('ry', [r0_eci[1], rf_eci[1]])
    phase.set_state_val('rz', [r0_eci[2], rf_eci[2]])
    phase.set_state_val('vx', [v0_eci[0], vf_eci[0]])
    phase.set_state_val('vy', [v0_eci[1], vf_eci[1]])
    phase.set_state_val('vz', [v0_eci[2], vf_eci[2]])
    phase.set_state_val('m',  [stage.m_total, stage.m_dry])

    phase.set_control_val('dir_x', [zenith0[0], east_eci[0]])
    phase.set_control_val('dir_y', [zenith0[1], east_eci[1]])
    phase.set_control_val('dir_z', [zenith0[2], east_eci[2]])
    phase.set_control_val('throttle', [1.0, 1.0])

    # =========================================================
    # Запуск (с уточнением сетки, если включено в конфиге)
    # =========================================================
    if stage.refine:
        refine_method = stage.refine_method
        refine_iter_limit = stage.refine_iter_limit
    else:
        refine_method = 'none'
        refine_iter_limit = 0

    dm.run_problem(
        p, simulate=True,
        refine_method=refine_method,
        refine_iteration_limit=refine_iter_limit,
    )

    sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'
    return p, sim_db