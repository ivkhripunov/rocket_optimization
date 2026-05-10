import dymos as dm
from src.stage_ode import StageODE
from src.phase_config import PhaseConfig
from src.frame_converter import EARTH_RAD


def build_stage_phase(
        config: PhaseConfig,
        is_first_phase: bool = True,
        transcription=None,
        optimize_design: bool = True,
        optimize_engine: bool = True,
        duration_bounds: tuple = (50.0, 1500.0),
        duration_ref: float = 300.0,
) -> dm.Phase:

    if transcription is None:
        transcription = dm.GaussLobatto(
            num_segments=config.num_segments,
            order=config.order,
            compressed=True,
        )

    phase = dm.Phase(
        ode_class=StageODE,
        ode_init_kwargs={
            'CD': config.CD,
            'S': config.S,
            'nose_radius': config.nose_radius,
            'use_atmosphere': config.use_atmosphere,
            'rho_ref': config.rho_ref,
            'h_scale': config.h_scale,
        },
        transcription=transcription,
    )

    # =========================================================
    # Параметры двигателя и конструкции
    # =========================================================
    phase.add_parameter(
        'thrust_max', units='N',
        val=config.thrust_max,
        opt=optimize_engine,
        lower=config.thrust_max_bounds[0],
        upper=config.thrust_max_bounds[1],
        ref=1.0e6,
    )
    phase.add_parameter(
        'Isp', units='s',
        val=config.Isp,
        opt=optimize_engine,
        lower=config.Isp_bounds[0],
        upper=config.Isp_bounds[1],
        ref=300.0,
    )
    phase.add_parameter(
        'm_dry', units='kg',
        val=config.m_dry,
        opt=optimize_design,
        lower=config.m_dry_bounds[0],
        upper=config.m_dry_bounds[1],
        ref=1.0e3,
    )
    phase.add_parameter(
        'm_propellant', units='kg',
        val=config.m_propellant,
        opt=optimize_design,
        lower=config.m_propellant_bounds[0],
        upper=config.m_propellant_bounds[1],
        ref=1.0e5,
    )

    # =========================================================
    # Время — фиксированное или свободное
    # =========================================================
    if config.fix_duration:
        phase.set_time_options(
            fix_initial=is_first_phase,
            fix_duration=True,
            duration_val=config.duration_value,
            duration_ref=max(config.duration_value, 1.0),
            units='s',
        )
    else:
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

    if optimize_design:
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
        phase.add_state('m', rate_source='mdot',
                        fix_initial=is_first_phase,
                        lower=config.m_min(),
                        units='kg', ref=1.0e5, defect_ref=1.0e3)

    # =========================================================
    # Управление направлением (всегда оптимизируется)
    # =========================================================
    for n in ('dir_x', 'dir_y', 'dir_z'):
        phase.add_control(n, opt=True, lower=-1.0, upper=1.0,
                          continuity=True, rate_continuity=True)

    # =========================================================
    # Throttle — control или фиксированный параметр
    # =========================================================
    if config.optimize_throttle:
        phase.add_control('throttle', opt=True,
                          lower=0.0, upper=1.0,
                          continuity=True, rate_continuity=True)
    else:
        phase.add_parameter('throttle',
                            val=config.throttle_default,
                            opt=False)

    # =========================================================
    # Путевые ограничения
    # =========================================================
    phase.add_path_constraint('dir_norm_sq', equals=1.0, ref=1.0)
    phase.add_path_constraint('h', lower=-100.0)
    phase.add_path_constraint('q_heat', upper=config.q_heat_max,
                              ref=config.q_heat_max)
    phase.add_path_constraint('q_dyn', upper=config.q_dyn_max,
                              ref=config.q_dyn_max)
    phase.add_path_constraint('g_load', upper=config.g_load_max,
                              ref=config.g_load_max)

    # =========================================================
    # Диагностика
    # =========================================================
    for n in ('r_mag', 'v_mag', 'v_radial',
              'dir_norm_sq', 'h', 'thrust_actual',
              'q_heat', 'q_dyn', 'g_load',
              'orbit_a', 'orbit_e', 'orbit_inc'):
        phase.add_timeseries_output(n)

    # =========================================================
    # Уточнение сетки
    # =========================================================
    phase.set_refine_options(
        refine=config.refine,
        tol=config.refine_tol,
        min_order=config.refine_min_order,
        max_order=config.refine_max_order,
        smoothness_factor=config.refine_smoothness,
    )

    return phase