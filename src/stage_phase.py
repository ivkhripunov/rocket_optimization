import dymos as dm
from src.stage_ode import StageODE
from src.phase_config import PhaseConfig
from src.frame_converter import EARTH_RAD


def build_stage_phase(
        config: PhaseConfig,
        is_first_phase: bool) -> dm.Phase:
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
    # Параметры
    # =========================================================

    if config.fix_duration:
        phase.set_time_options(
            fix_initial=is_first_phase,
            fix_duration=True,
            duration_val=config.duration,
            duration_ref=config.duration,
            units='s',
        )
    else:
        phase.set_time_options(
            fix_initial=is_first_phase,
            fix_duration=False,
            duration_val=config.duration,
            duration_bounds=config.duration_bounds,
            duration_ref=config.duration,
            units='s',
        )

    if config.fix_thrust:
        phase.add_parameter('thrust', units='N',
                            val=config.thrust, opt=False, ref=1e6)
    else:
        phase.add_parameter('thrust', units='N',
                            val=config.thrust, opt=True,
                            lower=config.thrust_bounds[0],
                            upper=config.thrust_bounds[1],
                            ref=1e6)

    if config.fix_throttle:
        phase.add_parameter('throttle',
                            val=config.throttle, opt=False)
    else:
        phase.add_control('throttle',
                          val=config.throttle, opt=True,
                          lower=config.throttle_bounds[0],
                          upper=config.throttle_bounds[1],
                          ref=1.0,
                          continuity=True, rate_continuity=True)

    if config.fix_m_dry:
        phase.add_parameter('m_dry', units='kg',
                            val=config.m_dry, opt=False, ref=1e3)
    else:
        phase.add_parameter('m_dry', units='kg',
                            val=config.m_dry, opt=True,
                            lower=config.m_dry_bounds[0],
                            upper=config.m_dry_bounds[1],
                            ref=1e3)

    if config.fix_m_propellant:
        phase.add_parameter('m_propellant', units='kg',
                            val=config.m_propellant, opt=False, ref=1e5)
    else:
        phase.add_parameter('m_propellant', units='kg',
                            val=config.m_propellant, opt=True,
                            lower=config.m_propellant_bounds[0],
                            upper=config.m_propellant_bounds[1],
                            ref=1e5)

    if config.fix_Isp:
        phase.add_parameter('Isp', units='s',
                            val=config.Isp, opt=False, ref=300.0)
    else:
        phase.add_parameter('Isp', units='s',
                            val=config.Isp, opt=True,
                            lower=config.Isp_bounds[0],
                            upper=config.Isp_bounds[1],
                            ref=300.0)

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

    optimize_design = not config.fix_m_dry or not config.fix_m_propellant
    free_initial_m = optimize_design or not is_first_phase

    phase.add_state('m', rate_source='mdot',
                    fix_initial=not free_initial_m,
                    units='kg', ref=1.0e5, defect_ref=1.0e3)

    # =========================================================
    # Управление направлением
    # =========================================================
    for n in ('dir_x', 'dir_y', 'dir_z'):
        phase.add_control(n, opt=True, lower=-1.0, upper=1.0,
                          continuity=True, rate_continuity=True)

    # =========================================================
    # Путевые ограничения
    # =========================================================
    phase.add_path_constraint('dir_norm_sq', equals=1.0, ref=1.0)
    phase.add_path_constraint('h', lower=-10.0)
    phase.add_path_constraint('orbit_e', upper=1.01)
    phase.add_path_constraint('orbit_a', lower=0.)

    phase.add_path_constraint('m_excess = m - m_dry', lower=0.0, ref=1.0e3)
    phase.add_boundary_constraint('m_init_check = m - m_dry - m_propellant',
                                  loc='initial', equals=0.0, ref=1.0e3)

    if config.q_heat_constraint:
        phase.add_path_constraint('q_heat', upper=config.q_heat_max)

    if config.q_dyn_constraint:
        phase.add_path_constraint('q_dyn', upper=config.q_dyn_max)

    if config.g_load_constraint:
        phase.add_path_constraint('g_load', upper=config.g_load_max)

    # =========================================================
    # Диагностика
    # =========================================================
    for n in ('r_mag', 'v_mag', 'v_radial',
              'dir_norm_sq', 'h', 'thrust_actual',
              'q_heat', 'q_dyn', 'g_load',
              'orbit_a', 'orbit_e', 'orbit_inc',
              'orbit_raan', 'orbit_arg_periapsis'):
        phase.add_timeseries_output(n)

    return phase
