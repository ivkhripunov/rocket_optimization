from typing import List, Optional
import numpy as np
import openmdao.api as om
import dymos as dm

from src.phase_config import PhaseConfig
from src.stage_phase import build_stage_phase
from src.stage_ode import EARTH_MU
from src.frame_converter import EARTH_RAD, EARTH_OMEGA, geographic_to_cartesian
from src.target_orbit import TargetOrbit, apply_target_orbit


def run_multi_stage(
        phases: List[PhaseConfig],
        launch_lat_deg: float,
        launch_lon_deg: float,
        launch_alt: float,
        objective: str,
        target_orbit: TargetOrbit,
        optimizer_tol: float = 1.0e-4,
        optimizer_max_iter: int = 1000,
        simulate: bool = True,
        restart_db: str = None,
):
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    # =========================================================
    # Сборка фаз
    # =========================================================
    phase_objs = []
    for i, cfg in enumerate(phases):
        is_first = (i == 0)

        phase = build_stage_phase(
            cfg,
            is_first_phase=is_first,
        )
        traj.add_phase(cfg.name, phase)
        phase_objs.append(phase)

    # =========================================================
    # Линковка фаз
    # =========================================================
    for i in range(len(phases) - 1):
        a_name = phases[i].name
        b_name = phases[i + 1].name

        traj.link_phases([a_name, b_name],
                         vars=['time',
                               'rx', 'ry', 'rz',
                               'vx', 'vy', 'vz',
                               'dir_x', 'dir_y', 'dir_z'])

        traj.add_linkage_constraint(
            phase_a=a_name, phase_b=b_name,
            var_a='m', var_b='m',
            loc_a='final', loc_b='initial',
            lower=0.0,
            ref=1.0e3,
        )

    # =========================================================
    # Целевая орбита (на последней фазе)
    # =========================================================
    first_phase = phase_objs[0]
    last_phase = phase_objs[-1]

    apply_target_orbit(last_phase, target_orbit)

    # =========================================================
    # Objective
    # =========================================================
    if objective == 'max_final_mass':
        last_phase.add_objective('m', loc='final', ref=-phases[-1].m_dry)
    elif objective == 'min_initial_mass':
        m_initial_estimate = phases[0].m_dry + phases[0].m_propellant
        first_phase.add_objective('m', loc='initial', ref=m_initial_estimate)
    else:
        raise ValueError(f'Неизвестный objective: {objective}')

    # =========================================================
    # Driver
    # =========================================================
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['invalid_desvar_behavior'] = 'ignore'
    p.driver.opt_settings['tol'] = optimizer_tol
    p.driver.opt_settings['max_iter'] = optimizer_max_iter
    p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=False)

    # =========================================================
    # Начальные условия + приближения
    # =========================================================
    lat0 = np.deg2rad(launch_lat_deg)
    lon0 = np.deg2rad(launch_lon_deg)

    x0, y0, z0 = geographic_to_cartesian(lat0, lon0, launch_alt)
    r0 = np.array([x0, y0, z0])
    omega_vec = np.array([0.0, 0.0, EARTH_OMEGA])
    v0 = np.cross(omega_vec, r0)
    zenith0 = r0 / np.linalg.norm(r0)

    east_eci = np.array([-np.sin(lon0), np.cos(lon0), 0.0])

    # Грубая оценка финальной точки
    rf_mag = target_orbit.a if target_orbit.a is not None else EARTH_RAD + 200_000.0
    rf_guess = zenith0 * rf_mag
    vf_speed = float(np.sqrt(EARTH_MU / rf_mag))
    vf_guess = vf_speed * east_eci

    # Накопление масс по фазам (для приближений)
    cumulative_initial_masses = [p.m_dry + p.m_propellant for p in phases]

    cumulative_t_start = [0.0]
    for cfg in phases[:-1]:
        last_t = cumulative_t_start[-1]
        dur = cfg.duration if cfg.fix_duration else 300.0
        cumulative_t_start.append(last_t + dur)

    # =========================================================
    # Заполнение приближений по фазам
    # =========================================================
    n_phases = len(phases)
    for i, (cfg, phase) in enumerate(zip(phases, phase_objs)):
        # Линейная интерполяция от старта к целевой точке через все фазы
        alpha_s = i / n_phases
        alpha_e = (i + 1) / n_phases

        r_s = r0 + alpha_s * (rf_guess - r0)
        r_e = r0 + alpha_e * (rf_guess - r0)
        v_s = v0 + alpha_s * (vf_guess - v0)
        v_e = v0 + alpha_e * (vf_guess - v0)

        duration_init = cfg.duration if cfg.fix_duration else 300.0
        t_initial = cumulative_t_start[i]

        phase.set_time_val(initial=t_initial, duration=duration_init)

        phase.set_state_val('rx', [r_s[0], r_e[0]])
        phase.set_state_val('ry', [r_s[1], r_e[1]])
        phase.set_state_val('rz', [r_s[2], r_e[2]])
        phase.set_state_val('vx', [v_s[0], v_e[0]])
        phase.set_state_val('vy', [v_s[1], v_e[1]])
        phase.set_state_val('vz', [v_s[2], v_e[2]])

        m_initial_phase = cumulative_initial_masses[i]
        m_final_phase = m_initial_phase - cfg.m_propellant
        phase.set_state_val('m', [m_initial_phase, m_final_phase])

        d_s = (1 - alpha_s) * zenith0 + alpha_s * east_eci
        d_e = (1 - alpha_e) * zenith0 + alpha_e * east_eci
        d_s /= np.linalg.norm(d_s)
        d_e /= np.linalg.norm(d_e)

        phase.set_control_val('dir_x', [d_s[0], d_e[0]])
        phase.set_control_val('dir_y', [d_s[1], d_e[1]])
        phase.set_control_val('dir_z', [d_s[2], d_e[2]])

        if not cfg.fix_throttle:
            phase.set_control_val('throttle', [1.0, 1.0])

    # =========================================================
    # Старт строго в зенит (для первой фазы)
    # =========================================================
    first_phase.add_boundary_constraint('dir_x', loc='initial', equals=zenith0[0])
    first_phase.add_boundary_constraint('dir_y', loc='initial', equals=zenith0[1])
    first_phase.add_boundary_constraint('dir_z', loc='initial', equals=zenith0[2])

    # =========================================================
    # Запуск
    # =========================================================
    dm.run_problem(p, simulate=simulate, restart=restart_db)

    sol_db = p.get_outputs_dir() / 'dymos_solution.db'
    if simulate and traj.sim_prob is not None:
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'
    else:
        sim_db = sol_db.parent / 'dymos_simulation.db'

    return p, sol_db, sim_db
