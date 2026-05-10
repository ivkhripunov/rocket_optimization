from src.phase_config import PhaseConfig
from src.stage_phase import build_stage_phase

import openmdao.api as om
import dymos as dm
import numpy as np

from src.frame_converter import (
    EARTH_RAD, EARTH_OMEGA,
    geographic_to_cartesian
)

from src.stage_ode import EARTH_MU


def run_single_stage(
        stage: PhaseConfig,
        launch_lat_deg: float,
        launch_lon_deg: float,
        launch_alt: float,
        target_alt: float = 200_000.0,
        duration_guess: float = 400.0,
        optimize_design: bool = False,
        optimize_engine: bool = False,
        optimizer_tol: float = 1e-4,
        optimizer_max_iter: int = 500,
):
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    phase = build_stage_phase(
        stage,
        is_first_phase=True,
        optimize_design=optimize_design,
        optimize_engine=optimize_engine,
        duration_bounds=(150.0, 800.0),
        duration_ref=duration_guess,
    )
    traj.add_phase(stage.name, phase)

    # ---- Целевая орбита ----
    target_radius = EARTH_RAD + target_alt
    target_speed = float(np.sqrt(EARTH_MU / target_radius))

    phase.add_boundary_constraint('r_mag', loc='final',
                                  equals=target_radius, ref=target_radius)
    phase.add_boundary_constraint('v_mag', loc='final',
                                  equals=target_speed, ref=target_speed)
    phase.add_boundary_constraint('v_radial', loc='final',
                                  lower=-10.0, upper=10.0)

    # ---- Целевая функция ----
    phase.add_objective('m', loc='final', ref=-stage.m_total())

    # ---- Настройки солвера ----
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['tol'] = optimizer_tol
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
    r0_eci = np.array([x0_ecef, y0_ecef, z0_ecef])

    omega_vec = np.array([0.0, 0.0, EARTH_OMEGA])
    v0_eci = np.cross(omega_vec, r0_eci)

    rf_eci = r0_eci * target_radius / EARTH_RAD
    east_eci = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    vf_eci = target_speed * east_eci

    zenith0 = r0_eci / np.linalg.norm(r0_eci)

    phase.add_boundary_constraint('dir_x', loc='initial', equals=zenith0[0])
    phase.add_boundary_constraint('dir_y', loc='initial', equals=zenith0[1])
    phase.add_boundary_constraint('dir_z', loc='initial', equals=zenith0[2])

    phase.set_time_val(initial=0.0, duration=duration_guess)

    phase.set_state_val('rx', [r0_eci[0], rf_eci[0]])
    phase.set_state_val('ry', [r0_eci[1], rf_eci[1]])
    phase.set_state_val('rz', [r0_eci[2], rf_eci[2]])
    phase.set_state_val('vx', [v0_eci[0], vf_eci[0]])
    phase.set_state_val('vy', [v0_eci[1], vf_eci[1]])
    phase.set_state_val('vz', [v0_eci[2], vf_eci[2]])
    phase.set_state_val('m', [stage.m_total(), stage.m_dry])

    phase.set_control_val('dir_x', [zenith0[0], east_eci[0]])
    phase.set_control_val('dir_y', [zenith0[1], east_eci[1]])
    phase.set_control_val('dir_z', [zenith0[2], east_eci[2]])
    phase.set_control_val('throttle', [1.0, 1.0])

    # =========================================================
    # Запуск расчета
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
