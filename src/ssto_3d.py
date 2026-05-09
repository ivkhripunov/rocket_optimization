import openmdao.api as om
import dymos as dm
import numpy as np
import jax.numpy as jnp

from src.frame_converter import EARTH_RAD, EARTH_OMEGA, geographic_to_cartesian, ecef_to_eci

EARTH_MU = 3.986004418e14
G0 = 9.80665


class SSTO3D(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

        # Аэродинамика
        self.options.declare('CD', types=float, default=0.5)
        self.options.declare('S', types=float, default=7.069)

        # Атмосфера
        self.options.declare('rho_ref', types=float, default=0.)
        self.options.declare('h_scale', types=float, default=8.44e3)

        # Двигатель: МАКСИМАЛЬНАЯ тяга и Isp
        self.options.declare('thrust_max', types=float, default=2.1e6)
        self.options.declare('Isp', types=float, default=265.2)

    def setup(self):
        nn = self.options['num_nodes']

        # ----- inputs: состояния -----
        for n in ('rx', 'ry', 'rz'):
            self.add_input(n, val=EARTH_RAD * np.zeros(nn), units='m')
        for n in ('vx', 'vy', 'vz'):
            self.add_input(n, val=np.zeros(nn), units='m/s')

        self.add_input('m', val=1.0e5 * np.ones(nn), units='kg')

        # ----- inputs: управления -----
        # вектор направления тяги (в инерциальной СК)
        self.add_input('dir_x', val=np.ones(nn))
        self.add_input('dir_y', val=np.zeros(nn))
        self.add_input('dir_z', val=np.zeros(nn))
        # дроссель — относительный уровень тяги в [0, 1]
        self.add_input('throttle', val=np.ones(nn))

        # ----- outputs: производные состояний -----
        for n in ('rxdot', 'rydot', 'rzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s')
        for n in ('vxdot', 'vydot', 'vzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), units='kg/s')

        # ----- outputs: диагностика -----
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
                       throttle):

        CDA = self.options['CD'] * self.options['S']
        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']
        F_T_max = self.options['thrust_max']
        Isp = self.options['Isp']

        # ---- фактическая тяга = max × throttle ----
        F_T = F_T_max * throttle

        # ---- нормировка вектора направления тяги ----
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

        # ---- атмосфера и относительная скорость ----
        rho = rho_ref * jnp.exp(-h / h_scale)
        v_atm_x = -EARTH_OMEGA * ry
        v_atm_y = EARTH_OMEGA * rx

        vrx = vx - v_atm_x
        vry = vy - v_atm_y
        vrz = vz
        v_rel = jnp.sqrt(vrx * vrx + vry * vry + vrz * vrz + 1.0)

        a_drag_x = -0.5 * CDA * rho * v_rel * vrx / m
        a_drag_y = -0.5 * CDA * rho * v_rel * vry / m
        a_drag_z = -0.5 * CDA * rho * v_rel * vrz / m
        #
        # ---- ускорение от тяги (теперь зависит от throttle через F_T) ----
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
        # массовый расход теперь переменный по узлам (зависит от throttle)
        mdot = -F_T / (Isp * G0)

        # ---- диагностика ----
        r_mag = r
        v_mag = jnp.sqrt(vx * vx + vy * vy + vz * vz)
        v_radial = (rx * vx + ry * vy + rz * vz) / r
        dir_norm_sq = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z
        thrust_actual = F_T

        return (rxdot, rydot, rzdot,
                vxdot, vydot, vzdot,
                mdot,
                r_mag, v_mag, v_radial, dir_norm_sq, h, thrust_actual)


def run_ssto_3d(launch_lat_deg=0, launch_lon_deg=0, launch_alt=0.0,
                target_alt=200_000.0,
                m0=117_000.0, mf_min=1.0,
                thrust_max_N=2.1e6, Isp_s=265.2,
                num_segments=5, order=3):
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    phase = dm.Phase(
        ode_class=SSTO3D,
        ode_init_kwargs={'thrust_max': thrust_max_N, 'Isp': Isp_s},
        transcription=dm.GaussLobatto(num_segments=num_segments,
                                      order=order, compressed=True),
    )

    traj.add_phase('phase0', phase)

    ref_duration = 200

    phase.set_time_options(fix_initial=True,
                           duration_bounds=(50.0, 400.0),
                           duration_ref=ref_duration,
                           units='s')

    # ---- состояния ----
    for n in ('rx', 'ry', 'rz'):
        phase.add_state(n, rate_source=n + 'dot', fix_initial=True,
                        units='m', ref=EARTH_RAD, defect_ref=1.0e5)
    for n in ('vx', 'vy', 'vz'):
        phase.add_state(n, rate_source=n + 'dot', fix_initial=True,
                        units='m/s', ref=1.0e3, defect_ref=1.0e3)
    phase.add_state('m', rate_source='mdot', fix_initial=True,
                    lower=mf_min, units='kg',
                    ref=1.0e5, defect_ref=1.0e3)

    # ---- управления ----
    for n in ('dir_x', 'dir_y', 'dir_z'):
        phase.add_control(n, opt=True, lower=-1.0, upper=1.0,
                          continuity=True, rate_continuity=True)

    phase.add_control('throttle', opt=True,
                      lower=0.0, upper=1.0,
                      continuity=True, rate_continuity=True,
                      targets=['throttle'])

    # ---- констрейнты ----
    phase.add_path_constraint('dir_norm_sq', equals=1.0, ref=1.0)

    phase.add_path_constraint('h', lower=0.)

    target_radius = EARTH_RAD + target_alt
    target_speed = float(np.sqrt(EARTH_MU / target_radius))

    phase.add_boundary_constraint('r_mag', loc='final',
                                  equals=target_radius, ref=target_radius)
    phase.add_boundary_constraint('v_mag', loc='final',
                                  equals=target_speed, ref=target_speed)
    phase.add_boundary_constraint('v_radial', loc='final', lower=-10.0, upper=10.0)

    # ---- диагностику — в timeseries ----
    for n in ('r_mag', 'v_mag', 'v_radial', 'dir_norm_sq', 'h', 'thrust_actual'):
        phase.add_timeseries_output(n)

    # =========================================================
    # ЦЕЛЬ: максимизировать конечную массу (= минимум расхода)
    # scaler=-1 потому что Dymos минимизирует, а нам нужен максимум
    # =========================================================
    phase.add_objective('m', loc='final', ref=-m0)

    # ---- driver ----
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['tol'] = 1e-4
    p.driver.opt_settings['max_iter'] = 500
    p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=False)

    # =========================================================
    # Начальные условия
    # =========================================================
    lat0 = np.deg2rad(launch_lat_deg)
    lon0 = np.deg2rad(launch_lon_deg)

    x0_ecef, y0_ecef, z0_ecef = geographic_to_cartesian(lat0, lon0, launch_alt)
    r0_eci = ecef_to_eci(x0_ecef, y0_ecef, z0_ecef, 0)
    v0_eci = np.cross([0.0, 0.0, EARTH_OMEGA], r0_eci)

    rf_eci = ecef_to_eci(x0_ecef, y0_ecef, z0_ecef, ref_duration)
    east_eci = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    vf_eci = target_speed * east_eci

    zenith0 = r0_eci / np.linalg.norm(r0_eci)

    phase.set_time_val(initial=0.0, duration=ref_duration, units='s')

    phase.set_state_val('rx', [r0_eci[0], rf_eci[0]])
    phase.set_state_val('ry', [r0_eci[1], rf_eci[1]])
    phase.set_state_val('rz', [r0_eci[2], rf_eci[2]])
    phase.set_state_val('vx', [v0_eci[0], vf_eci[0]])
    phase.set_state_val('vy', [v0_eci[1], vf_eci[1]])
    phase.set_state_val('vz', [v0_eci[2], vf_eci[2]])

    phase.set_state_val('m', [m0, m0 * 0.1])

    phase.set_control_val('dir_x', [zenith0[0], east_eci[0]])
    phase.set_control_val('dir_y', [zenith0[1], east_eci[1]])
    phase.set_control_val('dir_z', [zenith0[2], east_eci[2]])

    phase.set_control_val('throttle', [1.0, 1.0])

    # phase.set_refine_options(
    #     refine=True,
    #     tol=1e-3,
    #     min_order=3,
    #     max_order=3,
    #     smoothness_factor=2.0,
    # )

    dm.run_problem(
        p, simulate=True,
        # refine_method='hp',
        # refine_iteration_limit=2,
    )

    sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

    return p, sim_db
