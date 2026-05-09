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
        self.options.declare('rho_ref', types=float, default=1.225)
        self.options.declare('h_scale', types=float, default=8.44e3)

        # Двигатель
        self.options.declare('thrust', types=float, default=2.1e6)
        self.options.declare('Isp', types=float, default=265.2)

    def setup(self):
        nn = self.options['num_nodes']

        # ----- inputs -----
        for n in ('rx', 'ry', 'rz'):
            self.add_input(n, val=np.zeros(nn), units='m')
        for n in ('vx', 'vy', 'vz'):
            self.add_input(n, val=np.zeros(nn), units='m/s')

        self.add_input('m', val=1.0e5 * np.ones(nn), units='kg')

        self.add_input('dir_x', val=np.ones(nn))
        self.add_input('dir_y', val=np.zeros(nn))
        self.add_input('dir_z', val=np.zeros(nn))

        # ----- outputs -----
        for n in ('rxdot', 'rydot', 'rzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s')
        for n in ('vxdot', 'vydot', 'vzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), units='kg/s')

        # ----- outputs: диагностика для граничных условий -----
        self.add_output('r_mag', val=np.zeros(nn), units='m')
        self.add_output('v_mag', val=np.zeros(nn), units='m/s')
        self.add_output('v_radial', val=np.zeros(nn), units='m/s')
        self.add_output('dir_norm_sq', val=np.ones(nn))
        self.add_output('h', val=np.zeros(nn), units='m')

    def compute_primal(self,
                       rx, ry, rz,
                       vx, vy, vz,
                       m,
                       dir_x, dir_y, dir_z):

        CDA = self.options['CD'] * self.options['S']
        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']
        F_T = self.options['thrust']
        Isp = self.options['Isp']

        # ---- нормировка вектора направления тяги ----
        dir_norm = jnp.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z + 1e-12)
        dx = dir_x / dir_norm
        dy = dir_y / dir_norm
        dz = dir_z / dir_norm

        # ---- геоцентрическое расстояние и высота ----
        r = jnp.sqrt(rx * rx + ry * ry + rz * rz)
        h = r - EARTH_RAD

        # ---- гравитация (центральная) ----
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

        # ---- тяга ----
        a_thrust_x = (F_T / m) * dx
        a_thrust_y = (F_T / m) * dy
        a_thrust_z = (F_T / m) * dz

        # ---- производные состояний ----
        rxdot = vx
        rydot = vy
        rzdot = vz
        vxdot = a_grav_x + a_drag_x + a_thrust_x
        vydot = a_grav_y + a_drag_y + a_thrust_y
        vzdot = a_grav_z + a_drag_z + a_thrust_z
        mdot = jnp.full_like(m, -F_T / (Isp * G0))

        # ---- диагностика ----
        r_mag = r
        v_mag = jnp.sqrt(vx * vx + vy * vy + vz * vz)
        v_radial = (rx * vx + ry * vy + rz * vz) / r
        dir_norm_sq = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z

        return (rxdot, rydot, rzdot,
                vxdot, vydot, vzdot,
                mdot,
                r_mag, v_mag, v_radial, dir_norm_sq, h)


def run_ssto_3d(launch_lat_deg=28.5, launch_lon_deg=-80.5, launch_alt=0.0,
                target_alt=200_000.0,
                m0=117_000.0, mf_min=1000.0,
                thrust_N=2.1e6, Isp_s=265.2,
                num_segments=20, order=3,
                duration_guess=200.0):
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    phase = dm.Phase(
        ode_class=SSTO3D,
        ode_init_kwargs={'thrust': thrust_N, 'Isp': Isp_s},
        transcription=dm.GaussLobatto(num_segments=num_segments,
                                      order=order, compressed=True),
    )
    traj.add_phase('phase0', phase)

    ref_duration = 500

    phase.set_time_options(fix_initial=True,
                           duration_bounds=(50.0, 1000.0),
                           duration_ref=ref_duration,
                           units='s')

    for n in ('rx', 'ry', 'rz'):
        phase.add_state(n, rate_source=n + 'dot', fix_initial=True,
                        units='m', ref=EARTH_RAD, defect_ref=1.0e5)
    for n in ('vx', 'vy', 'vz'):
        phase.add_state(n, rate_source=n + 'dot', fix_initial=True,
                        units='m/s', ref=1.0e3, defect_ref=1.0e3)
    phase.add_state('m', rate_source='mdot', fix_initial=True,
                    lower=mf_min, units='kg',
                    ref=1.0e5, defect_ref=1.0e3)

    for n in ('dir_x', 'dir_y', 'dir_z'):
        phase.add_control(n, opt=True, lower=-1.0, upper=1.0,
                          continuity=True, rate_continuity=True)

    phase.add_path_constraint('dir_norm_sq', equals=1.0, ref=1.0)

    target_radius = EARTH_RAD + target_alt
    target_speed = float(np.sqrt(EARTH_MU / target_radius))

    phase.add_boundary_constraint('r_mag', loc='final',
                                  equals=target_radius, ref=target_radius)
    phase.add_boundary_constraint('v_mag', loc='final',
                                  equals=target_speed, ref=target_speed)
    phase.add_boundary_constraint('v_radial', loc='final', upper=100.0)
    
    # ---- диагностику — в timeseries для графиков ----
    for n in ('r_mag', 'v_mag', 'v_radial', 'dir_norm_sq', 'h'):
        phase.add_timeseries_output(n)

    phase.add_objective('time', loc='final', scaler=0.01)

    # ---- driver ----
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['tol'] = 1e-6
    p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=False)

    # =========================================================
    # Начальные условия и приближения
    # =========================================================
    lat0 = np.deg2rad(launch_lat_deg)
    lon0 = np.deg2rad(launch_lon_deg)

    x0_ecef, y0_ecef, z0_ecef = geographic_to_cartesian(lat0, lon0, launch_alt)
    r0_eci = ecef_to_eci(x0_ecef, y0_ecef, z0_ecef, 0)
    v0_eci = np.cross([0.0, 0.0, EARTH_OMEGA], r0_eci)

    # «грубый» финал: над стартом на нужной высоте, скорость на восток
    rf_eci = ecef_to_eci(x0_ecef, y0_ecef, z0_ecef, ref_duration)
    east_eci = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    vf_eci = target_speed * east_eci

    # начальное направление тяги — в зенит
    zenith0 = r0_eci / np.linalg.norm(r0_eci)

    phase.set_time_val(initial=0.0, duration=duration_guess)

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

    # ---- запуск ----
    dm.run_problem(p, simulate=True)

    sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

    return p, sim_db
