import numpy as np
from jax import numpy as jnp
import openmdao.api as om
from src.frame_converter import EARTH_RAD, EARTH_OMEGA

EARTH_MU = 3.986004418e14  # м^3/с^2, гравитационный параметр Земли
G0 = 9.80665  # м/с^2, стандартное ускорение


class StageODE(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare('CD', types=float)
        self.options.declare('S', types=float)

        self.options.declare('nose_radius', types=float, default=0.5)

        self.options.declare('use_atmosphere', types=bool)
        self.options.declare('rho_ref', types=float)
        self.options.declare('h_scale', types=float)

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

        # ---- inputs: параметры ступени ----
        self.add_input('thrust_max', val=2.1e6 * np.ones(nn), units='N')
        self.add_input('Isp', val=265.2 * np.ones(nn), units='s')
        self.add_input('m_dry', val=1.0e3 * np.ones(nn), units='kg')
        self.add_input('m_propellant', val=1.0e5 * np.ones(nn), units='kg')

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

        self.add_output('q_heat', val=np.zeros(nn), units='W/m**2')
        self.add_output('q_dyn', val=np.zeros(nn), units='Pa')

        self.add_output('g_load', val=np.zeros(nn))

        self.add_output('orbit_a', val=EARTH_RAD * np.ones(nn), units='m')
        self.add_output('orbit_e', val=np.zeros(nn))
        self.add_output('orbit_inc', val=np.zeros(nn), units='rad')

        self.add_output('orbit_raan', val=np.zeros(nn), units='rad')
        self.add_output('orbit_arg_periapsis', val=np.zeros(nn), units='rad')

    def compute_primal(self,
                       rx, ry, rz,
                       vx, vy, vz,
                       m,
                       dir_x, dir_y, dir_z,
                       throttle,
                       thrust_max, Isp,
                       m_dry, m_propellant):

        CDA = self.options['CD'] * self.options['S']
        use_atmosphere = self.options['use_atmosphere']
        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']

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
            v_atm_y = EARTH_OMEGA * rx
            vrx = vx - v_atm_x
            vry = vy - v_atm_y
            vrz = vz
            v_rel = jnp.sqrt(vrx * vrx + vry * vry + vrz * vrz + 1.0)

            a_drag_x = -0.5 * CDA * rho * v_rel * vrx / m
            a_drag_y = -0.5 * CDA * rho * v_rel * vry / m
            a_drag_z = -0.5 * CDA * rho * v_rel * vrz / m

            SUTTON_GRAVES_K = 1.7415e-4
            nose_radius = self.options['nose_radius']
            q_heat = SUTTON_GRAVES_K * jnp.sqrt(rho / nose_radius) * v_rel ** 3
            q_dyn = 0.5 * rho * v_rel * v_rel
        else:
            a_drag_x = jnp.zeros_like(vx)
            a_drag_y = jnp.zeros_like(vy)
            a_drag_z = jnp.zeros_like(vz)
            q_heat = jnp.zeros_like(vx)
            q_dyn = jnp.zeros_like(vx)

        # ---- ускорение от тяги ----
        a_thrust_x = (F_T / m) * dx
        a_thrust_y = (F_T / m) * dy
        a_thrust_z = (F_T / m) * dz

        # ---- ускорение от внешних сил ----
        a_spec_x = a_thrust_x + a_drag_x
        a_spec_y = a_thrust_y + a_drag_y
        a_spec_z = a_thrust_z + a_drag_z
        g_load = jnp.sqrt(a_spec_x ** 2 + a_spec_y ** 2 + a_spec_z ** 2) / G0

        # ---- производные состояний ----
        rxdot = vx
        rydot = vy
        rzdot = vz
        vxdot = a_grav_x + a_spec_x
        vydot = a_grav_y + a_spec_y
        vzdot = a_grav_z + a_spec_z
        mdot = -F_T / (Isp * G0)

        # ---- диагностика ----
        r_mag = r
        v_mag = jnp.sqrt(vx * vx + vy * vy + vz * vz)
        v_radial = (rx * vx + ry * vy + rz * vz) / r
        dir_norm_sq = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z
        thrust_actual = F_T

        # ---- Орбитальные элементы ----
        v2_orb = vx * vx + vy * vy + vz * vz
        rdotv_orb = rx * vx + ry * vy + rz * vz

        eps_orb = 0.5 * v2_orb - EARTH_MU / r
        orbit_a = -EARTH_MU / (2.0 * eps_orb)

        hx = ry * vz - rz * vy
        hy = rz * vx - rx * vz
        hz = rx * vy - ry * vx
        h_norm = jnp.sqrt(hx * hx + hy * hy + hz * hz + 1e-12)

        coef_r = v2_orb / EARTH_MU - 1.0 / r
        coef_v = rdotv_orb / EARTH_MU
        ex = coef_r * rx - coef_v * vx
        ey = coef_r * ry - coef_v * vy
        ez = coef_r * rz - coef_v * vz
        orbit_e = jnp.sqrt(ex * ex + ey * ey + ez * ez + 1e-12)

        cos_i = jnp.clip(hz / h_norm, -1.0 + 1e-12, 1.0 - 1e-12)
        orbit_inc = jnp.arccos(cos_i)

        Nx = -hy
        Ny = hx
        # Nz = 0  по определению
        n_norm = jnp.sqrt(Nx * Nx + Ny * Ny + 1e-12)

        orbit_raan = jnp.arctan2(Ny, Nx)  # ∈ [-π, π]

        hxN_x = -hz * hx
        hxN_y = -hz * hy
        hxN_z = hx * hx + hy * hy  # = n_norm²

        # N · e и (h × N) · e
        N_dot_e = Nx * ex + Ny * ey
        hxN_dot_e = hxN_x * ex + hxN_y * ey + hxN_z * ez

        # ω = atan2((h × N) · e, |h| · (N · e))
        orbit_arg_periapsis = jnp.arctan2(hxN_dot_e, h_norm * N_dot_e)

        return (rxdot, rydot, rzdot,
                vxdot, vydot, vzdot,
                mdot,
                r_mag, v_mag, v_radial, dir_norm_sq, h, thrust_actual,
                q_heat, q_dyn, g_load,
                orbit_a, orbit_e, orbit_inc,
                orbit_raan, orbit_arg_periapsis)
