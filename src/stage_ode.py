import numpy as np
import casadi as ca
import openmdao.api as om
from src.frame_converter import EARTH_RAD, EARTH_OMEGA

EARTH_MU = 3.986004418e14  # м³/с²
G0 = 9.80665  # м/с²
SGK = 1.7415e-4  # коэффициент Sutton-Graves


def _cross(a, b):
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _smooth_h(x, a=0.1):
    return 0.5 * (1.0 + ca.tanh(x / (2.0 * a)))


def rv2oe(rv, vv, mu, a_eps=0.1):
    eps = 1e-12

    K = ca.vertcat(0.0, 0.0, 1.0)
    hv = _cross(rv, vv)  # угловой момент
    nv = _cross(K, hv)  # вектор узла N = ẑ × h

    h2 = ca.fmax(_dot(hv, hv), eps)
    v2 = ca.fmax(_dot(vv, vv), eps)
    r = ca.sqrt(ca.fmax(_dot(rv, rv), eps))
    n = ca.sqrt(ca.fmax(_dot(nv, nv), eps))

    rv_dot_vv = _dot(rv, vv)

    ev = (1.0 / mu) * ((v2 - mu / r) * rv - rv_dot_vv * vv)

    p = h2 / mu
    e = ca.sqrt(ca.fmax(_dot(ev, ev), eps))
    a = p / ca.fmax(1.0 - e * e, eps)
    i_oe = ca.acos(ca.fmax(ca.fmin(hv[2] / ca.sqrt(h2), 1.0 - eps), -1.0 + eps))

    cos_Om = ca.fmax(ca.fmin(nv[0] / n, 1.0 - eps), -1.0 + eps)
    Om_raw = ca.acos(cos_Om)
    Om = (_smooth_h(nv[1], a_eps) * Om_raw
          + _smooth_h(-nv[1], a_eps) * (2.0 * ca.pi - Om_raw))

    nv_dot_ev = _dot(nv, ev)
    cos_om = ca.fmax(ca.fmin(nv_dot_ev / (n * e), 1.0 - eps), -1.0 + eps)
    om_raw = ca.acos(cos_om)
    om = (_smooth_h(ev[2], a_eps) * om_raw
          + _smooth_h(-ev[2], a_eps) * (2.0 * ca.pi - om_raw))

    ev_dot_rv = _dot(ev, rv)
    cos_nu = ca.fmax(ca.fmin(ev_dot_rv / (e * r), 1.0 - eps), -1.0 + eps)
    nu_raw = ca.acos(cos_nu)
    nu = (_smooth_h(rv_dot_vv, a_eps) * nu_raw
          + _smooth_h(-rv_dot_vv, a_eps) * (2.0 * ca.pi - nu_raw))

    return ca.vertcat(a, e, i_oe, Om, om, nu)


class StageODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('CD', types=float)
        self.options.declare('S', types=float)
        self.options.declare('nose_radius', types=float, default=0.5)
        self.options.declare('use_atmosphere', types=bool)
        self.options.declare('rho_ref', types=float, default=1.225)
        self.options.declare('h_scale', types=float, default=8.44e3)

        self._fn_map = None
        self._in_names = []
        self._out_names = []
        self._n_in = 0
        self._n_out = 0

    def _build_casadi(self):
        nn = self.options['num_nodes']
        CDA = self.options['CD'] * self.options['S']
        atm = self.options['use_atmosphere']
        rho_ref = self.options['rho_ref']
        h_sc = self.options['h_scale']
        Rn = self.options['nose_radius']

        self._in_names = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'm',
                          'dir_x', 'dir_y', 'dir_z', 'throttle',
                          'thrust', 'Isp', 'm_dry', 'm_propellant']
        self._n_in = len(self._in_names)

        x = ca.MX.sym('x', self._n_in)

        rx, ry, rz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        m = x[6]
        dx, dy, dz = x[7], x[8], x[9]
        thr = x[10]
        thrust = x[11]
        Isp = x[12]

        # ---- физика ----
        F_T = thrust * thr
        dn = ca.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-12)
        ex_ = dx / dn
        ey_ = dy / dn
        ez_ = dz / dn

        r = ca.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        h_alt = r - EARTH_RAD

        inv_r3 = r ** (-3)
        a_gx = -EARTH_MU * rx * inv_r3
        a_gy = -EARTH_MU * ry * inv_r3
        a_gz = -EARTH_MU * rz * inv_r3

        if atm:
            rho = rho_ref * ca.exp(-h_alt / h_sc)
            vrx = vx + EARTH_OMEGA * ry
            vry = vy - EARTH_OMEGA * rx
            v_rel = ca.sqrt(vrx ** 2 + vry ** 2 + vz ** 2 + 1.0)
            adx = -0.5 * CDA * rho * v_rel * vrx / m
            ady = -0.5 * CDA * rho * v_rel * vry / m
            adz = -0.5 * CDA * rho * v_rel * vz / m
            q_heat = SGK * ca.sqrt(rho / Rn) * v_rel ** 3
            q_dyn = 0.5 * rho * v_rel ** 2
        else:
            adx = ady = adz = ca.MX(0.0)
            q_heat = q_dyn = ca.MX(0.0)

        asx = F_T / m * ex_ + adx
        asy = F_T / m * ey_ + ady
        asz = F_T / m * ez_ + adz
        g_load = ca.sqrt(asx ** 2 + asy ** 2 + asz ** 2) / G0

        rxdot = vx
        rydot = vy
        rzdot = vz
        vxdot = a_gx + asx
        vydot = a_gy + asy
        vzdot = a_gz + asz
        mdot = -F_T / (Isp * G0)

        v_mag = ca.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        v_radial = (rx * vx + ry * vy + rz * vz) / r
        dir_nsq = dx ** 2 + dy ** 2 + dz ** 2

        oe = rv2oe(ca.vertcat(rx, ry, rz), ca.vertcat(vx, vy, vz), EARTH_MU)

        out_vec = ca.vertcat(
            rxdot, rydot, rzdot,
            vxdot, vydot, vzdot, mdot,
            r, v_mag, v_radial, dir_nsq, h_alt, F_T,
            q_heat, q_dyn, g_load,
            oe[0], oe[1], oe[2], oe[3], oe[4], oe[5],
        )
        self._out_names = [
            'rxdot', 'rydot', 'rzdot',
            'vxdot', 'vydot', 'vzdot', 'mdot',
            'r_mag', 'v_mag', 'v_radial', 'dir_norm_sq', 'h', 'thrust_actual',
            'q_heat', 'q_dyn', 'g_load',
            'orbit_a', 'orbit_e', 'orbit_inc', 'orbit_raan', 'orbit_arg_periapsis', 'orbit_nu',
        ]
        self._n_out = len(self._out_names)

        J_sym = ca.jacobian(out_vec, x)  # (n_out, n_in) — символьно

        fn = ca.Function('ode_and_jac', [x], [out_vec, J_sym])

        self._fn_map = fn.map(nn, 'serial')

    def setup(self):
        nn = self.options['num_nodes']
        self._build_casadi()

        for n in ('rx', 'ry', 'rz'):
            self.add_input(n, val=EARTH_RAD * np.ones(nn), units='m')
        for n in ('vx', 'vy', 'vz'):
            self.add_input(n, val=np.zeros(nn), units='m/s')
        self.add_input('m', val=1e5 * np.ones(nn), units='kg')
        self.add_input('dir_x', val=np.ones(nn))
        self.add_input('dir_y', val=np.zeros(nn))
        self.add_input('dir_z', val=np.zeros(nn))
        self.add_input('throttle', val=np.ones(nn))
        self.add_input('thrust', val=2.1e6 * np.ones(nn), units='N')
        self.add_input('Isp', val=265.2 * np.ones(nn), units='s')
        self.add_input('m_dry', val=1e3 * np.ones(nn), units='kg')
        self.add_input('m_propellant', val=1e5 * np.ones(nn), units='kg')

        for n in ('rxdot', 'rydot', 'rzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s')
        for n in ('vxdot', 'vydot', 'vzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), units='kg/s')
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
        self.add_output('orbit_nu', val=np.zeros(nn), units='rad')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn, dtype=int)
        for o in self._out_names:
            for i in self._in_names:
                self.declare_partials(o, i, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        X = np.stack([np.asarray(inputs[n]).ravel()
                      for n in self._in_names], axis=0)
        outs, _ = self._fn_map(X)
        out_arr = outs.full()
        for k, name in enumerate(self._out_names):
            outputs[name] = out_arr[k, :]

    def compute_partials(self, inputs, partials):
        X = np.stack([np.asarray(inputs[n]).ravel()
                      for n in self._in_names], axis=0)
        _, J_dm = self._fn_map(X)
        J = J_dm.full()
        for oi, oname in enumerate(self._out_names):
            for ii, iname in enumerate(self._in_names):
                partials[oname, iname] = J[oi, ii::self._n_in]
