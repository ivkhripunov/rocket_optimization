import openmdao.api as om
import dymos as dm
import numpy as np
import jax.numpy as jnp

from src.frame_converter import *

EARTH_MU = 3.986004418e14
EARTH_OMEGA = 7.2921150e-5
G0 = 9.80665


class SSTO3D(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

        # Параметры аэродинамики
        self.options.declare('CD', types=float, default=0.5)
        self.options.declare('S', types=float, default=7.069)

        # Параметры модели атмосферы
        self.options.declare('rho_ref', types=float, default=1.225)
        self.options.declare('h_scale', types=float, default=8.44e3)

        # Параметры тяги
        self.options.declare('thrust', types=float, default=2.1e6)
        self.options.declare('Isp', types=float, default=265.2)

    def setup(self):
        nn = self.options['num_nodes']

        # =========================================================
        # Input
        # =========================================================

        for n in ('rx', 'ry', 'rz'):
            self.add_input(n, val=np.zeros(nn), units='m')
        for n in ('vx', 'vy', 'vz'):
            self.add_input(n, val=np.zeros(nn), units='m/s')

        self.add_input('m', val=np.zeros(nn), units='kg')

        self.add_input('dir_x', val=np.ones(nn))
        self.add_input('dir_y', val=np.zeros(nn))
        self.add_input('dir_z', val=np.zeros(nn))

        # =========================================================
        # Output
        # =========================================================

        for n in ('rxdot', 'rydot', 'rzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s')
        for n in ('vxdot', 'vydot', 'vzdot'):
            self.add_output(n, val=np.zeros(nn), units='m/s**2')

        self.add_output('mdot', val=np.zeros(nn), units='kg/s')

    def compute(self, inputs, outputs):
        # =========================================================
        # Options
        # =========================================================

        CDA = self.options['CD'] * self.options['S']

        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']

        F_T = self.options['thrust']
        Isp = self.options['Isp']

        # =========================================================
        # Inputs
        # =========================================================

        rx, ry, rz = inputs['rx'], inputs['ry'], inputs['rz']
        vx, vy, vz = inputs['vx'], inputs['vy'], inputs['vz']

        m = inputs['m']

        dir_x, dir_y, dir_z = inputs['dir_x'], inputs['dir_y'], inputs['dir_z']

        # =========================================================
        # Position / velocity / direction vectors
        # =========================================================

        r_vec = jnp.stack([rx, ry, rz], axis=1)
        v_vec = jnp.stack([vx, vy, vz], axis=1)

        dir_vec = jnp.stack([dir_x, dir_y, dir_z], axis=1)
        dir_norm = jnp.linalg.norm(dir_vec, axis=1, keepdims=True)

        dir_vec = dir_vec / dir_norm

        # =========================================================
        # Gravity
        # =========================================================

        r = jnp.linalg.norm(r_vec, axis=1)

        a_grav = -EARTH_MU * r_vec / r[:, None] ** 3

        # =========================================================
        # Atmosphere density
        # =========================================================

        h = r - EARTH_OMEGA

        rho = rho_ref * jnp.exp(-h / h_scale)

        omega_vec = jnp.array([0.0, 0.0, EARTH_OMEGA])

        omega_vec = jnp.broadcast_to(
            omega_vec,
            r_vec.shape
        )

        v_atm = jnp.cross(omega_vec, r_vec)

        v_rel = v_vec - v_atm

        v_rel_norm = jnp.linalg.norm(v_rel, axis=1)

        a_drag = (
                -0.5
                * rho[:, None]
                * CDA
                * v_rel
                * v_rel_norm[:, None]
                / m[:, None]
        )

        # =========================================================
        # Thrust acceleration
        # =========================================================

        a_thrust = (F_T / m)[:, None] * dir_vec

        # =========================================================
        # Total acceleration
        # =========================================================

        a_total = a_grav + a_drag + a_thrust

        # =========================================================
        # Mass flow
        # =========================================================

        mdot = -F_T / (Isp * G0)

        # =========================================================
        # Outputs
        # =========================================================

        outputs['rxdot'] = vx
        outputs['rydot'] = vy
        outputs['rzdot'] = vz

        outputs['vxdot'] = a_total[:, 0]
        outputs['vydot'] = a_total[:, 1]
        outputs['vzdot'] = a_total[:, 2]

        outputs['mdot'] = mdot * jnp.ones_like(m)
