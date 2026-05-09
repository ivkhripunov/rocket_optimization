import openmdao.api as om
import numpy as np
import dymos as dm


class SSTO(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('g', types=float, default=9.80665,
                             desc='Gravitational acceleration, m/s**2')

        self.options.declare('rho_ref', types=float, default=1.225,
                             desc='Reference atmospheric density, kg/m**3')

        self.options.declare('h_scale', types=float, default=8.44E3,
                             desc='Reference altitude, m')

        self.options.declare('CD', types=float, default=0.5,
                             desc='coefficient of drag')

        self.options.declare('S', types=float, default=7.069,
                             desc='aerodynamic reference area (m**2)')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('y',
                       val=np.zeros(nn),
                       desc='altitude',
                       units='m')

        self.add_input('vx',
                       val=np.zeros(nn),
                       desc='x velocity',
                       units='m/s')

        self.add_input('vy',
                       val=np.zeros(nn),
                       desc='y velocity',
                       units='m/s')

        self.add_input('m',
                       val=np.zeros(nn),
                       desc='mass',
                       units='kg')

        self.add_input('theta',
                       val=np.zeros(nn),
                       desc='pitch angle',
                       units='rad')

        self.add_input('thrust',
                       val=2100000 * np.ones(nn),
                       desc='thrust',
                       units='N')

        self.add_input('Isp',
                       val=265.2 * np.ones(nn),
                       desc='specific impulse',
                       units='s')
        # Outputs
        self.add_output('xdot',
                        val=np.zeros(nn),
                        desc='velocity component in x',
                        units='m/s')

        self.add_output('ydot',
                        val=np.zeros(nn),
                        desc='velocity component in y',
                        units='m/s')

        self.add_output('vxdot',
                        val=np.zeros(nn),
                        desc='x acceleration magnitude',
                        units='m/s**2')

        self.add_output('vydot',
                        val=np.zeros(nn),
                        desc='y acceleration magnitude',
                        units='m/s**2')

        self.add_output('mdot',
                        val=np.zeros(nn),
                        desc='mass rate of change',
                        units='kg/s')

        self.add_output('rho',
                        val=np.zeros(nn),
                        desc='density',
                        units='kg/m**3')

        # Setup partials
        # Complex-step derivatives
        self.declare_coloring(wrt='*', method='cs')

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        vx = inputs['vx']
        vy = inputs['vy']
        m = inputs['m']
        F_T = inputs['thrust']
        Isp = inputs['Isp']
        y = inputs['y']

        g = self.options['g']
        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']

        CDA = self.options['CD'] * self.options['S']

        outputs['rho'] = rho_ref * np.exp(-y / h_scale)
        outputs['xdot'] = vx
        outputs['ydot'] = vy
        outputs['vxdot'] = (F_T * cos_theta - 0.5 * CDA * outputs['rho'] * vx ** 2) / m
        outputs['vydot'] = (F_T * sin_theta - 0.5 * CDA * outputs['rho'] * vy ** 2) / m - g
        outputs['mdot'] = -F_T / (g * Isp)


def run_ssto():
    #
    # Setup and solve the optimal control problem
    #
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring(tol=1.0E-12)

    #
    # Initialize our Trajectory and Phase
    #
    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=SSTO,
                     transcription=dm.GaussLobatto(num_segments=12, order=3, compressed=False))

    traj.add_phase('phase0', phase)
    p.model.add_subsystem('traj', traj)

    #
    # Set the options for the variables
    #
    phase.set_time_options(fix_initial=True, duration_bounds=(10, 500))

    phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                    rate_source='xdot')
    phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                    rate_source='ydot')
    phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                    rate_source='vxdot')
    phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                    rate_source='vydot')
    phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=100.0,
                    rate_source='mdot')

    phase.add_control('theta', units='rad', lower=-1.57, upper=1.57, targets=['theta'])
    phase.add_parameter('thrust', units='N', opt=False, val=2100000.0, targets=['thrust'])

    #
    # Set the options for our constraints and objective
    #
    phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
    phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
    phase.add_boundary_constraint('vy', loc='final', equals=0)

    phase.add_objective('time', loc='final', scaler=0.01)

    p.model.linear_solver = om.DirectSolver()

    #
    # Setup and set initial values
    #
    p.setup(check=True)

    phase.set_time_val(initial=0.0, duration=150.0)
    phase.set_state_val('x', [0, 1.15E5])
    phase.set_state_val('y', [0, 1.85E5])
    phase.set_state_val('vy', [1.0E-6, 0])
    phase.set_state_val('m', [117000, 1163])
    phase.set_control_val('theta', [1.5, -0.76])
    phase.set_parameter_val('thrust', 2.1, units='MN')

    dm.run_problem(p, simulate=True)

    sol_db = p.get_outputs_dir() / 'dymos_solution.db'

    sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

    return sol_db, sim_db