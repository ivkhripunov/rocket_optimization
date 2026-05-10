from src.single_stage import run_single_stage
from src.phase_config import PhaseConfig
from src.visualize import plot_eci_trajectory_3d, plot_eci_trajectory_zoomed, plot_ssto

p, sim_db = run_single_stage(PhaseConfig(), 0, 0, 0)

sol_db = p.get_outputs_dir() / 'dymos_solution.db'

plot_eci_trajectory_3d(sol_db)

plot_eci_trajectory_zoomed(sol_db)

plot_ssto(sol_db, sim_db)
