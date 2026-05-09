from src.ssto_3d import run_ssto_3d
from src.visualize import plot_eci_trajectory_3d, plot_ssto

p, sim_db = run_ssto_3d()

sol_db = p.get_outputs_dir() / 'dymos_solution.db'

plot_eci_trajectory_3d(sol_db)

plot_ssto(sol_db, sim_db)