from src.ssto import run_ssto
from src.visualize import plot_ssto

sol_db, sim_db = run_ssto()

plot_ssto(sol_db, sim_db)
