import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openmdao.api as om
import numpy as np
from pathlib import Path


def plot_ssto(sol_db: Path, sim_db: Path):
    output_dir = Path(sol_db).parent
    sol = om.CaseReader(sol_db).get_case('final')
    sim = om.CaseReader(sim_db).get_case('final') if Path(sim_db).exists() else None

    # собираем все timeseries-переменные из solution
    prefix = 'traj.phase0.timeseries.'
    all_keys = [k for k in sol.outputs.keys() if k.startswith(prefix)]
    param_keys = [k for k in all_keys if k != f'{prefix}time']

    sol_time = sol.get_val(f'{prefix}time').ravel()
    sim_time = sim.get_val(f'{prefix}time').ravel() if sim else None

    # единицы и подписи для известных переменных
    LABELS = {
        'x': ('Дальность', 'км', 1e-3),
        'y': ('Высота', 'км', 1e-3),
        'vx': ('Скорость vx', 'км/с', 1e-3),
        'vy': ('Скорость vy', 'км/с', 1e-3),
        'm': ('Масса', 'кг', 1.0),
        'theta': ('Угол тангажа θ', '°', 180 / np.pi),
        'rho': ('Плотность воздуха', 'кг/м³', 1.0),
    }

    SOL_KW = dict(marker='o', ms=3, linestyle='none', color='tab:blue', label='коллокация')
    SIM_KW = dict(marker=None, linestyle='-', color='tab:orange', label='симуляция')

    for key in param_keys:
        param = key.removeprefix(prefix)  # 'traj.phase0.timeseries.vx' → 'vx'
        label, unit, scale = LABELS.get(param, (param, 'ед.', 1.0))

        sol_vals = sol.get_val(key).ravel() * scale

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sol_time, sol_vals, **SOL_KW)

        if sim:
            try:
                sim_vals = sim.get_val(key).ravel() * scale
                ax.plot(sim_time, sim_vals, **SIM_KW)
            except KeyError:
                pass

        ax.set_xlabel('Время, с')
        ax.set_ylabel(f'{label}, {unit}')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        out_path = output_dir / f'{param}.png'
        fig.savefig(out_path, dpi=150)
        print(f'сохранён: {out_path}')
        plt.close(fig)

    # отдельно — траектория x-y
    if f'{prefix}x' in all_keys and f'{prefix}y' in all_keys:
        sol_x = sol.get_val(f'{prefix}x').ravel() * 1e-3
        sol_y = sol.get_val(f'{prefix}y').ravel() * 1e-3

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sol_x, sol_y, **SOL_KW)

        if sim:
            sim_x = sim.get_val(f'{prefix}x').ravel() * 1e-3
            sim_y = sim.get_val(f'{prefix}y').ravel() * 1e-3
            ax.plot(sim_x, sim_y, **SIM_KW)

        ax.axhline(185, color='gray', ls='--', lw=0.8, label='орбита 185 км')
        ax.set_xlabel('Дальность, км')
        ax.set_ylabel('Высота, км')
        ax.set_title('Траектория полёта')
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        out_path = output_dir / 'trajectory_xy.png'
        fig.savefig(out_path, dpi=150)
        print(f'сохранён: {out_path}')
        plt.close(fig)