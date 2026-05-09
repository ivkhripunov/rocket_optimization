import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import openmdao.api as om
import numpy as np
from pathlib import Path

R_EARTH = 6_378_137.0  # м, экваториальный радиус WGS-84

# Подписи и масштабы для всех известных переменных (2D и 3D задач)
LABELS = {
    # 2D переменные
    'x': ('Дальность', 'км', 1e-3),
    'y': ('Высота', 'км', 1e-3),
    'theta': ('Угол тангажа θ', '°', 180 / np.pi),

    # общие для 2D и 3D
    'm': ('Масса', 'кг', 1.0),
    'rho': ('Плотность воздуха', 'кг/м³', 1.0),

    # 3D позиция
    'rx': ('rx (ECI)', 'км', 1e-3),
    'ry': ('ry (ECI)', 'км', 1e-3),
    'rz': ('rz (ECI)', 'км', 1e-3),

    # 3D скорости
    'vx': ('Скорость vx', 'км/с', 1e-3),
    'vy': ('Скорость vy', 'км/с', 1e-3),
    'vz': ('Скорость vz', 'км/с', 1e-3),

    # 3D управление (вектор направления тяги)
    'dir_x': ('Направление dx', '—', 1.0),
    'dir_y': ('Направление dy', '—', 1.0),
    'dir_z': ('Направление dz', '—', 1.0),
    'dir_norm_sq': ('||dir||²', '—', 1.0),

    # 3D диагностика
    'r_mag': ('Геоцентрический радиус', 'км', 1e-3),
    'v_mag': ('Скорость |v|', 'км/с', 1e-3),
    'v_radial': ('Радиальная скорость', 'м/с', 1.0),
    'altitude': ('Высота над поверхностью', 'км', 1e-3),
}

SOL_KW = dict(marker='o', ms=3, linestyle='none', color='tab:blue', label='коллокация')
SIM_KW = dict(marker=None, linestyle='-', color='tab:orange', label='симуляция')


def plot_ssto(sol_db: Path, sim_db: Path):
    output_dir = Path(sol_db).parent
    sol = om.CaseReader(sol_db).get_case('final')
    sim = om.CaseReader(sim_db).get_case('final') if Path(sim_db).exists() else None

    prefix = 'traj.phase0.timeseries.'
    all_keys = [k for k in sol.outputs.keys() if k.startswith(prefix)]
    param_keys = [k for k in all_keys if k != f'{prefix}time']

    sol_time = sol.get_val(f'{prefix}time').ravel()
    sim_time = sim.get_val(f'{prefix}time').ravel() if sim else None

    # ---------- одиночные графики по каждой переменной ----------
    for key in param_keys:
        param = key.removeprefix(prefix)
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
        plt.close(fig)
        print(f'сохранён: {out_path}')


def plot_eci_trajectory_3d(sol_db: Path, sim_db: Path = None, output_dir: Path = None):
    R_EARTH = 6_378_137.0  # м, экваториальный радиус WGS-84

    sol = om.CaseReader(sol_db).get_case('final')
    sim = om.CaseReader(sim_db).get_case('final') if sim_db and Path(sim_db).exists() else None

    prefix = 'traj.phase0.timeseries.'
    rx = sol.get_val(f'{prefix}rx').ravel() / 1e3  # м → км
    ry = sol.get_val(f'{prefix}ry').ravel() / 1e3
    rz = sol.get_val(f'{prefix}rz').ravel() / 1e3

    R_km = R_EARTH / 1e3

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ---------------- Земля: каркасный глобус ----------------
    n_par = 12  # параллелей в полусфере
    n_mer = 18  # меридианов
    n_smooth = 100  # точек вдоль каждой линии (для гладкости)

    # параллели (φ = const), исключая полюсы
    lat_lines = np.linspace(-np.pi / 2, np.pi / 2, n_par + 2)[1:-1]
    lon_smooth = np.linspace(0, 2 * np.pi, n_smooth)
    for lat in lat_lines:
        x = R_km * np.cos(lat) * np.cos(lon_smooth)
        y = R_km * np.cos(lat) * np.sin(lon_smooth)
        z = R_km * np.sin(lat) * np.ones_like(lon_smooth)
        ax.plot(x, y, z, color='steelblue', lw=0.4, alpha=0.5)

    # меридианы (λ = const)
    lon_lines = np.linspace(0, 2 * np.pi, n_mer, endpoint=False)
    lat_smooth = np.linspace(-np.pi / 2, np.pi / 2, n_smooth)
    for lon in lon_lines:
        x = R_km * np.cos(lat_smooth) * np.cos(lon)
        y = R_km * np.cos(lat_smooth) * np.sin(lon)
        z = R_km * np.sin(lat_smooth)
        ax.plot(x, y, z, color='steelblue', lw=0.4, alpha=0.5)

    # экватор и Гринвичский меридиан — выделяем
    ax.plot(R_km * np.cos(lon_smooth), R_km * np.sin(lon_smooth),
            np.zeros_like(lon_smooth),
            color='royalblue', lw=1.2, alpha=0.9, label='Экватор')
    ax.plot(R_km * np.cos(lat_smooth), np.zeros_like(lat_smooth),
            R_km * np.sin(lat_smooth),
            color='firebrick', lw=1.2, alpha=0.9, label='Меридиан 0°')

    # оси ECI
    axis_len = R_km * 1.5
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', lw=0.8, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', lw=0.8, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', lw=0.8, arrow_length_ratio=0.05)
    ax.text(axis_len * 1.05, 0, 0, 'X', fontsize=10)
    ax.text(0, axis_len * 1.05, 0, 'Y', fontsize=10)
    ax.text(0, 0, axis_len * 1.05, 'Z', fontsize=10)

    # ---------------- Траектория ----------------
    ax.plot(rx, ry, rz, color='tab:orange', lw=2.2, label='Траектория (solution)')
    ax.scatter(*[rx[0], ry[0], rz[0]], color='green', s=80, depthshade=False,
               edgecolors='k', linewidths=0.5, label='Старт', zorder=10)
    ax.scatter(*[rx[-1], ry[-1], rz[-1]], color='red', s=80, depthshade=False,
               edgecolors='k', linewidths=0.5, label='Орбита', zorder=10)

    if sim:
        srx = sim.get_val(f'{prefix}rx').ravel() / 1e3
        sry = sim.get_val(f'{prefix}ry').ravel() / 1e3
        srz = sim.get_val(f'{prefix}rz').ravel() / 1e3
        ax.plot(srx, sry, srz, color='tab:blue', lw=1.0, ls='--', alpha=0.8,
                label='Траектория (simulation)')

    # ---------------- Масштабирование 1:1:1 ----------------
    extent = max(np.max(np.abs(rx)), np.max(np.abs(ry)), np.max(np.abs(rz)),
                 R_km) * 1.05
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel('X (ECI), км')
    ax.set_ylabel('Y (ECI), км')
    ax.set_zlabel('Z (ECI), км')
    ax.set_title('Траектория в инерциальной системе')
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=25, azim=-50)

    plt.show()

    out_dir = Path(output_dir) if output_dir else Path(sol_db).parent
    out_path = out_dir / 'trajectory_3d_eci.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'сохранён: {out_path}')
