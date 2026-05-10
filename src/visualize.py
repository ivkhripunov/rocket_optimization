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


def _gather_trajectory(case, phase_names):
    """Собрать (rx, ry, rz) по всем фазам и вернуть list массивов в км."""
    segments = []
    for name in phase_names:
        prefix = f'traj.{name}.timeseries.'
        try:
            rx = case.get_val(f'{prefix}rx').ravel() / 1e3
            ry = case.get_val(f'{prefix}ry').ravel() / 1e3
            rz = case.get_val(f'{prefix}rz').ravel() / 1e3
            segments.append((name, rx, ry, rz))
        except KeyError:
            pass
    return segments


def plot_eci_trajectory_3d(sol_db: Path,
                           sim_db: Path = None,
                           output_dir: Path = None,
                           phase_names=('phase0',)):
    """3D-траектория в инерциальной СК для одной или нескольких фаз."""
    sol = om.CaseReader(sol_db).get_case('final')
    sim = (om.CaseReader(sim_db).get_case('final')
           if sim_db and Path(sim_db).exists() else None)

    sol_segments = _gather_trajectory(sol, phase_names)
    sim_segments = _gather_trajectory(sim, phase_names) if sim else []

    if not sol_segments:
        print(f'предупреждение: не найдено timeseries для фаз {phase_names}')
        return

    R_km = R_EARTH / 1e3

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ---------------- Земля: каркасный глобус ----------------
    n_par = 12
    n_mer = 18
    n_smooth = 100

    lat_lines = np.linspace(-np.pi / 2, np.pi / 2, n_par + 2)[1:-1]
    lon_smooth = np.linspace(0, 2 * np.pi, n_smooth)
    for lat in lat_lines:
        x = R_km * np.cos(lat) * np.cos(lon_smooth)
        y = R_km * np.cos(lat) * np.sin(lon_smooth)
        z = R_km * np.sin(lat) * np.ones_like(lon_smooth)
        ax.plot(x, y, z, color='steelblue', lw=0.4, alpha=0.5)

    lon_lines = np.linspace(0, 2 * np.pi, n_mer, endpoint=False)
    lat_smooth = np.linspace(-np.pi / 2, np.pi / 2, n_smooth)
    for lon in lon_lines:
        x = R_km * np.cos(lat_smooth) * np.cos(lon)
        y = R_km * np.cos(lat_smooth) * np.sin(lon)
        z = R_km * np.sin(lat_smooth)
        ax.plot(x, y, z, color='steelblue', lw=0.4, alpha=0.5)

    ax.plot(R_km * np.cos(lon_smooth), R_km * np.sin(lon_smooth),
            np.zeros_like(lon_smooth),
            color='royalblue', lw=1.2, alpha=0.9, label='Экватор')
    ax.plot(R_km * np.cos(lat_smooth), np.zeros_like(lat_smooth),
            R_km * np.sin(lat_smooth),
            color='firebrick', lw=1.2, alpha=0.9, label='Меридиан 0°')

    axis_len = R_km * 1.5
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', lw=0.8, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', lw=0.8, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', lw=0.8, arrow_length_ratio=0.05)
    ax.text(axis_len * 1.05, 0, 0, 'X', fontsize=10)
    ax.text(0, axis_len * 1.05, 0, 'Y', fontsize=10)
    ax.text(0, 0, axis_len * 1.05, 'Z', fontsize=10)

    # ---------------- Траектория по фазам ----------------
    # Каждая фаза — своим цветом из качественной палитры
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(sol_segments), 3)))

    all_rx, all_ry, all_rz = [], [], []
    for i, (name, rx, ry, rz) in enumerate(sol_segments):
        color = palette[i]
        ax.plot(rx, ry, rz, color=color, lw=2.0,
                label=f'{name} (sol)')
        all_rx.append(rx)
        all_ry.append(ry)
        all_rz.append(rz)

    # Симуляция — пунктиром того же цвета
    for i, (name, srx, sry, srz) in enumerate(sim_segments):
        color = palette[i]
        ax.plot(srx, sry, srz, color=color, lw=1.0, ls='--', alpha=0.6,
                label=f'{name} (sim)')

    # Старт = начало первой фазы, орбита = конец последней
    first_name, frx, fry, frz = sol_segments[0]
    last_name, lrx, lry, lrz = sol_segments[-1]
    ax.scatter(frx[0], fry[0], frz[0], color='green', s=80, depthshade=False,
               edgecolors='k', linewidths=0.5, label='Старт', zorder=10)
    ax.scatter(lrx[-1], lry[-1], lrz[-1], color='red', s=80, depthshade=False,
               edgecolors='k', linewidths=0.5, label='Финал', zorder=10)

    # Маркеры между фазами (точки разделения ступеней)
    for i in range(len(sol_segments) - 1):
        _, _, _, _ = sol_segments[i]
        _, rx_next, ry_next, rz_next = sol_segments[i + 1]
        # Начало следующей фазы = точка разделения
        ax.scatter(rx_next[0], ry_next[0], rz_next[0],
                   color='black', s=40, marker='x', depthshade=False,
                   linewidths=1.5, zorder=11)

    # ---------------- Масштабирование 1:1:1 ----------------
    all_rx_flat = np.concatenate(all_rx)
    all_ry_flat = np.concatenate(all_ry)
    all_rz_flat = np.concatenate(all_rz)
    extent = max(np.max(np.abs(all_rx_flat)),
                 np.max(np.abs(all_ry_flat)),
                 np.max(np.abs(all_rz_flat)),
                 R_km) * 1.05
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel('X (ECI), км')
    ax.set_ylabel('Y (ECI), км')
    ax.set_zlabel('Z (ECI), км')
    ax.set_title('Траектория в инерциальной системе')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=25, azim=-50)

    out_dir = Path(output_dir) if output_dir else Path(sol_db).parent
    out_path = out_dir / 'trajectory_3d_eci.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'сохранён: {out_path}')


def plot_eci_trajectory_zoomed(sol_db: Path,
                               sim_db: Path = None,
                               output_dir: Path = None,
                               phase_names=('phase0',),
                               lat_pad_deg: float = 3.0,
                               lon_pad_deg: float = 5.0):
    """Приближённый вид области выведения по всем фазам."""
    sol = om.CaseReader(sol_db).get_case('final')
    sim = (om.CaseReader(sim_db).get_case('final')
           if sim_db and Path(sim_db).exists() else None)

    sol_segments = _gather_trajectory(sol, phase_names)
    sim_segments = _gather_trajectory(sim, phase_names) if sim else []

    if not sol_segments:
        print(f'предупреждение: не найдено timeseries для фаз {phase_names}')
        return

    R_km = R_EARTH / 1e3

    # ---- широта/долгота подспутниковых точек по всем фазам ----
    all_rx = np.concatenate([seg[1] for seg in sol_segments])
    all_ry = np.concatenate([seg[2] for seg in sol_segments])
    all_rz = np.concatenate([seg[3] for seg in sol_segments])

    r_mag = np.sqrt(all_rx * all_rx + all_ry * all_ry + all_rz * all_rz)
    lat_traj = np.arcsin(all_rz / r_mag)
    lon_traj = np.arctan2(all_ry, all_rx)

    # ---- диапазон сетки с отступом ----
    lat_pad = np.radians(lat_pad_deg)
    lon_pad = np.radians(lon_pad_deg)
    lat_min, lat_max = lat_traj.min() - lat_pad, lat_traj.max() + lat_pad
    lon_min, lon_max = lon_traj.min() - lon_pad, lon_traj.max() + lon_pad

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ---- сетка широта/долгота на поверхности ----
    n_par, n_mer, n_smooth = 8, 8, 80

    lon_smooth = np.linspace(lon_min, lon_max, n_smooth)
    lat_smooth = np.linspace(lat_min, lat_max, n_smooth)

    for lat in np.linspace(lat_min, lat_max, n_par):
        x = R_km * np.cos(lat) * np.cos(lon_smooth)
        y = R_km * np.cos(lat) * np.sin(lon_smooth)
        z = R_km * np.sin(lat) * np.ones_like(lon_smooth)
        ax.plot(x, y, z, color='steelblue', lw=0.6, alpha=0.7)

    for lon in np.linspace(lon_min, lon_max, n_mer):
        x = R_km * np.cos(lat_smooth) * np.cos(lon)
        y = R_km * np.cos(lat_smooth) * np.sin(lon)
        z = R_km * np.sin(lat_smooth)
        ax.plot(x, y, z, color='steelblue', lw=0.6, alpha=0.7)

    # ---- полупрозрачная подложка-поверхность для глубины ----
    LAT, LON = np.meshgrid(np.linspace(lat_min, lat_max, 30),
                           np.linspace(lon_min, lon_max, 30), indexing='ij')
    X = R_km * np.cos(LAT) * np.cos(LON)
    Y = R_km * np.cos(LAT) * np.sin(LON)
    Z = R_km * np.sin(LAT)
    ax.plot_surface(X, Y, Z, color='lightskyblue', alpha=0.15,
                    edgecolor='none', shade=False)

    # ---- траектории по фазам ----
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(sol_segments), 3)))

    for i, (name, rx, ry, rz) in enumerate(sol_segments):
        color = palette[i]
        ax.plot(rx, ry, rz, color=color, lw=2.2, label=f'{name} (sol)')

    for i, (name, srx, sry, srz) in enumerate(sim_segments):
        color = palette[i]
        ax.plot(srx, sry, srz, color=color, lw=1.0, ls='--', alpha=0.6,
                label=f'{name} (sim)')

    # Стартовая и финальная точки
    first_name, frx, fry, frz = sol_segments[0]
    last_name, lrx, lry, lrz = sol_segments[-1]
    ax.scatter(frx[0], fry[0], frz[0], color='green', s=120, depthshade=False,
               edgecolors='k', linewidths=0.8, label='Старт', zorder=10)
    ax.scatter(lrx[-1], lry[-1], lrz[-1], color='red', s=120, depthshade=False,
               edgecolors='k', linewidths=0.8, label='Финал', zorder=10)

    # Точки разделения ступеней
    for i in range(len(sol_segments) - 1):
        _, rx_next, ry_next, rz_next = sol_segments[i + 1]
        ax.scatter(rx_next[0], ry_next[0], rz_next[0],
                   color='black', s=50, marker='x', depthshade=False,
                   linewidths=1.5, zorder=11)

    # ---- bounding cube под траектории + патч поверхности ----
    bounds_x = np.concatenate([all_rx, X.ravel()])
    bounds_y = np.concatenate([all_ry, Y.ravel()])
    bounds_z = np.concatenate([all_rz, Z.ravel()])

    cx = (bounds_x.min() + bounds_x.max()) / 2
    cy = (bounds_y.min() + bounds_y.max()) / 2
    cz = (bounds_z.min() + bounds_z.max()) / 2
    extent = max(bounds_x.max() - bounds_x.min(),
                 bounds_y.max() - bounds_y.min(),
                 bounds_z.max() - bounds_z.min()) / 2 * 1.05

    ax.set_xlim(cx - extent, cx + extent)
    ax.set_ylim(cy - extent, cy + extent)
    ax.set_zlim(cz - extent, cz + extent)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel('X (ECI), км')
    ax.set_ylabel('Y (ECI), км')
    ax.set_zlabel('Z (ECI), км')
    ax.set_title('Область выведения (приближённый вид)')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=20, azim=-60)

    out_dir = Path(output_dir) if output_dir else Path(sol_db).parent
    out_path = out_dir / 'trajectory_3d_zoomed.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'сохранён: {out_path}')

def plot_multi_stage(sol_db, sim_db, phase_names):
    """Объединяет timeseries всех фаз в общий график."""
    sol = om.CaseReader(sol_db).get_case('final')
    sim = om.CaseReader(sim_db).get_case('final') if Path(sim_db).exists() else None
    output_dir = Path(sol_db).parent

    # Какие переменные есть (берём из первой фазы)
    pref0 = f'traj.{phase_names[0]}.timeseries.'
    all_keys = [k for k in sol.outputs.keys() if k.startswith(pref0)]
    param_names = [k.removeprefix(pref0) for k in all_keys
                   if k != f'{pref0}time']

    for param in param_names:
        label, unit, scale = LABELS.get(param, (param, 'ед.', 1.0))

        fig, ax = plt.subplots(figsize=(10, 4))

        for phase_name in phase_names:
            pref = f'traj.{phase_name}.timeseries.'
            try:
                t = sol.get_val(f'{pref}time').ravel()
                v = sol.get_val(f'{pref}{param}').ravel() * scale
                ax.plot(t, v, marker='o', ms=3, linestyle='-',
                        label=f'{phase_name} (sol)')

                if sim:
                    ts = sim.get_val(f'{pref}time').ravel()
                    vs = sim.get_val(f'{pref}{param}').ravel() * scale
                    ax.plot(ts, vs, linestyle='--', alpha=0.6,
                            label=f'{phase_name} (sim)')
            except KeyError:
                pass

        ax.set_xlabel('Время, с')
        ax.set_ylabel(f'{label}, {unit}')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        out_path = output_dir / f'{param}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'сохранён: {out_path}')
