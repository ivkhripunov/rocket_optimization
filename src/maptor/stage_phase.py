"""
build_maptor_phase() — строит одну Maptor-фазу из PhaseConfig.

Масштабирование состояний:
    r_s  = r  / R_SCALE   (м  → ~1–2)
    v_s  = v  / V_SCALE   (м/с → ~1–10)
    m_s  = m  / M_SCALE   (кг  → ~1–30)
    F_s  = F  / F_SCALE   (Н   → ~1–5)  для оптимизируемой тяги

Линковка масс между фазами:
    m_start(B) = m_final(A) - drop_after_A

Оптимизация конструкции — через константное состояние (нулевая производная):
    fix_thrust=False  → thrust_s  = state(dm/dt = 0), физ. тяга = thrust_s × F_SCALE
    fix_Isp=False     → Isp_s     = state(dm/dt = 0), физ. Isp  = Isp_s × ISP_SCALE
    fix_m_dry=False   → m_dry_s   = state(dm/dt = 0) + path-constraint m_s >= m_dry_s
"""

import numpy as np
import casadi as ca
import maptor as mtor

from src.maptor.constants import (
    EARTH_MU, EARTH_RAD, EARTH_OMEGA, G0,
    R_SCALE, V_SCALE, M_SCALE,
    R_MIN, R_MAX, V_MIN, V_MAX,
    OMEGA_MATRIX,
)
from src.maptor.phase_config import PhaseConfig

# Масштаб для тяги и Isp (дополнительные design переменные)
F_SCALE = 1.0e6  # Н → МН
ISP_SCALE = 300.0  # с → безразмерный ~ 1


# =============================================================================
# Динамика фазы (масштабированная)
# =============================================================================

def _phase_dynamics(phase, r_s, v_s, m_s, u,
                    config: PhaseConfig,
                    thrust_s_var=None,  # None → config.thrust (const)
                    Isp_s_var=None,  # None → config.Isp    (const)
                    m_dry_s_var=None):  # None → config.m_dry/M_SCALE (const)
    """
    Определить динамику и путевые ограничения одной фазы.

    thrust_s_var, Isp_s_var, m_dry_s_var — Maptor-состояния с нулевой
    производной (если не None).
    """
    # Символьные векторы
    r_vec_s = ca.vertcat(*[r_s[i] for i in range(3)])
    v_vec_s = ca.vertcat(*[v_s[i] for i in range(3)])
    u_vec = ca.vertcat(*[u[i] for i in range(3)])
    m_phys = m_s * M_SCALE

    # Физические координаты
    r_vec = r_vec_s * R_SCALE
    v_vec = v_vec_s * V_SCALE

    rad = ca.sqrt(ca.fmax(ca.dot(r_vec, r_vec), 1e-12))
    h_alt = rad - EARTH_RAD

    # Гравитация
    grav = -(EARTH_MU / rad ** 3) * r_vec

    # Скорость относительно атмосферы
    vrel = v_vec - OMEGA_MATRIX @ r_vec
    speedrel = ca.sqrt(ca.fmax(ca.dot(vrel, vrel), 1e-12))

    # Атмосферное сопротивление
    if config.use_atmosphere:
        rho = config.rho_ref * ca.exp(-h_alt / config.h_scale)
        bc = (rho / (2.0 * m_phys)) * config.CD * config.S
        drag = -vrel * bc * speedrel
        q_dyn = 0.5 * rho * speedrel ** 2
        q_heat = 1.7415e-4 * ca.sqrt(rho) * speedrel ** 3  # Sutton-Graves
    else:
        drag = ca.MX.zeros(3, 1)  # (3,1) — совпадает с формой grav и thrust_accel
        q_dyn = ca.MX(0.0)
        q_heat = ca.MX(0.0)

    # Тяга (физическая, в Н)
    if thrust_s_var is not None:
        thrust_phys = thrust_s_var * F_SCALE  # МН → Н
    else:
        thrust_phys = config.thrust  # Н (константа)

    # Isp (физический, в с)
    if Isp_s_var is not None:
        Isp_phys = Isp_s_var * ISP_SCALE  # безразм. → с
    else:
        Isp_phys = config.Isp  # с (константа)

    # Ускорение от тяги, расход
    Toverm = thrust_phys / m_phys
    thrust_accel = Toverm * u_vec
    mdot_phys = -thrust_phys / (Isp_phys * G0)

    # Суммарное ускорение
    accel = thrust_accel + drag + grav

    # Масштабированные производные состояний
    r_dot_s = v_vec / R_SCALE
    v_dot_s = accel / V_SCALE
    m_dot_s = mdot_phys / M_SCALE

    # Словарь динамики
    dyn = {}
    for i in range(3):
        dyn[r_s[i]] = r_dot_s[i]
        dyn[v_s[i]] = v_dot_s[i]
    dyn[m_s] = m_dot_s

    # Нулевые производные для design-переменных
    if thrust_s_var is not None:
        dyn[thrust_s_var] = ca.MX(0.0)
    if Isp_s_var is not None:
        dyn[Isp_s_var] = ca.MX(0.0)
    if m_dry_s_var is not None:
        dyn[m_dry_s_var] = ca.MX(0.0)

    phase.dynamics(dyn)

    # Путевые ограничения
    path = []
    path.append(ca.dot(u_vec, u_vec) == 1.0)  # единичный вектор тяги
    path.append(h_alt >= -100.0)  # не под землёй

    # Масса не ниже сухой (при оптимизации m_dry — динамическая граница)
    if m_dry_s_var is not None:
        path.append(m_s >= m_dry_s_var)

    if config.q_heat_constraint:
        path.append(q_heat <= config.q_heat_max)
    if config.q_dyn_constraint:
        path.append(q_dyn <= config.q_dyn_max)
    if config.g_load_constraint:
        g_load = ca.sqrt(ca.dot(thrust_accel + drag, thrust_accel + drag)) / G0
        path.append(g_load <= config.g_load_max)

    phase.path_constraints(*path)


# =============================================================================
# Публичная функция: сборка одной фазы
# =============================================================================

def build_maptor_phase(
        problem,
        config: PhaseConfig,
        t_start: float,
        prev_states: dict | None = None,
        mass_drop_kg: float = 0.0,
        r0_s: list | None = None,
        v0_s: list | None = None,
) -> dict:
    """
    Построить одну Maptor-фазу из PhaseConfig.

    Parameters
    ----------
    problem       : mtor.Problem
    config        : PhaseConfig
    t_start       : время начала этой фазы (с)
    prev_states   : результат build_maptor_phase предыдущей фазы
                    (None → первая фаза)
    mass_drop_kg  : масса, сбрасываемая ПОСЛЕ этой фазы (кг)
                    Это фиксированная аппаратная масса (корпуса ускорителей и т.д.),
                    НЕ связана напрямую с m_dry фазы.
    r0_s, v0_s    : масштабированные начальные условия (только для первой фазы)

    Returns
    -------
    dict с ключами:
        'phase'        : mtor.Phase
        'r_s'          : список [r0, r1, r2] (состояния)
        'v_s'          : список [v0, v1, v2]
        'm_s'          : состояние массы
        'u'            : список [u0, u1, u2] (управления)
        'mass_drop_s'  : масса сброса в масш. ед. (для следующей фазы)
        't_end'        : конечное время (числовой guess)
        'thrust_s_var' : design-переменная тяги (None если fix_thrust=True)
        'Isp_s_var'    : design-переменная Isp   (None если fix_Isp=True)
        'm_dry_s_var'  : design-переменная m_dry (None если fix_m_dry=True)
        'm_prop_s_var' : design-переменная m_prop (None если fix_m_propellant=True)
    """
    config.validate()

    phase = problem.set_phase(config.phase_id)

    # ── Время ────────────────────────────────────────────────────────────────
    if config.fix_duration:
        t_end = t_start + config.duration
        phase.time(initial=t_start, final=t_end)
    else:
        # duration_bounds задаёт (t_end_lo, t_end_hi) — абсолютное время
        t_lo, t_hi = config.duration_bounds
        t_end = t_start + config.duration  # guess
        phase.time(initial=t_start, final=(t_lo, t_hi))

    # ── Положение ────────────────────────────────────────────────────────────
    if prev_states is None:
        assert r0_s is not None, "Первая фаза требует r0_s"
        r_s = [
            phase.state(f'r{i}_s', initial=r0_s[i], boundary=(R_MIN, R_MAX))
            for i in range(3)
        ]
    else:
        r_s = [
            phase.state(f'r{i}_s',
                        initial=prev_states['r_s'][i].final,
                        boundary=(R_MIN, R_MAX))
            for i in range(3)
        ]

    # ── Скорость ─────────────────────────────────────────────────────────────
    if prev_states is None:
        assert v0_s is not None, "Первая фаза требует v0_s"
        v_s = [
            phase.state(f'v{i}_s', initial=v0_s[i], boundary=(V_MIN, V_MAX))
            for i in range(3)
        ]
    else:
        v_s = [
            phase.state(f'v{i}_s',
                        initial=prev_states['v_s'][i].final,
                        boundary=(V_MIN, V_MAX))
            for i in range(3)
        ]

    # ── Design-переменные конструкции ────────────────────────────────────────

    # Тяга (если оптимизируется)
    thrust_s_var = None
    if not config.fix_thrust:
        lo, hi = config.thrust_bounds
        thrust_s_var = phase.state(
            'thrust_s',
            initial=config.thrust / F_SCALE,
            boundary=(lo / F_SCALE, hi / F_SCALE),
        )

    # Isp (если оптимизируется)
    Isp_s_var = None
    if not config.fix_Isp:
        lo, hi = config.Isp_bounds
        Isp_s_var = phase.state(
            'Isp_s',
            initial=config.Isp / ISP_SCALE,
            boundary=(lo / ISP_SCALE, hi / ISP_SCALE),
        )

    # Сухая масса (если оптимизируется)
    m_dry_s_var = None
    if not config.fix_m_dry:
        lo_dry, hi_dry = config.m_dry_bounds
        m_dry_s_var = phase.state(
            'm_dry_s',
            initial=config.m_dry / M_SCALE,
            boundary=(lo_dry / M_SCALE, hi_dry / M_SCALE),
        )
        m_lower_s = lo_dry / M_SCALE  # нижняя граница m_s (самый лёгкий вариант)
    else:
        m_lower_s = config.m_dry / M_SCALE

    # ── Масса топлива (если оптимизируется) ──────────────────────────────────
    # Примечание: fix_m_propellant=False имеет смысл ТОЛЬКО для фаз
    # с свободной длительностью (validate() проверяет это).
    m_prop_s_var = None
    if not config.fix_m_propellant:
        lo_p, hi_p = config.m_propellant_bounds
        m_prop_s_var = phase.state(
            'm_prop_s',
            initial=config.m_propellant / M_SCALE,
            boundary=(lo_p / M_SCALE, hi_p / M_SCALE),
        )

    # ── Масса аппарата ───────────────────────────────────────────────────────
    # Начальное условие:
    #   - первая фаза: m_initial = m_dry + m_propellant (числовое или символьное)
    #   - остальные:   m_initial = m_final(prev) - drop_prev
    #
    # Верхняя граница: максимально возможная начальная масса
    # Нижняя граница: m_lower_s (m_dry или min(m_dry_bounds))

    m_initial_s = config.m_initial / M_SCALE
    m_upper_s = m_initial_s  # верхняя граница

    if not config.fix_m_propellant:
        # Верхняя граница с учётом максимального топлива
        m_upper_s = (config.m_dry + config.m_propellant_bounds[1]) / M_SCALE

    if prev_states is None:
        # Первая фаза: начало = m_dry + m_propellant
        if m_prop_s_var is not None:
            # Символьная начальная масса (m_dry + m_prop_s.initial)
            m_dry_init_s = (m_dry_s_var.initial if m_dry_s_var is not None
                            else config.m_dry / M_SCALE)
            m_init_expr = m_dry_init_s + m_prop_s_var.initial
        elif m_dry_s_var is not None:
            m_init_expr = m_dry_s_var.initial + config.m_propellant / M_SCALE
        else:
            m_init_expr = m_initial_s

        m_s = phase.state('m_s',
                          initial=m_init_expr,
                          boundary=(m_lower_s, m_upper_s))
    else:
        # Не первая фаза: линковка = m_final(prev) - drop(prev)
        prev_drop_s = prev_states['mass_drop_s']
        m_link = prev_states['m_s'].final - prev_drop_s

        if m_prop_s_var is not None:
            # При оптимизируемом топливе добавим event-constraint ниже
            m_s = phase.state('m_s',
                              initial=m_link,
                              boundary=(m_lower_s, m_upper_s))
        else:
            m_s = phase.state('m_s',
                              initial=m_link,
                              boundary=(m_lower_s, m_upper_s))

    # Если оптимизируем m_propellant, добавляем event-constraint:
    # m_s.initial == m_dry_s + m_prop_s
    # Это связывает initial mass с design-переменными.
    # НО при нефиксированной начальной фазе initial уже задано через linkage —
    # добавляем constraint только для ПЕРВОЙ фазы:
    if m_prop_s_var is not None and prev_states is not None:
        # Для не-первой фазы: m_start = m_final(prev) - drop
        # Топливо = m_start - m_dry → m_prop_s определяется автоматически
        # через path-dynamics. Добавляем граничный constraint:
        m_dry_s_init = (m_dry_s_var.initial if m_dry_s_var is not None
                        else config.m_dry / M_SCALE)
        phase.event_constraints(
            m_s.initial == m_dry_s_init + m_prop_s_var.initial
        )

    # ── Управление ───────────────────────────────────────────────────────────
    u = [phase.control(f'u{i}', boundary=(-1.0, 1.0)) for i in range(3)]

    # ── Динамика и путевые ограничения ───────────────────────────────────────
    _phase_dynamics(phase, r_s, v_s, m_s, u, config,
                    thrust_s_var=thrust_s_var,
                    Isp_s_var=Isp_s_var,
                    m_dry_s_var=m_dry_s_var)

    # ── Сетка ────────────────────────────────────────────────────────────────
    phase.mesh(config.nodes_per_interval, config.tau_boundaries)

    mass_drop_s = mass_drop_kg / M_SCALE

    return {
        'phase': phase,
        'r_s': r_s,
        'v_s': v_s,
        'm_s': m_s,
        'u': u,
        'mass_drop_s': mass_drop_s,
        't_end': t_end,
        'thrust_s_var': thrust_s_var,
        'Isp_s_var': Isp_s_var,
        'm_dry_s_var': m_dry_s_var,
        'm_prop_s_var': m_prop_s_var,
    }


# =============================================================================
# Initial guess для одной фазы
# =============================================================================

def set_phase_guess(phase_result: dict, config: PhaseConfig,
                    r_const_s: list, v_const_s: list,
                    m_init_s: float, m_final_s: float):
    """
    Задать начальное приближение для одной фазы.

    ВАЖНО: Maptor ожидает СПИСОК массивов (по одному на интервал),
    а не один большой массив. Это точно соответствует формату
    _generate_phase_guess() из Maptor-примера.

    r_const_s, v_const_s — константные положение/скорость (масш.).
    m_init_s, m_final_s  — масса (масш.), линейная интерполяция.
    """
    phase = phase_result['phase']
    nodes = config.nodes_per_interval  # напр. [4, 4]

    n_base = 7
    extra_vals = []

    if phase_result['thrust_s_var'] is not None:
        extra_vals.append(config.thrust / F_SCALE)
    if phase_result['Isp_s_var'] is not None:
        extra_vals.append(config.Isp / ISP_SCALE)
    if phase_result['m_dry_s_var'] is not None:
        extra_vals.append(config.m_dry / M_SCALE)
    if phase_result['m_prop_s_var'] is not None:
        extra_vals.append(config.m_propellant / M_SCALE)

    n_states = n_base + len(extra_vals)
    N_total = sum(nodes)

    # Полная линейная интерполяция массы по всем узлам
    m_all = np.linspace(m_init_s, m_final_s, N_total + 1)

    # Список массивов — по одному на интервал (как в Maptor-примере)
    states_list = []
    controls_list = []
    idx = 0  # указатель на текущий узел в m_all

    for N in nodes:  # N — число коллокационных точек в интервале
        # states shape: (n_states, N+1)  — N+1 узлов на интервал
        s = np.zeros((n_states, N + 1))
        for i in range(3):
            s[i, :] = r_const_s[i]
            s[i + 3, :] = v_const_s[i]
        s[6, :] = m_all[idx: idx + N + 1]

        for k, val in enumerate(extra_vals):
            s[n_base + k, :] = val

        states_list.append(s)

        # controls shape: (3, N)
        c = np.zeros((3, N))
        c[0, :] = 1.0  # u = [1, 0, 0]  — как в Maptor-примере
        controls_list.append(c)

        idx += N  # следующий интервал начинается с последнего узла этого

    phase.guess(states=states_list, controls=controls_list)