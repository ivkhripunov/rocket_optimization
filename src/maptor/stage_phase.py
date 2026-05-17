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

F_SCALE   = 1.0e6    # Н → МН
ISP_SCALE = 300.0    # с → безразмерный ~1


# =============================================================================
# Динамика
# =============================================================================

def _phase_dynamics(phase, r_s, v_s, m_s, u_dir, config,
                    thrust_s_var=None, Isp_s_var=None,
                    m_dry_s_var=None, throttle_ctrl=None):
    """
    Определить динамику и путевые ограничения фазы.

    u_dir        — список из 3 контролей, задающих НАПРАВЛЕНИЕ тяги (unit vector)
    throttle_ctrl — контроль [0,1] для масштабирования тяги (None → 1.0)
    """
    r_vec_s = ca.vertcat(*[r_s[i] for i in range(3)])
    v_vec_s = ca.vertcat(*[v_s[i] for i in range(3)])
    u_vec   = ca.vertcat(*[u_dir[i] for i in range(3)])

    r_vec  = r_vec_s * R_SCALE
    v_vec  = v_vec_s * V_SCALE
    m_phys = m_s * M_SCALE

    rad   = ca.sqrt(ca.fmax(ca.dot(r_vec, r_vec), 1e-12))
    h_alt = rad - EARTH_RAD

    # Гравитация
    grav = -(EARTH_MU / rad**3) * r_vec

    # Скорость относительно атмосферы
    vrel     = v_vec - OMEGA_MATRIX @ r_vec
    speedrel = ca.sqrt(ca.fmax(ca.dot(vrel, vrel), 1e-12))

    # Аэродинамическое сопротивление
    if config.use_atmosphere:
        rho    = config.rho_ref * ca.exp(-h_alt / config.h_scale)
        bc     = (rho / (2.0 * m_phys)) * config.CD * config.S
        drag   = -vrel * bc * speedrel
        q_dyn  = 0.5 * rho * speedrel**2
        q_heat = 1.7415e-4 * ca.sqrt(rho) * speedrel**3
    else:
        drag   = ca.MX.zeros(3, 1)
        q_dyn  = ca.MX(0.0)
        q_heat = ca.MX(0.0)

    # Тяга (физическая, Н)
    thrust_phys = thrust_s_var * F_SCALE if thrust_s_var is not None else config.thrust

    # Isp (физический, с)
    Isp_phys = Isp_s_var * ISP_SCALE if Isp_s_var is not None else config.Isp

    # Throttle (скаляр [0,1])
    throttle = throttle_ctrl if throttle_ctrl is not None else 1.0

    # Эффективная тяга
    F_eff = thrust_phys * throttle

    Toverm       = F_eff / m_phys
    thrust_accel = Toverm * u_vec
    mdot_phys    = -F_eff / (Isp_phys * G0)

    accel = thrust_accel + drag + grav

    # Масштабированные производные
    r_dot_s = v_vec  / R_SCALE
    v_dot_s = accel  / V_SCALE
    m_dot_s = mdot_phys / M_SCALE

    # Словарь динамики
    dyn = {}
    for i in range(3):
        dyn[r_s[i]] = r_dot_s[i]
        dyn[v_s[i]] = v_dot_s[i]
    dyn[m_s] = m_dot_s

    # Нулевые производные для const states (design vars)
    for var in (thrust_s_var, Isp_s_var, m_dry_s_var):
        if var is not None:
            dyn[var] = ca.MX(0.0)

    phase.dynamics(dyn)

    # Путевые ограничения
    path = []
    path.append(ca.dot(u_vec, u_vec) == 1.0)   # единичный вектор тяги
    path.append(h_alt >= -100.0)

    if m_dry_s_var is not None:
        # m_s не может опуститься ниже оптимизируемой сухой массы
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
# Вычисление границ m_s
# =============================================================================

def _mass_bounds(config: PhaseConfig) -> tuple:
    """
    Вернуть (m_lower_s, m_upper_s) — границы состояния массы в масш. ед.

    Нижняя граница: минимально возможная сухая масса.
        fix_m_dry=True  → config.m_dry / M_SCALE
        fix_m_dry=False → config.m_dry_bounds[0] / M_SCALE

    Верхняя граница: максимально возможная начальная масса.
        = max(m_dry) + max(m_prop)  из bounds или фиксированных значений.
    """
    # Нижняя
    if not config.fix_m_dry:
        m_lower_s = config.m_dry_bounds[0] / M_SCALE
    else:
        m_lower_s = config.m_dry / M_SCALE

    # Верхняя
    max_m_dry  = config.m_dry_bounds[1]  if not config.fix_m_dry        else config.m_dry
    max_m_prop = config.m_propellant_bounds[1] if not config.fix_m_propellant else config.m_propellant
    m_upper_s  = (max_m_dry + max_m_prop) / M_SCALE

    return m_lower_s, m_upper_s


# =============================================================================
# Вычисление символьного выражения m_dry + m_prop
# =============================================================================

def _mass_design_expr(config: PhaseConfig,
                      m_dry_s_var=None,
                      m_prop_s_var=None):
    """
    Вернуть символьное выражение m_dry_s + m_prop_s в масштабированных единицах.

    Для фиксированных параметров подставляет числовые константы.
    Для оптимизируемых — использует Maptor-state (= CasADi-символ).
    """
    m_dry_expr  = m_dry_s_var   if m_dry_s_var  is not None else config.m_dry        / M_SCALE
    m_prop_expr = m_prop_s_var  if m_prop_s_var is not None else config.m_propellant / M_SCALE
    return m_dry_expr + m_prop_expr


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

    Линковка масс между фазами (для не-первых фаз):

        m_s.initial = m_prev.final - drop            [линковка, всегда]

    Если хотя бы один из m_dry / m_propellant оптимизируется, добавляется
    event-constraint:

        m_prev.final - drop == m_dry_expr + m_prop_expr   [design constraint]

    Это одно уравнение с (до) двумя свободными переменными (m_dry_s_var, m_prop_s_var),
    поэтому оптимизатор может выбирать, как разделить стартовую массу на структуру
    и топливо.

    Parameters
    ----------
    mass_drop_kg : масса, сбрасываемая ПОСЛЕ этой фазы — фиксированный аппаратный
                   параметр, независимый от m_dry/m_propellant.
    """
    config.validate()

    phase = problem.set_phase(config.phase_id)

    # ── Время ────────────────────────────────────────────────────────────────
    if config.fix_duration:
        t_end = t_start + config.duration
        phase.time(initial=t_start, final=t_end)
    else:
        t_lo, t_hi = config.duration_bounds
        t_end = t_start + config.duration
        phase.time(initial=t_start, final=(t_lo, t_hi))

    # ── Положение ────────────────────────────────────────────────────────────
    if prev_states is None:
        assert r0_s is not None
        r_s = [phase.state(f'r{i}_s', initial=r0_s[i], boundary=(R_MIN, R_MAX))
               for i in range(3)]
    else:
        r_s = [phase.state(f'r{i}_s', initial=prev_states['r_s'][i].final,
                           boundary=(R_MIN, R_MAX))
               for i in range(3)]

    # ── Скорость ─────────────────────────────────────────────────────────────
    if prev_states is None:
        assert v0_s is not None
        v_s = [phase.state(f'v{i}_s', initial=v0_s[i], boundary=(V_MIN, V_MAX))
               for i in range(3)]
    else:
        v_s = [phase.state(f'v{i}_s', initial=prev_states['v_s'][i].final,
                           boundary=(V_MIN, V_MAX))
               for i in range(3)]

    # ── Design-переменные: тяга ───────────────────────────────────────────────
    thrust_s_var = None
    if not config.fix_thrust:
        lo, hi = config.thrust_bounds
        thrust_s_var = phase.state('thrust_s',
                                   initial=config.thrust / F_SCALE,
                                   boundary=(lo / F_SCALE, hi / F_SCALE))

    # ── Design-переменные: Isp ────────────────────────────────────────────────
    Isp_s_var = None
    if not config.fix_Isp:
        lo, hi = config.Isp_bounds
        Isp_s_var = phase.state('Isp_s',
                                initial=config.Isp / ISP_SCALE,
                                boundary=(lo / ISP_SCALE, hi / ISP_SCALE))

    # ── Design-переменные: m_dry ──────────────────────────────────────────────
    m_dry_s_var = None
    if not config.fix_m_dry:
        lo, hi = config.m_dry_bounds
        m_dry_s_var = phase.state('m_dry_s',
                                  initial=config.m_dry / M_SCALE,
                                  boundary=(lo / M_SCALE, hi / M_SCALE))

    # ── Design-переменные: m_propellant ───────────────────────────────────────
    m_prop_s_var = None
    if not config.fix_m_propellant:
        lo, hi = config.m_propellant_bounds
        m_prop_s_var = phase.state('m_prop_s',
                                   initial=config.m_propellant / M_SCALE,
                                   boundary=(lo / M_SCALE, hi / M_SCALE))

    # ── Throttle ──────────────────────────────────────────────────────────────
    throttle_ctrl = None
    if not config.fix_throttle:
        lo, hi = config.throttle_bounds
        throttle_ctrl = phase.control('throttle', boundary=(lo, hi))

    # ── Границы m_s ───────────────────────────────────────────────────────────
    m_lower_s, m_upper_s = _mass_bounds(config)

    # ── Символьное выражение m_dry + m_propellant ─────────────────────────────
    #    Используется как в initial condition, так и в event constraint.
    #    Для const states: state object IS the symbolic value (initial = final = var).
    design_expr = _mass_design_expr(config, m_dry_s_var, m_prop_s_var)
    has_design_mass = (m_dry_s_var is not None or m_prop_s_var is not None)

    # ── Состояние массы ───────────────────────────────────────────────────────
    if prev_states is None:
        # ┌─ Первая фаза ──────────────────────────────────────────────────────
        # initial = m_dry + m_propellant (числовое или символьное)
        #
        # Если оба фиксированы — это просто число.
        # Если хоть один оптимизируется — Maptor принимает CasADi-выражение.
        m_s = phase.state('m_s',
                          initial=design_expr,
                          boundary=(m_lower_s, m_upper_s))
    else:
        # ┌─ Не первая фаза ───────────────────────────────────────────────────
        # Линковка: m_start(B) = m_final(A) - drop_after_A
        prev_drop_s = prev_states['mass_drop_s']
        m_link      = prev_states['m_s'].final - prev_drop_s

        m_s = phase.state('m_s',
                          initial=m_link,
                          boundary=(m_lower_s, m_upper_s))

        if has_design_mass:
            # Event constraint: m_link == m_dry_B + m_prop_B
            #
            # Физический смысл: масса, пришедшая через линковку (m_prev - drop),
            # должна равняться заявленной стартовой массе этой фазы (m_dry + m_prop).
            #
            # Это одно уравнение с ≤2 свободными переменными (m_dry_s_var, m_prop_s_var)
            # → оптимизатор выбирает распределение, удовлетворяющее ограничению.
            #
            # m_link = m_prev_states['m_s'].final - prev_drop_s (CasADi-символ),
            # design_expr = m_dry_s_var + m_prop_s_var (тоже символ).
            # Maptor принимает cross-phase символьные ссылки в event_constraints.
            phase.event_constraints(m_link == design_expr)

    # ── Управление: направление тяги ─────────────────────────────────────────
    u_dir = [phase.control(f'u{i}', boundary=(-1.0, 1.0)) for i in range(3)]

    # ── Динамика + путевые ограничения ────────────────────────────────────────
    _phase_dynamics(phase, r_s, v_s, m_s, u_dir, config,
                    thrust_s_var=thrust_s_var,
                    Isp_s_var=Isp_s_var,
                    m_dry_s_var=m_dry_s_var,
                    throttle_ctrl=throttle_ctrl)

    # ── Сетка ────────────────────────────────────────────────────────────────
    phase.mesh(config.nodes_per_interval, config.tau_boundaries)

    return {
        'phase':         phase,
        'r_s':           r_s,
        'v_s':           v_s,
        'm_s':           m_s,
        'u_dir':         u_dir,
        'throttle_ctrl': throttle_ctrl,
        'mass_drop_s':   mass_drop_kg / M_SCALE,
        't_end':         t_end,
        'thrust_s_var':  thrust_s_var,
        'Isp_s_var':     Isp_s_var,
        'm_dry_s_var':   m_dry_s_var,
        'm_prop_s_var':  m_prop_s_var,
    }


# =============================================================================
# Initial guess
# =============================================================================

def set_phase_guess(phase_result: dict, config: PhaseConfig,
                    r_const_s: list, v_const_s: list,
                    m_init_s: float, m_final_s: float):
    """
    Задать начальное приближение для одной фазы.

    Maptor ожидает СПИСОК массивов — по одному на интервал сетки.

    Порядок строк в массиве states:
        r0_s, r1_s, r2_s, v0_s, v1_s, v2_s, m_s,
        [thrust_s], [Isp_s], [m_dry_s], [m_prop_s]
    """
    phase = phase_result['phase']
    nodes = config.nodes_per_interval

    # Значения design-переменных для guess
    extra_vals = []
    if phase_result['thrust_s_var']  is not None: extra_vals.append(config.thrust        / F_SCALE)
    if phase_result['Isp_s_var']     is not None: extra_vals.append(config.Isp           / ISP_SCALE)
    if phase_result['m_dry_s_var']   is not None: extra_vals.append(config.m_dry         / M_SCALE)
    if phase_result['m_prop_s_var']  is not None: extra_vals.append(config.m_propellant  / M_SCALE)

    n_states = 7 + len(extra_vals)
    # Throttle — control, не state, не добавляем в states
    n_controls = 3 + (1 if phase_result['throttle_ctrl'] is not None else 0)

    N_total = sum(nodes)
    m_all   = np.linspace(m_init_s, m_final_s, N_total + 1)

    states_list   = []
    controls_list = []
    idx = 0

    for N in nodes:
        s = np.zeros((n_states, N + 1))
        for i in range(3):
            s[i,   :] = r_const_s[i]
            s[i+3, :] = v_const_s[i]
        s[6, :] = m_all[idx: idx + N + 1]
        for k, val in enumerate(extra_vals):
            s[7 + k, :] = val
        states_list.append(s)

        c = np.zeros((n_controls, N))
        c[0, :] = 1.0    # u_dir = [1, 0, 0]
        if phase_result['throttle_ctrl'] is not None:
            c[3, :] = config.throttle   # throttle guess
        controls_list.append(c)

        idx += N

    phase.guess(states=states_list, controls=controls_list)