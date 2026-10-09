#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程估算 — 给定资料公式。

含铝演化（E / W / M 为 TNT 当量兼装药质量 kg，x 为含铝率小数，如 30% → 0.30）：
    R_max = 2.1405 * E^0.371 * exp[0.552 * (x - 0.3073)]   (m)
    v     = 196.72 * E^(-0.089) * exp[-1.78 * (x - 0.3073)] (m/s)
    t     = 0.0057 * E^0.426 * exp[0.630 * (x - 0.3073)]   (s)

    D_max = 2 * R_max
    t_m   = t   （达到最大半径/直径时间，与上式 t 同一符号）

火球总持续时间（展示 / 仿真时长）：
    t_d ≈ 0.30 * W^(1/3)   (s)，W 与 E、M 同为 TNT 当量 / 装药质量 (kg)

火球等效温度（资料闭式，M 取当量 W）：
    u = 13.16169 * t / M^(1/3)
    T(t; M, x) = 300 + (1355 + 150x) e^{-2u}
               + 5901.38 M^{0.07} (0.82 + 0.60x)
                 * (e^{-u} - 14/13 e^{-2u} + 1/13 e^{-15u})
    T_max = max_t T(t; M, x)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 火球温度预测上限 (K)：超过则截断
MAX_TEMPERATURE_K = 3000.0

# 含铝率参考点（公式指数项 x - 0.3073）
AL_REFERENCE_FRACTION = 0.3073

# T(t;M,x) 中 u = U_COEF * t / M^(1/3)
U_COEF = 13.16169
T_BASE_K = 300.0
T_TRANSIENT_BASE = 1355.0
T_TRANSIENT_AL = 150.0
T_SCALE_COEF = 5901.38
T_SCALE_M_EXP = 0.07
T_SCALE_AL_BASE = 0.82
T_SCALE_AL_COEF = 0.60

# 兼容旧调用签名（温度公式不再使用）
DEFAULT_CHI_R = 0.4
DEFAULT_EPSILON = 0.43
DEFAULT_KAPPA_A = 0.8888
DEFAULT_SIGMA = 5.6704e-8
DEFAULT_T_AMB_K = 297.15


@dataclass(frozen=True)
class EngineeringEstimateInputs:
    equivalent_kg: float
    al_fraction: float
    t_amb_k: float = DEFAULT_T_AMB_K
    chi_r: float = DEFAULT_CHI_R
    epsilon: float = DEFAULT_EPSILON
    kappa_a: float = DEFAULT_KAPPA_A
    sigma: float = DEFAULT_SIGMA


@dataclass(frozen=True)
class EngineeringEstimateResult:
    r_max_m: float
    d_max_m: float
    v_max_m_s: float
    t_m_s: float
    t_d_s: float
    t_max_k: float
    t_at_tmax_s: float

    @property
    def t_eq_k(self) -> float:
        """兼容旧字段名：现为公式峰值温度 T_max。"""
        return self.t_max_k

    @property
    def t_max_c(self) -> float:
        return self.t_max_k - 273.15

    @property
    def t_eq_c(self) -> float:
        return self.t_max_c


def _al_exp_shift(x_fraction: float) -> float:
    return float(x_fraction) - AL_REFERENCE_FRACTION


def fireball_r_v_t_from_equivalent(e_kg: float, x_fraction: float) -> tuple[float, float, float]:
    """R_max (m), v (m/s), t (s)。"""
    if e_kg <= 0:
        raise ValueError("当量 E 必须大于 0")
    if x_fraction < 0:
        raise ValueError("含铝率 x 不能为负")
    dx = _al_exp_shift(x_fraction)
    e = float(e_kg)
    r_max = 2.1405 * (e ** 0.371) * math.exp(0.552 * dx)
    v = 196.72 * (e ** (-0.089)) * math.exp(-1.78 * dx)
    t = 0.0057 * (e ** 0.426) * math.exp(0.630 * dx)
    return r_max, v, t


def fireball_total_duration_s(e_kg: float) -> float:
    """火球总持续时间 t_d ≈ 0.30·W^(1/3) (s)。"""
    if e_kg <= 0:
        raise ValueError("当量 W 必须大于 0")
    return 0.30 * (float(e_kg) ** (1.0 / 3.0))


def _temperature_scale_c(m_kg: float, x_fraction: float) -> tuple[float, float]:
    """返回 (A, C)：瞬态系数与含 M、x 的尺度系数。"""
    x = float(x_fraction)
    m = float(m_kg)
    a = T_TRANSIENT_BASE + T_TRANSIENT_AL * x
    c = T_SCALE_COEF * (m ** T_SCALE_M_EXP) * (T_SCALE_AL_BASE + T_SCALE_AL_COEF * x)
    return a, c


def temperature_from_u(u: float, m_kg: float, x_fraction: float) -> float:
    """由无量纲时间 u 计算 T(u; M, x) [K]。"""
    a, c = _temperature_scale_c(m_kg, x_fraction)
    eu = math.exp(-u)
    e2u = math.exp(-2.0 * u)
    e15u = math.exp(-15.0 * u)
    return T_BASE_K + a * e2u + c * (eu - (14.0 / 13.0) * e2u + (1.0 / 13.0) * e15u)


def fireball_temperature_k(t_s: float, m_kg: float, x_fraction: float) -> float:
    """T(t; M, x) [K]；t 为爆炸后时间 (s)。"""
    if m_kg <= 0:
        raise ValueError("装药质量 M 必须大于 0")
    if x_fraction < 0:
        raise ValueError("含铝率 x 不能为负")
    if t_s < 0:
        raise ValueError("时间 t 不能为负")
    u = U_COEF * float(t_s) / (float(m_kg) ** (1.0 / 3.0))
    return temperature_from_u(u, m_kg, x_fraction)


def fireball_max_temperature_k(m_kg: float, x_fraction: float) -> tuple[float, float]:
    """
    对 t≥0 求 T(t;M,x) 的最大值。

    Returns:
        (T_max_K, t_at_Tmax_s)
    """
    if m_kg <= 0:
        raise ValueError("装药质量 M 必须大于 0")
    if x_fraction < 0:
        raise ValueError("含铝率 x 不能为负")

    m = float(m_kg)
    x = float(x_fraction)
    # 在 u 空间粗搜 + 黄金分割细化（u∝t，峰值通常在较小 u）
    u_hi = 20.0
    n_grid = 4000
    best_u = 0.0
    best_t = temperature_from_u(0.0, m, x)
    for i in range(1, n_grid + 1):
        u = u_hi * i / n_grid
        tv = temperature_from_u(u, m, x)
        if tv > best_t:
            best_t = tv
            best_u = u

    du = u_hi / n_grid
    lo = max(0.0, best_u - 3.0 * du)
    hi = best_u + 3.0 * du
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        if hi - lo < 1e-12:
            break
        u1 = hi - phi * (hi - lo)
        u2 = lo + phi * (hi - lo)
        t1 = temperature_from_u(u1, m, x)
        t2 = temperature_from_u(u2, m, x)
        if t1 < t2:
            lo = u1
            if t2 > best_t:
                best_t = t2
                best_u = u2
        else:
            hi = u2
            if t1 > best_t:
                best_t = t1
                best_u = u1

    t_at = best_u * (m ** (1.0 / 3.0)) / U_COEF
    return min(float(best_t), MAX_TEMPERATURE_K), float(t_at)


def compute_engineering_estimate(inputs: EngineeringEstimateInputs) -> EngineeringEstimateResult:
    e = float(inputs.equivalent_kg)
    x = float(inputs.al_fraction)
    r_max, v, t_m = fireball_r_v_t_from_equivalent(e, x)
    d_max = 2.0 * r_max
    t_d = fireball_total_duration_s(e)
    # 公式中 M 取界面 TNT 当量 W（装药质量约定）
    t_max, t_at_tmax = fireball_max_temperature_k(e, x)

    return EngineeringEstimateResult(
        r_max_m=r_max,
        d_max_m=d_max,
        v_max_m_s=v,
        t_m_s=t_m,
        t_d_s=t_d,
        t_max_k=t_max,
        t_at_tmax_s=t_at_tmax,
    )


def format_engineering_result(
    inputs: EngineeringEstimateInputs,
    result: EngineeringEstimateResult,
) -> str:
    """兼容旧调用：返回 HTML 字符串（供 QTextEdit.setHtml）。"""
    from .result_html import format_engineering_result_html

    return format_engineering_result_html(inputs, result)


def parse_al_fraction(al_percent: float) -> float:
    return float(al_percent) / 100.0
