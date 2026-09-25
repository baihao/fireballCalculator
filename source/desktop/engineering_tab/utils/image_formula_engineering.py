#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程估算 — 给定资料公式。

含铝演化（E 为 TNT 当量 kg，x 为含铝率小数，如 30% → 0.30）：
    R_max = 2.1405 * E^0.371 * exp[0.552 * (x - 0.3073)]   (m)
    v     = 196.72 * E^(-0.089) * exp[-1.78 * (x - 0.3073)] (m/s)
    t     = 0.0057 * E^0.426 * exp[0.630 * (x - 0.3073)]   (s)

    D_max = 2 * R_max
    t_m   = t   （达到最大半径/直径时间，与上式 t 同一符号）

火球总持续时间：
    t_d ≈ 0.30 * W^(1/3)   (s)，W 与 E 同为 TNT 当量 (kg)

图 1–2 等效辐射温度（使用上式 t_d）：
    T_eq = [ T_amb^4 + chi_r * E * H_TNT / (epsilon * sigma * kappa_A * 4 pi R_max^2 t_d) ]^(1/4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import symbols as sym
SIGMA_W_M2_K4 = 5.6704e-8
H_TNT_J_PER_KG = 4.184e6

DEFAULT_CHI_R = 0.4
DEFAULT_EPSILON = 0.43
DEFAULT_KAPPA_A = 0.8888
DEFAULT_SIGMA = SIGMA_W_M2_K4

# 含铝率参考点（公式指数项 x - 0.3073）
AL_REFERENCE_FRACTION = 0.3073


@dataclass(frozen=True)
class EngineeringEstimateInputs:
    equivalent_kg: float
    al_fraction: float
    t_amb_k: float
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
    t_eq_k: float

    @property
    def t_eq_c(self) -> float:
        return self.t_eq_k - 273.15


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
    x = float(x_fraction)
    r_max = 2.1405 * (e ** 0.371) * math.exp(0.552 * dx)
    v = 196.72 * (e ** (-0.089)) * math.exp(-1.78 * dx)
    t = 0.0057 * (e ** 0.426) * math.exp(0.630 * dx)
    return r_max, v, t


def fireball_total_duration_s(e_kg: float) -> float:
    """火球总持续时间 t_d ≈ 0.30·W^(1/3) (s)。"""
    if e_kg <= 0:
        raise ValueError("当量 W 必须大于 0")
    return 0.30 * (float(e_kg) ** (1.0 / 3.0))


def equivalent_radiation_temperature_k(
    *,
    e_kg: float,
    t_amb_k: float,
    r_max_m: float,
    t_d_s: float,
    chi_r: float = DEFAULT_CHI_R,
    epsilon: float = DEFAULT_EPSILON,
    kappa_a: float = DEFAULT_KAPPA_A,
    sigma: float = DEFAULT_SIGMA,
) -> float:
    if r_max_m <= 0 or t_d_s <= 0:
        raise ValueError(f"{sym.R_MAX_PLAIN} 与 {sym.T_D_PLAIN} 必须大于 0")
    if epsilon <= 0 or kappa_a <= 0 or sigma <= 0:
        raise ValueError(f"{sym.EPSILON}、{sym.KAPPA_A_PLAIN}、{sym.SIGMA} 必须大于 0")
    if chi_r < 0:
        raise ValueError(f"{sym.CHI_R_PLAIN} 不能为负")
    numerator = chi_r * e_kg * H_TNT_J_PER_KG
    denominator = epsilon * sigma * kappa_a * 4.0 * math.pi * (r_max_m ** 2) * t_d_s
    return (t_amb_k ** 4 + numerator / denominator) ** 0.25


def compute_engineering_estimate(inputs: EngineeringEstimateInputs) -> EngineeringEstimateResult:
    e = float(inputs.equivalent_kg)
    x = float(inputs.al_fraction)
    r_max, v, t_m = fireball_r_v_t_from_equivalent(e, x)
    d_max = 2.0 * r_max

    t_d = fireball_total_duration_s(e)

    t_eq = equivalent_radiation_temperature_k(
        e_kg=e,
        t_amb_k=float(inputs.t_amb_k),
        r_max_m=r_max,
        t_d_s=t_d,
        chi_r=float(inputs.chi_r),
        epsilon=float(inputs.epsilon),
        kappa_a=float(inputs.kappa_a),
        sigma=float(inputs.sigma),
    )

    return EngineeringEstimateResult(
        r_max_m=r_max,
        d_max_m=d_max,
        v_max_m_s=v,
        t_m_s=t_m,
        t_d_s=t_d,
        t_eq_k=t_eq,
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
