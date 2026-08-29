#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数预测 — 工程计算公式说明与当前工况参数值。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from fireball_heat_radiation_calculator import EPSILON, SIGMA

from .calculator import (
    DEFAULT_HEAT_FLUX_DISTANCES_M,
    DEFAULT_RADIATION_X,
    REFERENCE_EQUIVALENT_KG,
    REFERENCE_DURATION_MS,
)

_KBC_SOURCE_LABELS = {
    "krr": "核岭回归预测",
    "calculator": "计算器当量缩放",
    "explicit_kbc": "显式拖曳曲线（用户/模型给定 K,B,C）",
    "calculator_scaled": "计算器当量缩放",
}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "—"


def build_formula_reference_text(
    *,
    is_equivalent_mode: bool,
    equivalent: Optional[float] = None,
    al_percent: Optional[float] = None,
    k: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    env_temp: Optional[float] = None,
    env_humidity: Optional[float] = None,
    env_pressure: Optional[float] = None,
    duration: Optional[float] = None,
    material_name: str = "",
    kbc_source: str = "",
    standard_equivalent: Optional[float] = None,
    equivalent_ratio: Optional[float] = None,
    kbc_display: Optional[Tuple[float, float, float]] = None,
    temperature_time_scale: Optional[float] = None,
    has_training_temperature: bool = False,
    heat_flux_distances: Sequence[float] = DEFAULT_HEAT_FLUX_DISTANCES_M,
) -> str:
    """生成公式说明文本；未计算时根据侧栏输入展示当前参数。"""
    lines: List[str] = []
    mode_text = "当量仿真" if is_equivalent_mode else "参数仿真"
    lines.append(f"【当前模式】{mode_text}")
    lines.append("")

    kd = b_val = c_val = None
    if kbc_display is not None:
        kd, b_val, c_val = kbc_display
    elif not is_equivalent_mode:
        kd, b_val, c_val = k, b, c

    src_text = _KBC_SOURCE_LABELS.get(kbc_source, kbc_source or "—")

    lines.append("【1. 火球直径时序（拖曳曲线）】")
    lines.append("  D(t) = K × [1 − B × exp(−C × t²)]")
    lines.append("  · t：时间 (s)；界面输入/采样以 ms 计，内部换算 t_s = t_ms / 1000")
    lines.append("  · C 在公式中与 t²(s²) 相乘，代码中 C_eff = C × 10⁶")
    if kd is not None and b_val is not None and c_val is not None:
        c_eff = float(c_val) * 1e6
        lines.append(f"  · 来源：{src_text}")
        lines.append(f"  · K = {_fmt(kd)} m，B = {_fmt(b_val)}，C = {_fmt(c_val)}，C_eff = {_fmt(c_eff)}")
    else:
        lines.append(f"  · 来源：{src_text}")
        lines.append("  · K, B, C：—（待计算或待输入）")
    lines.append("")

    if is_equivalent_mode:
        lines.append("【2. 当量缩放关系】")
        lines.append("  M = m / m_std")
        lines.append("  K_d = 2 × √M × K_rad,std")
        lines.append("  B_eff = B_std")
        lines.append("  C_eff = C_std / M")
        if equivalent is not None and standard_equivalent is not None:
            lines.append(
                f"  · m = {_fmt(equivalent)} kg TNT，m_std = {_fmt(standard_equivalent)} kg TNT"
            )
            if equivalent_ratio is not None:
                lines.append(f"  · M = {_fmt(equivalent_ratio)}")
        elif equivalent is not None:
            lines.append(f"  · m = {_fmt(equivalent)} kg TNT，m_std = —")
        else:
            lines.append("  · m, m_std, M：—")
        if material_name:
            lines.append(f"  · 材料档：{material_name}")
        if al_percent is not None:
            lines.append(f"  · 含铝量：{_fmt(al_percent)} %")
        lines.append("")
    else:
        lines.append("【2. 当量缩放关系】")
        lines.append("  参数仿真模式下直径直接使用输入 K,B,C，不经过当量缩放。")
        lines.append("")

    lines.append("【3. 火球温度时序】")
    if has_training_temperature:
        lines.append("  T(t)：由导入模型中的实验温度序列线性插值得到")
    else:
        lines.append("  参考曲线：100 kg TNT 标定温度剖面 T_ref(t_ref)")
        lines.append("  t_ref = t / scale，scale = (m / 100)^(2/3)")
        lines.append(f"  · 标定当量 m_ref = {_fmt(REFERENCE_EQUIVALENT_KG)} kg TNT")
        lines.append(f"  · 标定参考时长 ≈ {_fmt(REFERENCE_DURATION_MS)} ms")
        if temperature_time_scale is not None:
            lines.append(f"  · scale = {_fmt(temperature_time_scale)}")
        elif equivalent is not None:
            scale = (float(equivalent) / REFERENCE_EQUIVALENT_KG) ** (2.0 / 3.0)
            lines.append(f"  · scale = (m/100)^(2/3) = {_fmt(scale)}")
        else:
            lines.append("  · scale：—")
    lines.append("")

    ta_k = (float(env_temp) + 273.15) if env_temp is not None else None
    dist_text = ", ".join(f"{d:g}" for d in heat_flux_distances)

    lines.append("【4. 热通量】")
    lines.append("  E(t) = ε × σ × T(t)⁴")
    lines.append("  F(x, t) = (1/4) × [D(t) / x]²")
    lines.append("  q(x, t) = E(t) × F(x, t) × τ(x)")
    lines.append(f"  · ε = {_fmt(EPSILON)}，σ = {_fmt(SIGMA, 3)} W/(m²·K⁴)")
    if ta_k is not None:
        lines.append(f"  · T_a = {_fmt(env_temp)} °C = {_fmt(ta_k)} K")
    else:
        lines.append("  · T_a：—")
    lines.append(f"  · 图表采样距离 x = {dist_text} m")
    lines.append("")

    lines.append("【5. 大气透射率 τ(x)】")
    lines.append("  X_CO2 = (273 / T_a) × x")
    lines.append("  X_H2O = (288.651 / T_a) × (760 / 101325) × (RH/100 × Pw,sat) × x")
    lines.append(
        "  τ = 1.006 − 0.1171·log₁₀(X_H2O) − 0.02368·[log₁₀(X_H2O)]²"
        " − 0.03188·log₁₀(X_CO2) + 0.001164·[log₁₀(X_CO2)]²"
    )
    if env_humidity is not None and env_pressure is not None:
        lines.append(
            f"  · RH = {_fmt(env_humidity)} %，Pw,sat = {_fmt(env_pressure)} Pa"
        )
    else:
        lines.append("  · RH, Pw,sat：—")
    lines.append("")

    lines.append("【6. 累积热辐射】")
    lines.append("  H(x) = ∫₀^T q(x, t) dt   （梯形数值积分，T 为仿真时长）")
    lines.append("  输出单位：kJ/m²（J/m² ÷ 1000）")
    lines.append(
        f"  · 距离采样 x ∈ [{_fmt(DEFAULT_RADIATION_X[0])}, {_fmt(DEFAULT_RADIATION_X[1])}] m，"
        f"共 {int(DEFAULT_RADIATION_X[2])} 点"
    )
    if duration is not None:
        lines.append(f"  · 积分上限 T = {_fmt(duration)} ms")
    else:
        lines.append("  · 积分上限 T：—")

    return "\n".join(lines)


def build_formula_reference_from_prediction(
    prediction_data: Dict[str, Any],
    *,
    is_equivalent_mode: bool,
    al_percent: float,
    kbc_source: str = "",
) -> str:
    """根据一次成功计算后的 ``prediction_data`` 刷新公式面板。"""
    std_eq = None
    if prediction_data.get("equivalent_ratio") is not None and prediction_data.get("equivalent"):
        m = float(prediction_data["equivalent"])
        ratio = float(prediction_data["equivalent_ratio"])
        if ratio > 0:
            std_eq = m / ratio

    return build_formula_reference_text(
        is_equivalent_mode=is_equivalent_mode,
        equivalent=prediction_data.get("equivalent"),
        al_percent=al_percent,
        env_temp=prediction_data.get("env_temp"),
        env_humidity=prediction_data.get("env_humidity"),
        env_pressure=prediction_data.get("env_pressure"),
        duration=prediction_data.get("duration"),
        material_name=str(prediction_data.get("material_name", "")),
        kbc_source=kbc_source or str(prediction_data.get("kbc_source", "")),
        standard_equivalent=std_eq,
        equivalent_ratio=prediction_data.get("equivalent_ratio"),
        kbc_display=prediction_data.get("kbc_display"),
        temperature_time_scale=prediction_data.get("temperature_time_scale"),
        has_training_temperature=prediction_data.get("has_training_temperature", False),
    )
