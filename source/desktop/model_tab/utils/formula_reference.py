#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数预测 — 仿真结果摘要（右侧面板「仿真结果」）。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calculator import (
    DEFAULT_SIMULATION_EQUIVALENT_KG,
    preview_temperature_series_for_heat_flux,
    summarize_max_temperature_heat_flux,
)
from .simulation_log import NEAR_MAX_DIAMETER_FRACTION, _time_to_fraction_of_max

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
        v = float(value)
        if np.isnan(v):
            return "—"
        return f"{v:.{digits}g}"
    except (TypeError, ValueError):
        return "—"


def avg_radial_velocity_to_90pct_max_radius(
    k_diameter_m: float,
    b: float,
    c_material: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    拖曳曲线 D(t)=K(1−B·e^{−C_eff·t_s²}) 下，从 R(0)=0 到 0.9·R_max 的平均径向速度。

    Returns:
        (t_90_ms, r0_m, r90_m, v_avg_m_per_ms, v_avg_m_per_s) 或 None（B≤0.1 等）
    """
    k = float(k_diameter_m)
    b_val = float(b)
    c_val = float(c_material)
    if k <= 0 or b_val <= 0.1 or c_val <= 0:
        return None
    c_s = c_val * 1e6
    t90_ms = math.sqrt(math.log(b_val / 0.1) / c_s) * 1000.0
    if t90_ms <= 0 or not math.isfinite(t90_ms):
        return None
    r0 = 0.0
    r90 = 0.45 * k
    v_mm = (r90 - r0) / t90_ms
    return t90_ms, r0, r90, v_mm, v_mm * 1000.0


def _kbc_triple(
    *,
    is_equivalent_mode: bool,
    k: Optional[float],
    b: Optional[float],
    c: Optional[float],
    kbc_display: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    if kbc_display is not None:
        return float(kbc_display[0]), float(kbc_display[1]), float(kbc_display[2])
    if not is_equivalent_mode and k is not None and b is not None and c is not None:
        return float(k), float(b), float(c)
    return None


def _append_temperature_heat_flux_lines(
    lines: List[str],
    summary: Dict[str, float],
    *,
    preview: bool = False,
) -> None:
    lines.append("【温度】（与热通量 q(x,t) 仿真同一 T(t)）")
    if not isinstance(summary, dict) or len(summary) == 0:
        lines.append("  最大温度 —（需填写仿真时长与当量，或完成一次计算）")
        return
    prefix = "  预览 " if preview else "  "
    lines.append(
        f"{prefix}最大温度 {_fmt(summary.get('T_C'), 4)} °C"
        f"（{_fmt(summary.get('T_K'), 4)} K）@ t={_fmt(summary.get('at_ms'), 4)} ms"
    )
    if summary.get("reference_distance_m") is not None:
        lines.append(
            f"{prefix}参考距离 x={_fmt(summary.get('reference_distance_m'), 4)} m："
            f"峰值热通量 {_fmt(summary.get('peak_heat_flux_W_m2'), 4)} W/m² "
            f"@ t={_fmt(summary.get('peak_heat_flux_at_ms'), 4)} ms"
        )
        if summary.get("T_at_peak_heat_flux_K") is not None:
            lines.append(
                f"{prefix}该时刻温度 "
                f"{_fmt(summary.get('T_at_peak_heat_flux_C'), 4)} °C"
                f"（{_fmt(summary.get('T_at_peak_heat_flux_K'), 4)} K）"
            )


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
    heat_flux_distances: Sequence[float] = (),  # 保留签名兼容
    preview_equivalent_kg: Optional[float] = None,
    training_temperature_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    peak_temperature_k: Optional[float] = None,
) -> str:
    """未执行仿真时：展示工况与可由 K,B,C 预估的指标。"""
    lines: List[str] = []
    lines.append("【仿真结果】尚未计算，完成「开始计算」后刷新本面板。")
    lines.append("")
    lines.append("【工况】")
    if is_equivalent_mode:
        if equivalent is not None:
            lines.append(f"  当量 {_fmt(equivalent)} kg TNT")
        if al_percent is not None:
            lines.append(f"  含铝 {_fmt(al_percent)} %")
        if material_name:
            lines.append(f"  材料档 {material_name}")
    else:
        lines.append("  模式：参数仿真（K / B / C）")
        if peak_temperature_k is not None:
            lines.append(f"  火球最大温度 {_fmt(peak_temperature_k)} K")
    if duration is not None:
        lines.append(f"  仿真时长 {_fmt(duration)} ms")
    lines.append("")

    kbc = _kbc_triple(
        is_equivalent_mode=is_equivalent_mode,
        k=k,
        b=b,
        c=c,
        kbc_display=kbc_display,
    )
    if kbc is not None:
        kd, b_val, c_val = kbc
        lines.append("【拖曳参数（预览）】")
        lines.append(f"  K = {_fmt(kd)} m，B = {_fmt(b_val)}，C = {_fmt(c_val)}")
        vel = avg_radial_velocity_to_90pct_max_radius(kd, b_val, c_val)
        if vel is not None:
            t90, r0, r90, v_mm, v_s = vel
            lines.append("")
            lines.append("【膨胀（由 K,B,C 预估，t=0 起算）】")
            lines.append(f"  R(0) = 0 m，0.9·R_max = {_fmt(r90, 4)} m")
            lines.append(f"  t_90% = {_fmt(t90, 4)} ms")
            lines.append(f"  平均径向速度 = {_fmt(v_mm, 4)} m/ms（{_fmt(v_s, 4)} m/s）")
    else:
        lines.append("【拖曳参数】K, B, C —（待输入或待预测）")

    eq_kg = preview_equivalent_kg
    if eq_kg is None:
        eq_kg = equivalent if equivalent is not None else DEFAULT_SIMULATION_EQUIVALENT_KG
    temp_preview: Dict[str, float] = {}
    if duration is not None and float(duration) > 0:
        al_frac = float(al_percent) / 100.0 if al_percent is not None else 0.30
        t_amb = (float(env_temp) + 273.15) if env_temp is not None else 297.15
        param_only = not is_equivalent_mode and k is not None and b is not None and c is not None
        series = preview_temperature_series_for_heat_flux(
            float(duration),
            float(eq_kg) if eq_kg is not None and not param_only else None,
            training_temperature_data,
            al_fraction=al_frac,
            t_amb_k=t_amb,
            parameter_simulation=param_only,
            peak_temperature_k=peak_temperature_k if param_only else None,
        )
        if series is not None:
            t_prev, t_k_prev = series
            temp_preview = summarize_max_temperature_heat_flux(t_prev, t_k_prev, {})
    lines.append("")
    _append_temperature_heat_flux_lines(
        lines,
        temp_preview,
        preview=isinstance(temp_preview, dict) and len(temp_preview) > 0,
    )

    if env_temp is not None:
        lines.append("")
        lines.append("【环境】")
        lines.append(
            f"  T_a = {_fmt(env_temp)} °C，RH = {_fmt(env_humidity)} %，"
            f"Pw,sat = {_fmt(env_pressure)} Pa"
        )

    return "\n".join(lines)


def build_formula_reference_from_prediction(
    prediction_data: Dict[str, Any],
    *,
    al_percent: float,
    kbc_source: str = "",
) -> str:
    """根据一次成功计算后的 ``prediction_data`` 刷新仿真结果面板。"""
    lines: List[str] = []
    t_ms = np.asarray(prediction_data["time_ms"], dtype=np.float64)
    d_m = np.asarray(prediction_data["diameter_data"], dtype=np.float64)
    raw_heat = prediction_data.get("heat_flux_data")
    heat_store = raw_heat if isinstance(raw_heat, dict) else {}
    temp_summary = prediction_data.get("max_temperature_heat_flux")
    if not isinstance(temp_summary, dict) or len(temp_summary) == 0:
        t_k = np.asarray(prediction_data.get("temperature_data", []), dtype=np.float64)
        temp_summary = summarize_max_temperature_heat_flux(t_ms, t_k, heat_store)

    kbc = prediction_data.get("kbc_display")
    if kbc is not None:
        kd, b_val, c_val = float(kbc[0]), float(kbc[1]), float(kbc[2])
    else:
        kd = b_val = c_val = None

    src = kbc_source or str(prediction_data.get("kbc_source", ""))
    src_text = _KBC_SOURCE_LABELS.get(src, src or "—")

    lines.append("【仿真结果】")
    lines.append(
        f"  时长 {_fmt(prediction_data.get('duration', t_ms[-1]))} ms｜"
        f"采样 {t_ms.size} 点"
    )
    if prediction_data.get("simulation_mode") != "parameter":
        eq = prediction_data.get("equivalent")
        mat = prediction_data.get("material_name") or "—"
        if eq is not None:
            lines.append(
                f"  当量 {_fmt(eq)} kg TNT｜含铝 {_fmt(al_percent)} %｜{mat}"
            )
        m_ratio = prediction_data.get("equivalent_ratio")
        if m_ratio is not None:
            lines.append(f"  M = {_fmt(m_ratio)}")
    else:
        lines.append("  模式：参数仿真（K / B / C）")
        peak_k = prediction_data.get("temperature_peak_k")
        if peak_k is not None:
            lines.append(f"  火球最大温度 {_fmt(peak_k)} K")
    lines.append(
        f"  环境 T={_fmt(prediction_data.get('env_temp'))} °C，"
        f"RH={_fmt(prediction_data.get('env_humidity'))} %，"
        f"Pw,sat={_fmt(prediction_data.get('env_pressure'))} Pa"
    )
    lines.append("")

    lines.append("【火球直径】")
    i_dmax = int(np.argmax(d_m))
    d_max = float(d_m[i_dmax])
    at_max = np.isclose(d_m, d_max, rtol=1e-6, atol=1e-8)
    t_first_max = float(t_ms[int(np.flatnonzero(at_max)[0])])
    t_near, d_near = _time_to_fraction_of_max(t_ms, d_m, NEAR_MAX_DIAMETER_FRACTION)
    if kd is not None and t_first_max > float(t_ms[0]) + 1e-6:
        lines.append(
            f"  渐近最大直径 K = {_fmt(kd, 4)} m（来源：{src_text}；"
            f"序列自 t≥{_fmt(t_first_max, 4)} ms 数值上等于 K）"
        )
        lines.append(f"  B = {_fmt(b_val)}，C = {_fmt(c_val)}")
    else:
        lines.append(
            f"  序列最大直径 {_fmt(d_max, 4)} m @ t={_fmt(t_first_max, 4)} ms"
        )
        if kd is not None:
            lines.append(f"  拖曳渐近 K = {_fmt(kd, 4)} m（来源：{src_text}）")
            lines.append(f"  B = {_fmt(b_val)}，C = {_fmt(c_val)}")
    lines.append(
        f"  达到最大直径 {int(NEAR_MAX_DIAMETER_FRACTION * 100)}% 的时刻 "
        f"t={_fmt(t_near, 4)} ms（D={_fmt(d_near, 4)} m，与图上平台区一致）"
    )
    lines.append("")
    _append_temperature_heat_flux_lines(
        lines, temp_summary if isinstance(temp_summary, dict) else {}
    )
    lines.append("")

    lines.append("【膨胀至 90% 最大半径】")
    lines.append("  R_max = K/2，R(0)=0，t_90 满足 D(t)=0.9K")
    if kd is not None and b_val is not None and c_val is not None:
        vel = avg_radial_velocity_to_90pct_max_radius(kd, b_val, c_val)
        if vel is not None:
            t90, r0, r90, v_mm, v_s = vel
            lines.append(f"  R(0) = 0 m，0.9·R_max = {_fmt(r90, 4)} m")
            lines.append(f"  t_90% = {_fmt(t90, 4)} ms")
            lines.append(f"  平均径向速度 = {_fmt(v_mm, 4)} m/ms（{_fmt(v_s, 4)} m/s）")
        else:
            lines.append("  无法计算（需 B > 0.1 且 C > 0）")
    else:
        lines.append("  K, B, C 不可用")

    raw_rad = prediction_data.get("heat_radiation_data")
    rad = raw_rad if isinstance(raw_rad, dict) else {}
    h_raw = rad.get("heat_radiation")
    h_rad = np.asarray([] if h_raw is None else h_raw, dtype=np.float64)
    if h_rad.size:
        lines.append("")
        lines.append("【累积热辐射】")
        lines.append(f"  峰值 {_fmt(float(np.max(h_rad)), 4)} kJ/m²（随距离曲线）")

    return "\n".join(lines)
