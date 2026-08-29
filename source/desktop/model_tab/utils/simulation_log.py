#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工程计算仿真结果摘要，供标签页「仿真日志」展示。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

NEAR_MAX_DIAMETER_FRACTION = 0.95


def _fmt_g(x: float, digits: int = 6) -> str:
    return f"{float(x):.{digits}g}"


def _time_to_fraction_of_max(
    t_ms: np.ndarray, values: np.ndarray, fraction: float
) -> tuple[float, float]:
    """返回达到 ``fraction × max(values)`` 的最早时刻及对应数值。"""
    t = np.asarray(t_ms, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0
    vmax = float(np.max(v))
    if vmax <= 0:
        return float(t[0]), float(v[0])
    target = fraction * vmax
    hit = np.flatnonzero(v >= target)
    if hit.size == 0:
        return float(t[-1]), float(v[-1])
    i = int(hit[0])
    return float(t[i]), float(v[i])


def _value_at_nearest_distance(
    distances: np.ndarray, values: np.ndarray, x_m: float
) -> tuple[float, float]:
    d = np.asarray(distances, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if d.size == 0:
        return x_m, 0.0
    i = int(np.argmin(np.abs(d - x_m)))
    return float(d[i]), float(v[i])


def build_simulation_log_lines(
    prediction_data: Dict[str, Any],
    *,
    al_percent: float,
    kbc_source_label: str = "",
) -> List[str]:
    """
    从 ``prediction_data``（``build_prediction_bundle`` 输出）生成仿真关键信息文本行。
    """
    t_ms = np.asarray(prediction_data["time_ms"], dtype=np.float64)
    d_m = np.asarray(prediction_data["diameter_data"], dtype=np.float64)
    t_k = np.asarray(prediction_data["temperature_data"], dtype=np.float64)

    i_dmax = int(np.argmax(d_m))
    d_max = float(d_m[i_dmax])
    at_max = np.isclose(d_m, d_max, rtol=1e-6, atol=1e-8)
    t_first_max = float(t_ms[int(np.flatnonzero(at_max)[0])])
    t_last_max = float(t_ms[int(np.flatnonzero(at_max)[-1])])
    t_near, d_near = _time_to_fraction_of_max(t_ms, d_m, NEAR_MAX_DIAMETER_FRACTION)

    i_tmax = int(np.argmax(t_k))
    t_peak = float(t_ms[i_tmax])
    T_peak = float(t_k[i_tmax])
    T_end = float(t_k[-1])

    kbc = prediction_data.get("kbc_display") or (0.0, 0.0, 0.0)
    src = kbc_source_label or str(prediction_data.get("kbc_source", ""))
    src_map = {
        "krr": "核岭回归预测 K/B/C",
        "calculator": "计算器当量缩放",
        "explicit_kbc": "显式 K/B/C 拖曳式",
        "calculator_scaled": "计算器当量缩放",
    }
    src_text = src_map.get(src, src or "—")

    lines: List[str] = []
    lines.append("【仿真参数】")
    if prediction_data.get("simulation_mode") == "parameter":
        lines.append(
            f"  模式：参数仿真（拖曳曲线 K/B/C）｜"
            f"时长 { _fmt_g(prediction_data.get('duration', t_ms[-1])) } ms｜"
            f"步长采样 {t_ms.size} 点"
        )
    else:
        lines.append(
            f"  当量 { _fmt_g(prediction_data['equivalent']) } kg TNT｜"
            f"含铝 { _fmt_g(al_percent) } %｜"
            f"材料 {prediction_data.get('material_name', '—')}"
        )
        lines.append(
            f"  时长 { _fmt_g(prediction_data.get('duration', t_ms[-1])) } ms｜"
            f"步长采样 {t_ms.size} 点｜"
            f"当量比 M={ _fmt_g(prediction_data.get('equivalent_ratio', 1.0)) }"
        )
    lines.append(
        f"  环境：T={ _fmt_g(prediction_data.get('env_temp', 0)) } °C，"
        f"RH={ _fmt_g(prediction_data.get('env_humidity', 0)) } %，"
        f"Pw,sat={ _fmt_g(prediction_data.get('env_pressure', 0)) } Pa"
    )
    lines.append(f"  直径模型：{src_text}")
    lines.append(
        f"  K={ _fmt_g(kbc[0]) } m，B={ _fmt_g(kbc[1]) }，C={ _fmt_g(kbc[2]) }"
    )

    lines.append("")
    lines.append("【火球直径】")
    if t_last_max > t_first_max + 1e-6:
        lines.append(
            f"  最大直径 { _fmt_g(d_max, 4) } m（t≥{ _fmt_g(t_first_max, 4) } ms 进入稳态，"
            f"仿真末 t={ _fmt_g(float(t_ms[-1]), 4) } ms）"
        )
    else:
        lines.append(
            f"  最大直径 { _fmt_g(d_max, 4) } m（t={ _fmt_g(t_first_max, 4) } ms）"
        )
    lines.append(
        f"  达到最大直径 {int(NEAR_MAX_DIAMETER_FRACTION * 100)}% 的时刻 "
        f"t={ _fmt_g(t_near, 4) } ms（D={ _fmt_g(d_near, 4) } m）"
    )
    if t_ms.size >= 2:
        lines.append(
            f"  初始直径 { _fmt_g(d_m[0], 4) } m → 末时刻 { _fmt_g(d_m[-1], 4) } m"
        )

    lines.append("")
    lines.append("【火球温度】")
    lines.append(f"  峰值温度 { _fmt_g(T_peak, 4) } K（t={ _fmt_g(t_peak, 4) } ms）")
    lines.append(f"  末时刻温度 { _fmt_g(T_end, 4) } K")

    heat_store: Dict[str, np.ndarray] = prediction_data.get("heat_flux_data") or {}
    if heat_store:
        lines.append("")
        lines.append("【热通量】（W/m²，各距离峰值及出现时刻）")
        dist_keys = sorted(heat_store.keys(), key=lambda s: float(s))
        for key in dist_keys:
            q = np.asarray(heat_store[key], dtype=np.float64)
            i_q = int(np.argmax(q))
            lines.append(
                f"  x={key} m：峰值 { _fmt_g(q[i_q], 4) } @ t={ _fmt_g(t_ms[i_q], 4) } ms"
            )
        # 代表距离：最近与最远
        q_first = np.asarray(heat_store[dist_keys[0]], dtype=np.float64)
        q_last = np.asarray(heat_store[dist_keys[-1]], dtype=np.float64)
        lines.append(
            f"  对比：近端 x={dist_keys[0]} m 峰值 { _fmt_g(float(np.max(q_first)), 4) }，"
            f"远端 x={dist_keys[-1]} m 峰值 { _fmt_g(float(np.max(q_last)), 4) }"
        )

    rad = prediction_data.get("heat_radiation_data") or {}
    x_raw = rad.get("distances")
    h_raw = rad.get("heat_radiation")
    x_rad = np.asarray(x_raw if x_raw is not None else [], dtype=np.float64)
    h_rad = np.asarray(h_raw if h_raw is not None else [], dtype=np.float64)
    if x_rad.size and h_rad.size:
        lines.append("")
        lines.append("【累积热辐射】（kJ/m²，随距离变化）")
        i_hmax = int(np.argmax(h_rad))
        lines.append(
            f"  全距离段峰值 { _fmt_g(h_rad[i_hmax], 4) } kJ/m² @ x={ _fmt_g(x_rad[i_hmax], 4) } m"
        )
        for x_ref in (6.0, 8.0, 10.0):
            if x_rad.min() <= x_ref <= x_rad.max():
                x_act, h_act = _value_at_nearest_distance(x_rad, h_rad, x_ref)
                lines.append(
                    f"  x≈{ _fmt_g(x_ref, 4) } m：{ _fmt_g(h_act, 4) } kJ/m²"
                    f"（采样点 x={ _fmt_g(x_act, 4) } m）"
                )
        lines.append(
            f"  距离范围 [{ _fmt_g(x_rad[0], 4) }, { _fmt_g(x_rad[-1], 4) }] m，"
            f"共 {x_rad.size} 个采样点"
        )

    return lines
