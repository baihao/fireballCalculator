#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机器视觉 — 分割与拟合关键参数文本（右侧面板）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

NEAR_MAX_FRACTION = 0.95
RADIUS_90_MAX_FRACTION = 0.90


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


def _interp_time(
    t0: float, t1: float, r0: float, r1: float, r_target: float
) -> float:
    if abs(r1 - r0) < 1e-12:
        return t1
    frac = (r_target - r0) / (r1 - r0)
    return t0 + frac * (t1 - t0)


def _avg_velocity_to_90pct_max_radius_m_s(
    series: Sequence[Tuple[float, float]],
) -> Optional[float]:
    """
    从首有效点半径至达到 0.9·R_max 的平均膨胀速度 (m/s)。
    series 为 (时间 ms, 直径 m)。
    """
    if len(series) < 2:
        return None
    times = np.asarray([t for t, _ in series], dtype=np.float64)
    radii = 0.5 * np.asarray([d for _, d in series], dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(radii)):
        return None
    r_max = float(np.max(radii))
    if r_max <= 0:
        return None
    r_target = RADIUS_90_MAX_FRACTION * r_max
    r_start = float(radii[0])
    t_start = float(times[0])
    if r_start >= r_target:
        return 0.0
    hit = np.flatnonzero(radii >= r_target)
    if hit.size == 0:
        return None
    j = int(hit[0])
    if j == 0:
        t_target = float(times[0])
    else:
        t_target = _interp_time(
            float(times[j - 1]),
            float(times[j]),
            float(radii[j - 1]),
            float(radii[j]),
            r_target,
        )
    dt_ms = t_target - t_start
    if dt_ms <= 0:
        return None
    return (r_target - r_start) / (dt_ms * 1e-3)


def _max_diameter_m(series: Sequence[Tuple[float, float]]) -> Optional[float]:
    if not series:
        return None
    diams = np.asarray([d for _, d in series], dtype=np.float64)
    if diams.size == 0 or not np.any(np.isfinite(diams)):
        return None
    return float(np.nanmax(diams))


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        if v <= 1.0:
            return f"{v * 100:.1f} %"
        return f"{v:.1f} %"
    except (TypeError, ValueError):
        return "—"


def build_key_metrics_text(
    *,
    parameters: Optional[Dict[str, Any]] = None,
    pixel_length: Optional[float] = None,
    explosion_duration_ms: Optional[float] = None,
    image_count: int = 0,
    seg_summary: Optional[Dict[str, Any]] = None,
    current_frame_index: int = 0,
    current_frame_result: Optional[Dict[str, Any]] = None,
    diameter_series: Optional[Sequence[Tuple[float, float]]] = None,
    drag_fit: Optional[Dict[str, Any]] = None,
    segmentation_avg_ms_per_image: Optional[float] = None,
) -> str:
    """生成分割 / 直径 / 拖曳拟合关键参数说明。"""
    params = parameters or {}
    lines: List[str] = []

    lines.append("【输入与标定】")
    lines.append(f"  当量 {_fmt(params.get('equivalent'))} kg TNT｜含铝 {_fmt(params.get('al_percent'))} %")
    lines.append(
        f"  爆炸时长 {_fmt(explosion_duration_ms or params.get('explosion_duration'))} ms｜"
        f"帧率 {_fmt(params.get('frame_rate_fps'))} fps"
    )
    lines.append(
        f"  视场 {_fmt(params.get('field_of_view_m'))} m｜"
        f"像素标定 {_fmt(pixel_length or params.get('pixel_length'))} m/px"
    )
    lines.append(f"  序列帧数 {image_count if image_count else '—'}")
    lines.append("")

    series = list(diameter_series or [])
    v90 = _avg_velocity_to_90pct_max_radius_m_s(series) if series else None
    d_max = _max_diameter_m(series) if series else None

    lines.append("【关键指标】")
    if segmentation_avg_ms_per_image is not None and segmentation_avg_ms_per_image >= 0:
        lines.append(
            f"  分割每张图片的平均时间 {_fmt(segmentation_avg_ms_per_image, 4)} ms"
        )
    else:
        lines.append("  分割每张图片的平均时间 —")
    if v90 is not None:
        lines.append(
            f"  火球膨胀到90%最大半径的平均速度 {_fmt(v90, 4)} m/s"
        )
    else:
        lines.append("  火球膨胀到90%最大半径的平均速度 —")
    if d_max is not None:
        lines.append(f"  火球最大直径 {_fmt(d_max, 4)} m")
    else:
        lines.append("  火球最大直径 —")
    lines.append("")

    lines.append("【分割质量】")
    if seg_summary and seg_summary.get("total", 0) > 0:
        total = int(seg_summary["total"])
        success = int(seg_summary.get("success", 0))
        failed = max(total - success, 0)
        rate = (100.0 * success / total) if total > 0 else 0.0
        lines.append(f"  成功 {success} / 总 {total} 帧（失败 {failed}，成功率 {rate:.1f} %）")
    else:
        lines.append("  尚未完成分割")

    frame_no = current_frame_index + 1 if image_count > 0 else current_frame_index + 1
    if current_frame_result and current_frame_result.get("success"):
        mr = (current_frame_result.get("max_radius") or {})
        cen = current_frame_result.get("centroid") or {}
        lines.append(
            f"  当前帧 #{frame_no}：最大半径 {_fmt(mr.get('value'), 4)} px"
        )
        if cen.get("x") is not None and cen.get("y") is not None:
            lines.append(
                f"  质心 ({_fmt(cen.get('x'), 4)}, {_fmt(cen.get('y'), 4)}) px"
            )
    elif seg_summary and seg_summary.get("total", 0) > 0:
        lines.append(f"  当前帧 #{frame_no}：分割未成功或无数据")
    lines.append("")

    lines.append("【直径实测】")
    if series:
        times = np.asarray([t for t, _ in series], dtype=np.float64)
        diams = np.asarray([d for _, d in series], dtype=np.float64)
        i_max = int(np.argmax(diams))
        lines.append(f"  有效点数 {len(series)}")
        lines.append(
            f"  最大直径 {_fmt(diams[i_max], 4)} m @ t={_fmt(times[i_max], 4)} ms"
        )
        lines.append(
            f"  初始 {_fmt(diams[0], 4)} m → 末时刻 {_fmt(diams[-1], 4)} m"
        )
        if times.size >= 2:
            vel = np.gradient(diams, times)
            i_v = int(np.argmax(vel))
            lines.append(
                f"  峰值膨胀速率 {_fmt(vel[i_v], 4)} m/ms @ t={_fmt(times[i_v], 4)} ms"
            )
            target = NEAR_MAX_FRACTION * float(np.max(vel))
            hit = np.flatnonzero(vel >= target)
            if hit.size:
                j = int(hit[0])
                lines.append(
                    f"  达峰值速率 {int(NEAR_MAX_FRACTION * 100)}%："
                    f"t={_fmt(times[j], 4)} ms（v={_fmt(vel[j], 4)} m/ms）"
                )
    else:
        lines.append("  尚无直径曲线（需完成分割）")
    lines.append("")

    lines.append("【拖曳拟合 D(t)=K(1−B·e^(−C·t²))】")
    if drag_fit and drag_fit.get("K") is not None:
        df = drag_fit.get("data_filtering") or {}
        lines.append(f"  K = {_fmt(drag_fit.get('K'))} m")
        lines.append(f"  B = {_fmt(drag_fit.get('B'))}")
        lines.append(f"  C = {_fmt(drag_fit.get('C'))} ms⁻²")
        if drag_fit.get("r_squared") is not None:
            lines.append(
                f"  R² = {_fmt(drag_fit.get('r_squared'), 4)}｜"
                f"RMSE = {_fmt(drag_fit.get('rmse'), 4)} m｜"
                f"MAE = {_fmt(drag_fit.get('mae'), 4)} m"
            )
        if drag_fit.get("max_relative_error") is not None:
            lines.append(f"  最大相对误差 {_fmt(drag_fit.get('max_relative_error'), 4)} %")
        cutoff = df.get("cutoff_time")
        if cutoff is not None:
            lines.append(f"  截断时间 {_fmt(cutoff, 4)} ms")
        if df.get("original_data_points") is not None:
            lines.append(
                f"  过滤前/后点数 {df.get('original_data_points')} → {df.get('filtered_data_points')}"
                f"（保留率 {_pct(df.get('data_retention_rate'))}）"
            )
        tr = df.get("filtered_time_range")
        if tr and len(tr) >= 2:
            lines.append(f"  有效时间范围 [{_fmt(tr[0], 4)}, {_fmt(tr[1], 4)}] ms")
    elif series:
        lines.append("  拟合未成功或参数不可用")
    else:
        lines.append("  待分割完成后拟合")

    return "\n".join(lines)
