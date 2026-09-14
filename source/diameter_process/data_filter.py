#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径数据过滤模块

1. 稳健离群剔除：滚动 median + k·MAD 与尾段基准上界（剔除 raw 尖峰）
2. 烟雾截断：峰值基准 median + k·MAD，峰值后相对下降截断
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

DEFAULT_MAD_K = 3.0
MIN_RETENTION_RATE = 0.05
MIN_POINTS_AFTER_CUTOFF = 50
MIN_CUTOFF_TIME_MS = 50.0
DEFAULT_WARMUP_MS = 50.0
EARLY_CLUSTER_HORIZON_MS = 120.0
TAIL_UPPER_MARGIN = 1.05


def _calculate_sliding_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if len(data) < window_size:
        return data.copy()
    half = window_size // 2
    smoothed = []
    for i in range(len(data)):
        lo = max(0, i - half)
        hi = min(len(data), i + half + 1)
        smoothed.append(np.mean(data[lo:hi]))
    return np.array(smoothed)


def _calculate_sliding_average_inlier_aware(
    data: np.ndarray, inlier: np.ndarray, window_size: int
) -> np.ndarray:
    """
    滑动平均仅对 inlier 样本求均值（离群点在窗口内不参与）。
    避免已剔除的尖峰仍通过邻域窗口污染早期平滑值。
    """
    n = len(data)
    if n == 0:
        return data.copy()
    half = max(1, window_size // 2)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        mask = inlier[lo:hi]
        if not np.any(mask):
            out[i] = float(data[i])
            continue
        out[i] = float(np.mean(data[lo:hi][mask]))
    return out


def _refine_early_inliers_by_cluster(
    t: np.ndarray,
    D: np.ndarray,
    inlier: np.ndarray,
    warmup_ms: float,
    horizon_ms: float = EARLY_CLUSTER_HORIZON_MS,
) -> np.ndarray:
    """
    暖机后早期若明显高于「暖机内 inlier 簇」，视为分割离群，避免抬高拖曳初始尺度。
    """
    early_inl = inlier & (t < warmup_ms)
    if int(np.sum(early_inl)) < 5:
        return inlier
    cap = float(np.percentile(D[early_inl], 90)) * 1.05
    spike = inlier & (t < horizon_ms) & (D > cap + 1e-9)
    return inlier & ~spike


def _early_phase_upper_bound(t: np.ndarray, D: np.ndarray, warmup_ms: float) -> Optional[float]:
    """
    暖机前火球应小于后段膨胀初期的稳健尺度；抑制「开头仍偏大的 inlier」。
    """
    ref = (t >= warmup_ms) & (t <= warmup_ms + 400.0)
    if int(np.sum(ref)) < 30:
        return None
    ref_d = D[ref]
    p25 = float(np.percentile(ref_d, 25))
    med = float(np.median(ref_d))
    return min(p25 * 1.4, med * 0.80)


def _calculate_rolling_median(data: np.ndarray, window_size: int) -> np.ndarray:
    n = len(data)
    if n == 0:
        return data.copy()
    half = max(1, window_size // 2)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(data[lo:hi]))
    return out


def _calculate_rolling_mad(
    data: np.ndarray, window_size: int, rolling_median: np.ndarray
) -> np.ndarray:
    n = len(data)
    if n == 0:
        return data.copy()
    half = max(1, window_size // 2)
    mad = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = data[lo:hi]
        mad[i] = float(np.median(np.abs(window - rolling_median[i])))
    return np.maximum(mad, 1e-6)


def _robust_tail_upper_bound(
    t: np.ndarray, D: np.ndarray, warmup_ms: float, mad_k: float
) -> Tuple[float, float, float]:
    """用暖机后尾段估计全局稳健上界（抑制开头 raw 尖峰）。"""
    mask = t >= warmup_ms
    if int(np.sum(mask)) < 10:
        mask = np.ones(len(t), dtype=bool)
    ref = D[mask]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    mad = max(mad, 0.05)
    upper = med + mad_k * mad
    return upper, med, mad


def compute_inlier_mask(
    time_data: List[float],
    diameter_data: List[float],
    window_size: int = 10,
    mad_k: float = DEFAULT_MAD_K,
    warmup_ms: float = DEFAULT_WARMUP_MS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    判定 inlier：raw 直径不得显著高于局部稳健上界，且不得超过尾段基准上界。
    """
    t = np.array(time_data, dtype=float)
    D = np.array(diameter_data, dtype=float)
    rolling_med = _calculate_rolling_median(D, window_size)
    rolling_mad = _calculate_rolling_mad(D, window_size, rolling_med)
    local_upper = rolling_med + mad_k * rolling_mad
    tail_upper, tail_med, tail_mad = _robust_tail_upper_bound(t, D, warmup_ms, mad_k)
    global_cap = tail_upper * TAIL_UPPER_MARGIN

    inlier = (D <= local_upper + 1e-9) & (D <= global_cap + 1e-9)
    early_cap = _early_phase_upper_bound(t, D, warmup_ms)
    if early_cap is not None:
        early_mask = t < warmup_ms
        inlier = inlier & (~early_mask | (D <= early_cap + 1e-9))
    inlier = _refine_early_inliers_by_cluster(t, D, inlier, warmup_ms)
    meta = {
        "tail_upper_bound_m": tail_upper,
        "tail_median_m": tail_med,
        "tail_mad_m": tail_mad,
        "global_cap_m": global_cap,
        "early_phase_cap_m": early_cap,
        "outliers_removed": int(np.sum(~inlier)),
    }
    return inlier, meta


def robust_diameter_ceiling(
    diameter_data: List[float],
    time_data: List[float],
    warmup_ms: float = DEFAULT_WARMUP_MS,
    mad_k: float = DEFAULT_MAD_K,
) -> float:
    """拟合 K 下界等用的稳健直径上界（非 raw max）。"""
    t = np.array(time_data, dtype=float)
    D = np.array(diameter_data, dtype=float)
    tail_upper, _, _ = _robust_tail_upper_bound(t, D, warmup_ms, mad_k)
    p98 = float(np.percentile(D, 98)) if len(D) else tail_upper
    return max(tail_upper * TAIL_UPPER_MARGIN, min(float(np.max(D)), p98))


def _robust_peak_index_and_baseline(
    t: np.ndarray,
    D: np.ndarray,
    window_size: int,
    mad_k: float,
    warmup_ms: float,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    rolling_med = _calculate_rolling_median(D, window_size)
    rolling_mad = _calculate_rolling_mad(D, window_size, rolling_med)
    upper = rolling_med + mad_k * rolling_mad

    eligible = t >= warmup_ms
    if not np.any(eligible):
        eligible = np.ones(len(t), dtype=bool)

    local_indices = np.where(eligible)[0]
    peak_idx = int(local_indices[np.argmax(rolling_med[eligible])])
    peak_baseline = float(max(upper[peak_idx], rolling_med[peak_idx]))
    peak_time = float(t[peak_idx])
    return peak_idx, peak_baseline, peak_time, rolling_med, rolling_mad


def filter_smoke_interference_data(
    time_data: List[float],
    diameter_data: List[float],
    drop_threshold: float = 0.02,
    window_size: int = 10,
    mad_k: float = DEFAULT_MAD_K,
    warmup_ms: float = DEFAULT_WARMUP_MS,
) -> Tuple[List[float], List[float]]:
    try:
        if len(time_data) != len(diameter_data):
            raise ValueError("时间和直径数据长度不匹配")

        t = np.array(time_data, dtype=float)
        D = np.array(diameter_data, dtype=float)
        smoothed_D_full = _calculate_sliding_average(D, window_size)

        if len(time_data) < window_size + 5:
            print("⚠️ 数据点太少，无法进行有效过滤")
            return [], smoothed_D_full.tolist()

        peak_idx, peak_baseline, peak_time, rolling_med, rolling_mad = (
            _robust_peak_index_and_baseline(t, D, window_size, mad_k, warmup_ms)
        )
        print(
            f"稳健峰值: 基准={peak_baseline:.2f}m (median+MAD, k={mad_k:g}) "
            f"@ {peak_time:.1f}ms [滚动median={rolling_med[peak_idx]:.2f}m, "
            f"MAD={rolling_mad[peak_idx]:.3f}m]"
        )

        if peak_idx >= len(t) - window_size:
            print("峰值在数据末尾附近，无需烟雾截断")
            return [], smoothed_D_full.tolist()

        t_after = t[peak_idx:]
        smoothed_D_after = smoothed_D_full[peak_idx:]

        cutoff_times: List[float] = []
        for i in range(len(smoothed_D_after)):
            if peak_baseline <= 0:
                break
            relative_drop = (peak_baseline - smoothed_D_after[i]) / peak_baseline
            if relative_drop > drop_threshold:
                cutoff_time = float(t_after[i])
                cutoff_times.append(cutoff_time)
                print(f"检测到烟雾干扰: 时间 {cutoff_time:.1f}ms, 相对峰值基准下降 {relative_drop:.1%}")
                break

        return cutoff_times, smoothed_D_full.tolist()

    except Exception as e:
        print(f"⚠️ 数据过滤失败: {e}")
        try:
            D = np.array(diameter_data, dtype=float)
            return [], _calculate_sliding_average(D, window_size).tolist()
        except Exception:
            return [], diameter_data


def _cutoff_passes_sanity(
    cutoff_time: float,
    n_kept: int,
    n_total: int,
    min_retention: float,
    min_points: int,
    min_cutoff_ms: float,
) -> bool:
    if n_total <= 0:
        return False
    retention = n_kept / n_total
    if retention < min_retention:
        print(f"⚠️ 截断自检: 保留率 {retention:.1%} < {min_retention:.1%}，放弃烟雾截断")
        return False
    if n_kept < min_points:
        print(f"⚠️ 截断自检: 保留点数 {n_kept} < {min_points}，放弃烟雾截断")
        return False
    if cutoff_time < min_cutoff_ms:
        print(f"⚠️ 截断自检: 截断时间 {cutoff_time:.1f}ms < {min_cutoff_ms:.1f}ms，放弃烟雾截断")
        return False
    return True


def apply_data_filter(
    time_data: List[float],
    diameter_data: List[float],
    drop_threshold: float = 0.02,
    window_size: int = 10,
    mad_k: float = DEFAULT_MAD_K,
    warmup_ms: float = DEFAULT_WARMUP_MS,
    min_retention_rate: float = MIN_RETENTION_RATE,
    min_points_after_cutoff: int = MIN_POINTS_AFTER_CUTOFF,
    min_cutoff_time_ms: float = MIN_CUTOFF_TIME_MS,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    返回拟合用 (time, 平滑直径) 及过滤统计。
    流程：离群剔除 → 烟雾截断（可选）→ 自检。
    """
    n_total = len(time_data)
    stats: Dict[str, Any] = {
        "outliers_removed": 0,
        "smoke_cutoff_applied": False,
        "cutoff_time": None,
        "inlier_mask": [],
        "tail_upper_bound_m": None,
    }

    if n_total == 0:
        return [], [], stats

    t = np.array(time_data, dtype=float)
    D = np.array(diameter_data, dtype=float)

    inlier, inlier_meta = compute_inlier_mask(
        time_data, diameter_data, window_size, mad_k, warmup_ms
    )
    stats.update(inlier_meta)
    stats["inlier_mask"] = inlier.tolist()

    n_out = int(np.sum(~inlier))
    if n_out > 0:
        extra = ""
        if inlier_meta.get("early_phase_cap_m") is not None:
            extra = f"，暖机前上限 {inlier_meta['early_phase_cap_m']:.2f}m"
        print(
            f"稳健离群剔除: 移除 {n_out}/{n_total} 点 "
            f"(尾段上界 {inlier_meta['global_cap_m']:.2f}m / 局部 median+{mad_k}·MAD{extra})"
        )

    smoothed_inlier_aware = _calculate_sliding_average_inlier_aware(D, inlier, window_size)

    t_inl = t[inlier]
    D_inl_raw = D[inlier]
    D_inl_smooth = smoothed_inlier_aware[inlier]

    if len(t_inl) < 4:
        print("⚠️ 离群剔除后数据过少，回退为全序列平滑")
        fallback = _calculate_sliding_average(D, window_size)
        return time_data, fallback.tolist(), stats

    cutoff_times, smoothed_after_outlier = filter_smoke_interference_data(
        t_inl.tolist(),
        D_inl_raw.tolist(),
        drop_threshold,
        window_size,
        mad_k=mad_k,
        warmup_ms=0.0,
    )
    # 使用与 inlier 对齐的平滑值（filter 内部重算 smooth，与 D_inl_smooth 一致长度）
    _ = smoothed_after_outlier
    fit_t = t_inl.tolist()
    fit_d = [
        float(raw if ti < warmup_ms else sm)
        for ti, raw, sm in zip(t_inl, D_inl_raw, D_inl_smooth)
    ]

    if not cutoff_times:
        print(f"拟合用数据: {len(fit_t)}/{n_total} 点（已剔除离群，无烟雾截断）")
        stats["data_retention_rate"] = len(fit_t) / n_total
        return fit_t, fit_d, stats

    cutoff_time = cutoff_times[0]
    n_kept = int(np.sum(t_inl <= cutoff_time))

    if not _cutoff_passes_sanity(
        cutoff_time,
        n_kept,
        len(t_inl),
        min_retention_rate,
        min_points_after_cutoff,
        min_cutoff_time_ms,
    ):
        print(f"拟合用数据: {len(fit_t)}/{n_total} 点（已剔除离群，烟雾截断未通过自检）")
        stats["data_retention_rate"] = len(fit_t) / n_total
        return fit_t, fit_d, stats

    stats["smoke_cutoff_applied"] = True
    stats["cutoff_time"] = cutoff_time
    fit_t = []
    fit_d = []
    for ti, di_raw, di_sm in zip(t_inl, D_inl_raw, D_inl_smooth):
        if ti <= cutoff_time:
            fit_t.append(float(ti))
            fit_d.append(float(di_raw if ti < warmup_ms else di_sm))
        else:
            break

    print(
        f"数据过滤: 拟合 {len(fit_t)}/{n_total} 点 "
        f"（离群剔除 + 烟雾截断 {cutoff_time:.1f}ms）"
    )
    stats["data_retention_rate"] = len(fit_t) / n_total
    return fit_t, fit_d, stats


def analyze_data_phases(time_data: List[float], diameter_data: List[float]) -> dict:
    try:
        t = np.array(time_data)
        D = np.array(diameter_data)
        max_idx = int(np.argmax(D))
        max_diameter = float(D[max_idx])
        max_time = float(t[max_idx])
        return {
            "total_duration": float(t[-1] - t[0]),
            "max_diameter": max_diameter,
            "max_time": max_time,
            "max_index": max_idx,
            "initial_diameter": float(D[0]),
            "final_diameter": float(D[-1]),
            "diameter_range": float(max_diameter - D[0]),
            "growth_phase": {
                "start_time": float(t[0]),
                "end_time": max_time,
                "duration": float(max_time - t[0]),
                "diameter_change": float(max_diameter - D[0]),
            },
            "post_max_phase": {
                "start_time": max_time,
                "end_time": float(t[-1]),
                "duration": float(t[-1] - max_time),
                "diameter_change": float(D[-1] - max_diameter),
            },
        }
    except Exception as e:
        print(f"⚠️ 阶段分析失败: {e}")
        return {}
