#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核岭回归（KRR）：在桌面进程的 ``sys.path`` 下调用 ``kernel_regression``，训练后在当量/含铝网格上预测 K/B/C。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..training_dataset_model import TrainingDatasetModel

_EQUIV_GRID_N = 100
_AL_GRID_N = 3


def _ensure_import_paths() -> None:
    """保证可导入 ``kernel_regression`` 与 ``training_tab``（与 ``run.py`` 一致）。"""
    desktop = Path(__file__).resolve().parent.parent.parent
    source = desktop.parent
    for p in (desktop, source):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def kr_model_directory(training_data_folder: str | Path) -> Path:
    """
    与训练数据目录同级、名为 ``kr_model`` 的父路径，供 ``train_kernel_regression_kbc`` 作为 ``model_path``：
    artefact 落在 ``kr_model/kernel_regression_<timestamp>/``。
    """
    d = Path(training_data_folder).expanduser().resolve()
    parent = d.parent
    out = parent / "kr_model"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _equiv_al_sampling_from_records(eq: np.ndarray, al: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """当量 [min,max] 上等间隔 100 点；含铝 [min,max] 上等间隔 3 点（退化时展开邻域）。"""
    emin, emax = float(np.min(eq)), float(np.max(eq))
    if not np.isfinite(emin) or not np.isfinite(emax):
        raise ValueError("当量范围无效")
    if emax <= emin:
        pad = 0.5 * (abs(emin) + 1.0)
        emin, emax = emin - pad, emax + pad
    equiv_grid = np.linspace(emin, emax, _EQUIV_GRID_N, dtype=np.float64)

    amin, amax = float(np.min(al)), float(np.max(al))
    if not np.isfinite(amin) or not np.isfinite(amax):
        raise ValueError("含铝量范围无效")
    if np.isclose(amin, amax):
        delta = 5.0
        amin = max(0.0, amin - delta)
        amax = min(100.0, amax + delta)
    al_levels = np.linspace(amin, amax, _AL_GRID_N, dtype=np.float64)
    return equiv_grid, al_levels


@dataclass(frozen=True)
class KrrPredictGrid:
    artifact_root: Path
    equiv_grid: np.ndarray
    al_levels: np.ndarray
    K: np.ndarray  # shape (n_al, n_eq)
    B: np.ndarray
    C: np.ndarray


@dataclass(frozen=True)
class KrrTrainingSummary:
    """与一次 ``train_kernel_regression_kbc`` 对齐，便于训练日志格式化。"""

    alpha: float
    kernel: str
    errors_by_target: Mapping[str, Mapping[str, list[float]]]


def krr_training_log_lines(summary: KrrTrainingSummary) -> list[str]:
    """供 ``QPlainTextEdit`` 逐行输出的模型训练摘要（LOOCV、RBF sigma / sklearn gamma）。"""
    _ensure_import_paths()
    from kernel_regression.train_kbc_kernel_ridge import sklearn_rbf_gamma_from_sigma

    ed0 = summary.errors_by_target.get("K")
    n_sig = len((ed0 or {}).get("sigma", []))

    lines: list[str] = [
        "[任务] 模型训练 — 核岭回归（KernelRidge + RBF）",
        f"  相关参数：kernel={summary.kernel}，正则 α（sklearn KernelRidge.alpha）={summary.alpha:g}",
        f"  RBF：核形式 k=exp(-‖Δx‖²/(2σ²))；sklearn 使用 gamma=1/(2σ²)。LOOCV 在约 {n_sig} 个 σ 候选上选优。",
        "  各目标最优 σ 处的 LOOCV 误差（train_MSE：每折在全训练子集上的拟合 MSE 再平均；test_MSE：留出样本的预测平方误差再平均）：",
    ]
    for tgt, label in (("K", "K（最大直径）"), ("B", "B（初始状态常数）"), ("C", "C（时间常数）")):
        ed = summary.errors_by_target.get(tgt)
        if not ed:
            continue
        sigmas = ed["sigma"]
        tr_m = ed["train_mse"]
        te_m = ed["test_mse"]
        if not sigmas:
            continue
        j = int(np.argmin(np.asarray(te_m, dtype=np.float64)))
        sigma = float(sigmas[j])
        gamma = sklearn_rbf_gamma_from_sigma(sigma)
        lines.append(
            f"    {label} — best σ={sigma:.6g}，sklearn γ={gamma:.6g}，"
            f"LOOCV train_MSE={float(tr_m[j]):.6g}，LOOCV test_MSE={float(te_m[j]):.6g}"
        )
    return lines


def krr_prediction_log_lines(grid: KrrPredictGrid) -> list[str]:
    """预测网格的范围与规模（对应训练数据中当量 / 含铝 RANGE 采样）。"""
    eg = np.asarray(grid.equiv_grid, dtype=np.float64).ravel()
    al = np.asarray(grid.al_levels, dtype=np.float64).ravel()
    eq_lo, eq_hi = float(eg[0]), float(eg[-1])
    al_lo, al_hi = float(al[0]), float(al[-1])
    al_parts = ", ".join(f"{float(x):.6g}" for x in al)
    return [
        "[任务] 预测 — 网格采样（用于 K/B/C 曲线叠加散点）",
        f"  预测采样（由训练集中当量的 min–max 等间隔）：当量 ∈ [{eq_lo:g}, {eq_hi:g}] kg TNT，点数={len(eg)}",
        f"  预测采样（由训练集中含铝量的 min–max 等间隔）：含铝量 ∈ [{al_lo:g}, {al_hi:g}] %，档数={len(al)}",
        f"  三档含铝量取值（%）：{al_parts}",
    ]


def run_train_and_predict(
    training_model: TrainingDatasetModel,
) -> tuple[Path, KrrPredictGrid, KrrTrainingSummary]:
    """
    调用 ``train_kernel_regression_kbc``，再在给定当量/含铝网格上逐点 ``predict_kernel_regression_kbc``。

    Returns:
        saved_root: ``kernel_regression_<timestamp>`` 目录。
        KrrPredictGrid: 预测矩阵（行 = 含铝档，列 = 当量采样点）。
        KrrTrainingSummary: LOOCV 曲线与选用的 α / 核类型，便于日志。
    """
    _ensure_import_paths()
    from kernel_regression.train_kbc_kernel_ridge import (
        DEFAULT_KERNEL_RIDGE_ALPHA,
        predict_kernel_regression_kbc,
        train_kernel_regression_kbc,
    )

    if training_model.data_folder is None or not training_model.records:
        raise ValueError("无训练数据")

    folder = training_model.data_folder
    kr_parent = kr_model_directory(folder)

    saved_root, errors_by_target = train_kernel_regression_kbc(training_model, kr_parent, alpha=None)
    summary = KrrTrainingSummary(
        alpha=float(DEFAULT_KERNEL_RIDGE_ALPHA),
        kernel="rbf",
        errors_by_target=errors_by_target,
    )

    eq = np.array([r.equivalent_kg_tnt for r in training_model.records], dtype=np.float64)
    al = np.array([r.al_percent for r in training_model.records], dtype=np.float64)
    equiv_grid, al_levels = _equiv_al_sampling_from_records(eq, al)

    n_al, n_eq = len(al_levels), len(equiv_grid)
    Kp = np.zeros((n_al, n_eq), dtype=np.float64)
    Bp = np.zeros((n_al, n_eq), dtype=np.float64)
    Cp = np.zeros((n_al, n_eq), dtype=np.float64)

    root = saved_root.resolve()
    for ia, alv in enumerate(al_levels):
        for je, ev in enumerate(equiv_grid):
            k, b, c = predict_kernel_regression_kbc(root, float(ev), float(alv))
            Kp[ia, je] = k
            Bp[ia, je] = b
            Cp[ia, je] = c

    grid = KrrPredictGrid(
        artifact_root=root,
        equiv_grid=equiv_grid,
        al_levels=al_levels,
        K=Kp,
        B=Bp,
        C=Cp,
    )
    return saved_root, grid, summary
