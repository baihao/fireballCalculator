#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型训练 — 训练摘要面板文本（LOOCV 精度与主要超参）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .krr_workflow import KrrPredictGrid, KrrTrainingSummary, _ensure_import_paths

_TARGET_LABELS = {
    "K": "K（最大直径，m）",
    "B": "B（初始状态常数）",
    "C": "C（时间常数，ms⁻²）",
}

_RMSE_UNIT = {
    "K": "m",
    "B": "—",
    "C": "ms⁻²",
}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{digits}g}"
    except (TypeError, ValueError):
        return "—"


def _best_loocv_row(
    errors_by_target: Mapping[str, Mapping[str, Sequence[float]]],
    target: str,
) -> Optional[Dict[str, float]]:
    ed = errors_by_target.get(target)
    if not ed:
        return None
    sigmas = ed.get("sigma") or []
    tr_m = ed.get("train_mse") or []
    te_m = ed.get("test_mse") or []
    if not sigmas or len(sigmas) != len(tr_m) or len(sigmas) != len(te_m):
        return None
    j = int(np.argmin(np.asarray(te_m, dtype=np.float64)))
    _ensure_import_paths()
    from kernel_regression.train_kbc_kernel_ridge import sklearn_rbf_gamma_from_sigma

    sigma = float(sigmas[j])
    test_mse = float(te_m[j])
    train_mse = float(tr_m[j])
    return {
        "sigma": sigma,
        "gamma": float(sklearn_rbf_gamma_from_sigma(sigma)),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_rmse": float(np.sqrt(test_mse)) if test_mse >= 0 else float("nan"),
    }


def _lines_from_training_summary(
    summary: KrrTrainingSummary,
    *,
    n_samples: int,
    split_strategy: str,
    artifact_root: Optional[str] = None,
    data_folder: Optional[str] = None,
    grid: Optional[KrrPredictGrid] = None,
    status: str = "训练成功",
) -> str:
    lines: list[str] = []
    strat = "留一交叉验证（LOOCV）" if split_strategy == "loocv" else split_strategy
    lines.append("【训练状态】")
    lines.append(f"  状态：{status}")
    lines.append(f"  样本数 n={n_samples}｜划分：{strat}")
    if artifact_root:
        lines.append(f"  产物：{artifact_root}")
    if data_folder:
        lines.append(f"  数据目录：{data_folder}")
    lines.append("")

    ed0 = summary.errors_by_target.get("K") or {}
    n_sig = len(ed0.get("sigma") or [])

    lines.append("【模型参数】")
    lines.append("  算法：核岭回归 KernelRidge + RBF")
    lines.append(f"  kernel={summary.kernel}｜正则 α={summary.alpha:g}")
    lines.append(f"  σ 候选网格：{n_sig} 点（LOOCV 逐目标选优）")
    lines.append("  特征：X = (当量 kg TNT, 当量 × 含铝%)")
    lines.append("")

    lines.append("【LOOCV 精度（K / B / C）】")
    lines.append("  train MSE：每折训练子集拟合误差均值；test MSE：留出样本预测误差均值")
    for tgt in ("K", "B", "C"):
        row = _best_loocv_row(summary.errors_by_target, tgt)
        label = _TARGET_LABELS.get(tgt, tgt)
        if row is None:
            lines.append(f"  {label}：—")
            continue
        unit = _RMSE_UNIT.get(tgt, "")
        rmse_suffix = f" {unit}" if unit and unit != "—" else ""
        lines.append(f"  {label}")
        lines.append(
            f"    σ={_fmt(row['sigma'])}  γ={_fmt(row['gamma'])}"
        )
        lines.append(
            f"    train MSE={_fmt(row['train_mse'], 4)}  "
            f"test MSE={_fmt(row['test_mse'], 4)}  "
            f"test RMSE={_fmt(row['test_rmse'], 4)}{rmse_suffix}"
        )
    lines.append("")

    if grid is not None:
        eg = np.asarray(grid.equiv_grid, dtype=np.float64).ravel()
        al = np.asarray(grid.al_levels, dtype=np.float64).ravel()
        al_parts = ", ".join(_fmt(float(x), 4) for x in al)
        lines.append("【预测网格】")
        lines.append(
            f"  当量 ∈ [{_fmt(eg[0], 4)}, {_fmt(eg[-1], 4)}] kg TNT，{len(eg)} 点"
        )
        lines.append(
            f"  含铝 ∈ [{_fmt(al[0], 4)}, {_fmt(al[-1], 4)}] %，{len(al)} 档：{al_parts}"
        )

    return "\n".join(lines)


def training_summary_from_manifest(manifest_path: Path) -> Optional[KrrTrainingSummary]:
    """从 artefact ``manifest.json`` 重建 ``KrrTrainingSummary``（用于会话恢复）。"""
    try:
        with open(manifest_path, encoding="utf-8") as fp:
            mf = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None

    targets = mf.get("targets") or {}
    sigmas = mf.get("sigmas") or []
    errors_by_target: Dict[str, Dict[str, list[float]]] = {}

    for tgt in ("K", "B", "C"):
        info = targets.get(tgt) or {}
        if not info or not sigmas:
            continue
        best_sigma = info.get("best_sigma")
        if best_sigma is None:
            continue
        j = min(
            range(len(sigmas)),
            key=lambda i: abs(float(sigmas[i]) - float(best_sigma)),
        )
        train_mse = [float("nan")] * len(sigmas)
        test_mse = [float("nan")] * len(sigmas)
        train_mse[j] = float(info.get("best_loocv_train_mse", float("nan")))
        test_mse[j] = float(info.get("best_loocv_test_mse", float("nan")))
        errors_by_target[tgt] = {
            "sigma": [float(s) for s in sigmas],
            "train_mse": train_mse,
            "test_mse": test_mse,
        }

    if not errors_by_target:
        return None

    return KrrTrainingSummary(
        alpha=float(mf.get("alpha", 0.001)),
        kernel=str(mf.get("kernel", "rbf")),
        errors_by_target=errors_by_target,
    )


def build_training_summary_text(
    *,
    status: str = "未训练",
    n_samples: int = 0,
    split_strategy: str = "loocv",
    data_folder: Optional[str] = None,
    artifact_root: Optional[str] = None,
    train_summary: Optional[KrrTrainingSummary] = None,
    predict_grid: Optional[KrrPredictGrid] = None,
) -> str:
    """生成「训练摘要」面板全文。"""
    if train_summary is not None and status == "训练成功":
        return _lines_from_training_summary(
            train_summary,
            n_samples=n_samples,
            split_strategy=split_strategy,
            artifact_root=artifact_root,
            data_folder=data_folder,
            grid=predict_grid,
            status=status,
        )

    if artifact_root and train_summary is None:
        mf = Path(artifact_root).expanduser() / "manifest.json"
        if mf.is_file():
            recovered = training_summary_from_manifest(mf)
            if recovered is not None:
                return _lines_from_training_summary(
                    recovered,
                    n_samples=n_samples,
                    split_strategy=split_strategy,
                    artifact_root=artifact_root,
                    data_folder=data_folder,
                    grid=predict_grid,
                    status="训练成功（自 manifest 恢复）",
                )

    lines = ["【训练状态】", f"  状态：{status}"]
    if n_samples > 0:
        strat = "留一交叉验证（LOOCV）" if split_strategy == "loocv" else split_strategy
        lines.append(f"  样本数 n={n_samples}｜划分：{strat}")
    if data_folder:
        lines.append(f"  数据目录：{data_folder}")
    lines.append("")
    lines.append("【模型参数】")
    lines.append("  算法：核岭回归 KernelRidge + RBF（待训练）")
    lines.append("")
    lines.append("【LOOCV 精度】")
    lines.append("  完成训练后将显示 K / B / C 各目标的 train/test MSE 与 test RMSE。")
    return "\n".join(lines)
