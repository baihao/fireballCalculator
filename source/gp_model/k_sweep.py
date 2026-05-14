"""固定含铝量，沿当量扫掠预测 K、C（B 为训练产物 b_mean 常数）并出图。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from gp_model.train_infer import PredictiveResult

TASK_NAMES = ("K", "B", "C")


def build_X_star(
    equivalent_min: float,
    equivalent_max: float,
    num_points: int,
    al_percent: float,
) -> np.ndarray:
    """形状 (m, 2)：列 0 为当量 (kg)，列 1 为含铝量 (%)。"""
    eq = np.linspace(equivalent_min, equivalent_max, num_points, dtype=np.float64)
    al = np.full_like(eq, float(al_percent))
    return np.column_stack([eq, al])


def sweep_kbc_vs_equivalent(
    artifact: dict[str, Any],
    *,
    al_percent: float,
    equivalent_min: float,
    equivalent_max: float,
    num_points: int,
) -> tuple[np.ndarray, "PredictiveResult"]:
    """返回 (equivalent, pred)。"""
    from gp_model.train_infer import predict_mogp

    X_star = build_X_star(equivalent_min, equivalent_max, num_points, al_percent)
    pred = predict_mogp(artifact, X_star)
    eq = X_star[:, 0]
    return eq, pred


def plot_kbc_sweep(
    equivalent: np.ndarray,
    pred: "PredictiveResult",
    *,
    al_percent: float,
) -> plt.Figure:
    """同一张图内三行子图：K、B、C 的后验均值与潜函数 f 的 ±2σ。"""
    colors = ("#1f77b4", "#2ca02c", "#ff7f0e")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for t in range(3):
        ax = axes[t]
        m = pred.mean[:, t]
        sf = pred.std_latent[:, t]
        band = 2.0 * sf
        c = colors[t]
        ax.plot(equivalent, m, color=c, lw=1.8, label=f"{TASK_NAMES[t]} posterior mean")
        ax.fill_between(
            equivalent,
            m - band,
            m + band,
            color=c,
            alpha=0.22,
            label=f"{TASK_NAMES[t]} ± 2σ (latent f)",
        )
        ax.set_ylabel(TASK_NAMES[t])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Equivalent mass (kg)")
    fig.suptitle(
        f"Dual GP K, C vs equivalent (independent); B = b_mean (Al = {al_percent:g} %)",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def save_kbc_sweep_csv(
    path: Path | str,
    equivalent: np.ndarray,
    pred: "PredictiveResult",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["equivalent_kg"]
    for name in TASK_NAMES:
        header.extend(
            [
                f"{name}_mean",
                f"{name}_variance_y",
                f"{name}_std_y",
                f"{name}_variance_f",
                f"{name}_std_f",
            ]
        )
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(equivalent)):
            row = [f"{equivalent[i]:.6g}"]
            for t in range(3):
                row.extend(
                    [
                        f"{pred.mean[i, t]:.12g}",
                        f"{pred.variance[i, t]:.12g}",
                        f"{pred.std[i, t]:.12g}",
                        f"{pred.variance_latent[i, t]:.12g}",
                        f"{pred.std_latent[i, t]:.12g}",
                    ]
                )
            w.writerow(row)


def save_kbc_sweep_json(
    path: Path | str,
    *,
    al_percent: float,
    equivalent_min: float,
    equivalent_max: float,
    num_points: int,
    equivalent: np.ndarray,
    pred: "PredictiveResult",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(equivalent)):
        pt: dict[str, Any] = {"equivalent_kg": float(equivalent[i])}
        for t, name in enumerate(TASK_NAMES):
            pt[f"{name}_mean"] = float(pred.mean[i, t])
            pt[f"{name}_variance_y"] = float(pred.variance[i, t])
            pt[f"{name}_std_y"] = float(pred.std[i, t])
            pt[f"{name}_variance_f"] = float(pred.variance_latent[i, t])
            pt[f"{name}_std_f"] = float(pred.std_latent[i, t])
        rows.append(pt)
    payload = {
        "al_percent": al_percent,
        "equivalent_min": equivalent_min,
        "equivalent_max": equivalent_max,
        "num_points": num_points,
        "note": (
            "K,C: independent single-task GP predictive (*_y) and latent f (*_f). "
            "B is constant b_mean (variance columns 0); see gp_fireball_kc_lmc_strategy.md."
        ),
        "points": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
