#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定 RBF 长度尺度 \(\sigma\) 与 \(\alpha\) 训练 Kernel Ridge（K/B/C），可选生成 \(K\) 拟合诊断图：
训练点「当量–\(K\)」散点 + 固定含铝下当量网格上的预测 \(K\) 曲线。

核采用 \(k=\exp(-\|x{-}x'\|^2/(2\sigma^2))\)；``sklearn`` 的 ``gamma`` 为 \(1/(2\sigma^2)\)。

建议在仓库 ``source`` 目录执行::

    python kernel_regression/run_test_with_gamma.py \\
      --data-dir /path/to/training_json_dir \\
      --out-dir ./krr_outputs \\
      --sigma 10 \\
      --graph \\
      --alpha 1e-2

（``--gamma`` 与 ``--sigma`` 同义，便于旧命令行。）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_python_path() -> Path:
    here = Path(__file__).resolve()
    src = here.parent.parent
    desktop = src / "desktop"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(desktop))
    return src


_ensure_python_path()

from kernel_regression.graph import _resolve_png_out  # noqa: E402
from kernel_regression.train_kbc_kernel_ridge import (  # noqa: E402
    DEFAULT_KERNEL_RIDGE_ALPHA,
    FEATURE_DESC,
    MODEL_ARTIFACT_FILENAMES,
    build_X,
    fit_full_model,
    sklearn_rbf_gamma_from_sigma,
    _ensure_timestamp_root,
    _save_model_bundle,
)
from training_tab.training_dataset_model import TrainingDatasetModel  # noqa: E402
from training_tab.utils.dataset_io import import_training_folder  # noqa: E402

DEFAULT_K_FIT_PLOT = "kbc_K_gamma_fit_scatter_and_curve.png"


def _train_mse_in_sample(model, X: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(X)
    return float(np.mean((y.ravel() - pred.ravel()) ** 2))


def plot_k_gamma_fit_diagnostic(
    equiv_train: np.ndarray,
    k_train: np.ndarray,
    equiv_sweep: np.ndarray,
    k_pred_sweep: np.ndarray,
    *,
    sigma: float,
    alpha: float,
    al_percent_fixed: float,
    output_path: Path,
) -> Path:
    matplotlib.rcParams["axes.unicode_minus"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.scatter(
        equiv_train,
        k_train,
        color="#f97316",
        s=36,
        zorder=3,
        label="训练样本（每条记录的当量与 K）",
    )
    ax.plot(
        equiv_sweep,
        k_pred_sweep,
        color="#38bdf8",
        linewidth=2.0,
        label=f"预测 K（含铝固定 {al_percent_fixed:g}%）",
    )
    ax.set_xlabel("当量 (kg TNT)")
    ax.set_ylabel("K")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    g_sk = sklearn_rbf_gamma_from_sigma(sigma)
    ax.set_title(
        f"RBF Kernel Ridge：σ={sigma:g}，γ_sklearn=1/(2σ²)={g_sk:.4g}，α={alpha:g} "
        "— 训练散点 vs 当量扫描预测",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="指定 σ（RBF 长度尺度）、α 训练 K/B/C（不写 LOOCV），可选生成 K 拟合诊断图"
    )
    ap.add_argument("--data-dir", type=str, required=True, help="训练 JSON 目录（同 run.py dataset_io）")
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="artefact 父目录；将创建 kernel_regression_<timestamp> 并写入三套 joblib",
    )
    ap.add_argument(
        "--sigma",
        "--gamma",
        dest="sigma",
        type=float,
        required=True,
        help="RBF 长度尺度 σ（--gamma 同义）；sklearn 内部 γ=1/(2σ²)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=f"KernelRidge 正则；默认 {DEFAULT_KERNEL_RIDGE_ALPHA:g}（同 train_kernel_regression_kbc 默认）",
    )
    ap.add_argument(
        "--graph",
        action="store_true",
        help="生成 PNG：训练当量-K 散点 + 扫描当量预测 K 曲线",
    )
    ap.add_argument(
        "--equiv-min",
        type=int,
        default=1,
        help="预测曲线起始当量（含端点）",
    )
    ap.add_argument(
        "--equiv-max",
        type=int,
        default=150,
        help="预测曲线结束当量（含端点）",
    )
    ap.add_argument(
        "--al-percent",
        type=float,
        default=40.0,
        help="预测曲线所用的固定含铝量（%%），默认 40",
    )
    ap.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help=f"PNG 路径；默认 artefact 内 {DEFAULT_K_FIT_PLOT}（相对路径相对 artefact）",
    )
    args = ap.parse_args()

    alpha_v = float(DEFAULT_KERNEL_RIDGE_ALPHA if args.alpha is None else args.alpha)
    sigma_v = float(args.sigma)
    gamma_sk_v = sklearn_rbf_gamma_from_sigma(sigma_v)

    res = import_training_folder(
        Path(args.data_dir), recursive=True, strict_drag_fit_success=False
    )
    if not res.ok or not res.records:
        sys.stderr.write(f"训练失败: {res.error_message or '无有效记录'}\n")
        sys.exit(1)

    tm = TrainingDatasetModel()
    tm.set_loaded_training_folder(res.folder_resolved, res.records)
    records = tm.records

    eq = np.array([r.equivalent_kg_tnt for r in records], dtype=np.float64)
    al = np.array([r.al_percent for r in records], dtype=np.float64)
    K = np.array([r.K for r in records], dtype=np.float64)
    B = np.array([r.B for r in records], dtype=np.float64)
    C = np.array([r.C for r in records], dtype=np.float64)
    X = build_X(eq, al)

    if len(records) < 2:
        sys.stderr.write("训练至少需要 2 条样本。\n")
        sys.exit(2)

    sys.stderr.write(
        "[train] sigma=%.12g sklearn_rbf_gamma=%.12g alpha=%.12g\n"
        % (sigma_v, gamma_sk_v, alpha_v)
    )
    sys.stderr.write(
        "[train] X shape=%s cols=(equiv_kg_TNT, equiv*(al_pct/100))\n%s\n"
        % (X.shape, np.array2string(X, precision=6))
    )
    for _name, _y in ("K", K), ("B", B), ("C", C):
        sys.stderr.write(
            "[train] y_%s shape=%s\n%s\n"
            % (_name, _y.shape, np.array2string(_y, precision=6))
        )

    saved_root, ts_dir = _ensure_timestamp_root(Path(args.out_dir))
    filenames = dict(zip(("K", "B", "C"), MODEL_ARTIFACT_FILENAMES))
    targets_y = {"K": K, "B": B, "C": C}

    manifest: dict = {
        "timestamp_dir": ts_dir,
        "training_mode": "fixed_sigma",
        "sigma_fixed": sigma_v,
        "sklearn_rbf_gamma": gamma_sk_v,
        "alpha": alpha_v,
        "rbf_parameterization": "k=exp(-||dx||^2/(2*sigma^2)); sklearn gamma=1/(2*sigma^2)",
        "n_samples": len(records),
        "data_folder": tm.data_folder,
        "kernel": "rbf",
        "feature_desc": FEATURE_DESC,
        "targets": {},
    }

    model_k = None
    for name in ("K", "B", "C"):
        ys = targets_y[name]
        model = fit_full_model(X, ys, sigma_v, alpha=alpha_v)
        if name == "K":
            model_k = model
        mse_train = _train_mse_in_sample(model, X, ys)
        path_b = saved_root / filenames[name]
        _save_model_bundle(
            path_b,
            model,
            target=name,
            best_sigma=sigma_v,
            alpha=alpha_v,
            feature_desc=FEATURE_DESC,
        )
        manifest["targets"][name] = {
            "sigma": sigma_v,
            "sklearn_rbf_gamma": gamma_sk_v,
            "alpha": alpha_v,
            "train_mse_in_sample": mse_train,
            "model_file": filenames[name],
        }

    with open(saved_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(str(saved_root.resolve()))

    if args.graph:
        al_fix = float(args.al_percent)
        equiv_sweep = np.arange(int(args.equiv_min), int(args.equiv_max) + 1, dtype=np.float64)
        al_sweep = np.full_like(equiv_sweep, al_fix)
        X_sweep = build_X(equiv_sweep, al_sweep)
        k_line = model_k.predict(X_sweep)

        outp = _resolve_png_out(saved_root, args.plot_out, DEFAULT_K_FIT_PLOT)
        png = plot_k_gamma_fit_diagnostic(
            eq,
            K,
            equiv_sweep,
            k_line,
            sigma=sigma_v,
            alpha=alpha_v,
            al_percent_fixed=al_fix,
            output_path=outp,
        )
        sys.stderr.write(f"[graph] {png}\n")


if __name__ == "__main__":
    main()
