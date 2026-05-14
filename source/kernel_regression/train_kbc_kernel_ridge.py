#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
供 ``training_tab`` / 包内调用：Kernel Ridge（RBF）拟合 K、B、C，LOOCV 选 gamma。

不包含命令行与训练目录加载；CLI 见 ``run.py``。数学约定见 ``KRR_KBC_METHOD.md``。

独立脚本中与 ``training_tab`` 共用且 ``PYTHONPATH`` 未就绪时：**先**
``import kernel_regression.train_kbc_kernel_ridge``（会向 ``sys.path`` 前置 ``source/desktop``、``source``）
再导入 ``training_tab``；或自行设置 ``PYTHONPATH=<repo>/source:<repo>/source/desktop``。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# training_tab / gp_model 位于 ``source/desktop``、``source``；保证非 GUI 环境下可导入
_PKG_ROOT = Path(__file__).resolve().parent.parent  # …/source
_DESKTOP_PKG = _PKG_ROOT / "desktop"
for _p in (_DESKTOP_PKG, _PKG_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from typing import Any, Mapping, TypedDict

import numpy as np

try:
    from joblib import dump, load
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import LeaveOneOut
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "需要安装 scikit-learn 与 joblib（见 requirements.txt）"
    ) from e

from training_tab.training_dataset_model import TrainingDatasetModel

# ---------------------------------------------------------------------------

FEATURE_DESC = "X = (equiv_kg_TNT, equiv * (al_percent/100) [= m_Al kg])"

MODEL_ARTIFACT_FILENAMES = ("kbc_krr_K.joblib", "kbc_krr_B.joblib", "kbc_krr_C.joblib")

DEFAULT_KERNEL_RIDGE_ALPHA = 1e-3


class TargetGammaErrors(TypedDict):
    gamma: list[float]
    train_mse: list[float]
    test_mse: list[float]


ErrorsByTarget = Mapping[str, TargetGammaErrors]


def build_X(equiv: np.ndarray, al_pct: np.ndarray) -> np.ndarray:
    """
    ``al_pct`` 为含铝量百分比（如 30 表示 30%）。第二维：
    ``equiv * (al_pct / 100)``，在当量为装药总质量 (kg) 的约定下等价于其中 **铝粉质量 (kg)**。
    """
    eq = np.asarray(equiv, dtype=np.float64).ravel()
    al = np.asarray(al_pct, dtype=np.float64).ravel()
    if eq.size != al.size:
        raise ValueError("当量与含铝量长度不一致")
    al_frac = al / 100.0
    return np.column_stack([eq, eq * al_frac])


def gamma_grid_from_equiv(equiv: np.ndarray, n_steps: int = 30) -> np.ndarray:
    eq = np.asarray(equiv, dtype=np.float64).ravel()
    mn, mx = float(np.min(eq)), float(np.max(eq))
    span = mx - mn
    stride = span / 30.0 if span > 1e-12 else 1.0
    return np.array([1.0 + i * stride for i in range(n_steps + 1)], dtype=np.float64)


def stride_equiv_from_array(equiv: np.ndarray) -> float:
    eq = np.asarray(equiv, dtype=np.float64).ravel()
    mn, mx = float(np.min(eq)), float(np.max(eq))
    span = mx - mn
    return float(span / 30.0) if span > 1e-12 else 1.0


def loocv_train_test_mse(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float,
    *,
    alpha: float,
) -> tuple[float, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    if n < 2:
        raise ValueError("LOOCV 至少需要 2 条样本")

    loo = LeaveOneOut()
    train_mses: list[float] = []
    test_sq_errors: list[float] = []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        model.fit(X_tr, y_tr)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        train_mses.append(float(np.mean((y_tr - pred_tr) ** 2)))
        test_sq_errors.append(float((y_te[0] - pred_te[0]) ** 2))

    return float(np.mean(train_mses)), float(np.mean(test_sq_errors))


def sweep_gamma_loocv(
    X: np.ndarray,
    y: np.ndarray,
    gammas: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    tr_list: list[float] = []
    te_list: list[float] = []
    for g in gammas:
        tr, te = loocv_train_test_mse(X, y, float(g), alpha=alpha)
        tr_list.append(tr)
        te_list.append(te)
    tr_arr = np.asarray(tr_list, dtype=np.float64)
    te_arr = np.asarray(te_list, dtype=np.float64)
    j = int(np.argmin(te_arr))
    return gammas, tr_arr, te_arr, float(gammas[j])


def fit_full_model(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float,
    *,
    alpha: float,
) -> KernelRidge:
    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    model.fit(X, y)
    return model


def _save_model_bundle(
    path: Path,
    model: KernelRidge,
    *,
    target: str,
    best_gamma: float,
    alpha: float,
    feature_desc: str,
) -> None:
    bundle: dict[str, Any] = {
        "target": target,
        "best_gamma": best_gamma,
        "alpha": alpha,
        "kernel": "rbf",
        "feature_desc": feature_desc,
        "model": model,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, path)


def _ensure_timestamp_root(parent: Path) -> tuple[Path, str]:
    parent = parent.expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"kernel_regression_{ts}"
    root = parent / dirname
    if root.exists():
        raise FileExistsError(f"输出目录已存在（时间戳碰撞）: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root, dirname


def _errors_dict_from_arrays(
    gammas: np.ndarray,
    tr_ms: np.ndarray,
    te_ms: np.ndarray,
) -> TargetGammaErrors:
    return {
        "gamma": [float(g) for g in gammas],
        "train_mse": [float(x) for x in tr_ms],
        "test_mse": [float(x) for x in te_ms],
    }


def train_kernel_regression_kbc(
    training_model: TrainingDatasetModel,
    model_path: str | Path,
    alpha: float | None = None,
) -> tuple[Path, ErrorsByTarget]:
    """
    从 ``TrainingDatasetModel.records`` 训练 K/B/C，写入 ``{model_path}/kernel_regression_{timestamp}/``。

    Returns:
        saved_root: 实际 artifact 目录（含三套 joblib、CSV、manifest.json）。
        errors_by_target: 各目标与各 gamma 的 LOOCV 训练/测试 MSE。
    """
    alpha_v = float(DEFAULT_KERNEL_RIDGE_ALPHA if alpha is None else alpha)

    records = training_model.records
    if len(records) < 2:
        raise ValueError("训练至少需要 2 条样本以供 LOOCV")

    eq = np.array([r.equivalent_kg_tnt for r in records], dtype=np.float64)
    al = np.array([r.al_percent for r in records], dtype=np.float64)
    K = np.array([r.K for r in records], dtype=np.float64)
    B = np.array([r.B for r in records], dtype=np.float64)
    C = np.array([r.C for r in records], dtype=np.float64)
    X = build_X(eq, al)

    gammas = gamma_grid_from_equiv(eq, n_steps=30)
    stride_val = stride_equiv_from_array(eq)

    saved_root, ts_dir = _ensure_timestamp_root(Path(model_path))

    manifest: dict[str, Any] = {
        "timestamp_dir": ts_dir,
        "n_samples": len(records),
        "data_folder": training_model.data_folder,
        "alpha": alpha_v,
        "kernel": "rbf",
        "gamma_formula": "gamma_i = 1 + i * stride, i=0..30; stride = (max_equiv - min_equiv) / 30",
        "stride_equiv": stride_val,
        "gammas": gammas.tolist(),
        "targets": {},
    }

    targets_y = {"K": K, "B": B, "C": C}
    filenames = dict(zip(("K", "B", "C"), MODEL_ARTIFACT_FILENAMES))
    errors_by_target: dict[str, TargetGammaErrors] = {}

    for name in ("K", "B", "C"):
        ys = targets_y[name]
        gm, tr_ms, te_ms, best_g = sweep_gamma_loocv(X, ys, gammas, alpha=alpha_v)
        errors_by_target[name] = _errors_dict_from_arrays(gm, tr_ms, te_ms)

        csv_path = saved_root / f"kbc_krr_loocv_{name}.csv"
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("gamma,train_mse_loocv_mean,test_mse_loocv_mean\n")
            for gi, tt, vv in zip(gm, tr_ms, te_ms):
                fh.write(f"{gi:.12g},{tt:.12g},{vv:.12g}\n")

        model = fit_full_model(X, ys, best_g, alpha=alpha_v)
        bundle_path = saved_root / filenames[name]
        _save_model_bundle(
            bundle_path,
            model,
            target=name,
            best_gamma=best_g,
            alpha=alpha_v,
            feature_desc=FEATURE_DESC,
        )

        j = int(np.argmin(te_ms))
        manifest["targets"][name] = {
            "best_gamma": best_g,
            "best_loocv_test_mse": float(te_ms[j]),
            "best_loocv_train_mse": float(tr_ms[j]),
            "model_file": filenames[name],
            "loocv_csv": csv_path.name,
        }

    with open(saved_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    return saved_root, errors_by_target


def predict_kernel_regression_kbc(
    model_path: str | Path,
    equiv_kg_tnt: float,
    al_percent: float,
) -> tuple[float, float, float]:
    """
    在单次训练生成的 ``kernel_regression_{timestamp}`` 目录上预测 ``K,B,C``。

    ``model_path`` 须指向该子目录本身（内含 ``kbc_krr_*.joblib``）。

    ``al_percent`` 为含铝量百分比（如 30 表示 30%）；第二特征为
    ``equiv_kg_tnt * (al_percent / 100)``，与 ``build_X`` 一致。
    """
    root = Path(model_path).expanduser().resolve()
    x2 = float(equiv_kg_tnt) * (float(al_percent) / 100.0)
    xe = np.array([[equiv_kg_tnt, x2]], dtype=np.float64)

    k_hat = float(_load_bundle_predict(root / MODEL_ARTIFACT_FILENAMES[0], xe))
    b_hat = float(_load_bundle_predict(root / MODEL_ARTIFACT_FILENAMES[1], xe))
    c_hat = float(_load_bundle_predict(root / MODEL_ARTIFACT_FILENAMES[2], xe))
    return k_hat, b_hat, c_hat


def _load_bundle_predict(bundle_path: Path, X_row: np.ndarray) -> float:
    if not bundle_path.is_file():
        raise FileNotFoundError(f"缺少模型文件: {bundle_path}")
    bundle = load(bundle_path)
    m: KernelRidge = bundle["model"]
    return float(m.predict(X_row)[0])
