"""从训练数据目录加载 (X, Y=[K,C])、b_mean，见设计文档 §3。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gp_model.config import DEFAULT_TRAINED_FILENAME

logger = logging.getLogger(__name__)


@dataclass
class SampleMeta:
    """单条样本元数据。"""

    path: str
    equivalent_raw: str | None = None
    al_percent_raw: str | None = None
    drag_fit_success: bool | None = None
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class Dataset:
    """n 个样本：X ∈ R^{n×2}，Y ∈ R^{n×2} 列为 [K, C]；b_mean 为训练集 B 的均值。"""

    X: np.ndarray  # float64, shape (n, 2)
    Y: np.ndarray  # float64, shape (n, 2) 列顺序 K, C
    b_mean: float
    B_train: np.ndarray  # float64, shape (n,) 每条样本的 B（仅追溯/作图）
    meta: list[SampleMeta] = field(default_factory=list)


def _parse_one_json(
    path: Path,
    *,
    strict_drag_fit_success: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, SampleMeta]:
    meta = SampleMeta(path=str(path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if strict_drag_fit_success:
            raise RuntimeError(f"无法解析 {path}: {e}") from e
        meta.skipped = True
        meta.skip_reason = f"json_error: {e}"
        return None, None, None, meta

    params = data.get("parameters") or {}
    drag = data.get("drag_fit") or {}

    meta.equivalent_raw = params.get("equivalent")
    meta.al_percent_raw = params.get("al_percent")
    meta.drag_fit_success = drag.get("success")

    if not isinstance(params, dict) or not isinstance(drag, dict):
        if strict_drag_fit_success:
            raise RuntimeError(f"{path}: 缺少 parameters 或 drag_fit")
        meta.skipped = True
        meta.skip_reason = "missing_parameters_or_drag_fit"
        return None, None, None, meta

    if drag.get("success") is not True:
        if strict_drag_fit_success:
            raise RuntimeError(f"{path}: drag_fit.success 不为 true")
        meta.skipped = True
        meta.skip_reason = "drag_fit.success is not true"
        return None, None, None, meta

    try:
        eq = float(params.get("equivalent"))
        al = float(params.get("al_percent"))
    except (TypeError, ValueError):
        if strict_drag_fit_success:
            raise RuntimeError(f"{path}: equivalent / al_percent 无效") from None
        meta.skipped = True
        meta.skip_reason = "bad_equivalent_or_al_percent"
        return None, None, None, meta

    try:
        K = float(drag["K"])
        B = float(drag["B"])
        C = float(drag["C"])
    except (KeyError, TypeError, ValueError):
        if strict_drag_fit_success:
            raise RuntimeError(f"{path}: K,B,C 缺失或无效") from None
        meta.skipped = True
        meta.skip_reason = "missing_or_invalid_K_B_C"
        return None, None, None, meta

    X = np.array([[eq, al]], dtype=np.float64)
    Y_kc = np.array([[K, C]], dtype=np.float64)
    return X, Y_kc, B, meta


def load_training_dir(
    data_dir: Path | str,
    *,
    recursive: bool = False,
    strict_drag_fit_success: bool = True,
) -> Dataset:
    """
    扫描 ``data_dir`` 下 JSON，解析 ``parameters`` / ``drag_fit``，构成 Dataset。
    默认仅顶层 ``*.json``；``recursive=True`` 时使用 ``rglob``。
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"训练数据目录不存在: {root}")

    if recursive:
        files = sorted(root.rglob("*.json"))
    else:
        files = sorted(root.glob("*.json"))

    X_list: list[list[float]] = []
    Y_list: list[list[float]] = []
    B_list: list[float] = []
    meta_list: list[SampleMeta] = []

    for fp in files:
        if fp.name == DEFAULT_TRAINED_FILENAME:
            # train 写入的同目录产物，非 drag_fit 样本；避免误报 WARNING
            continue
        X, Y_kc, B_one, meta = _parse_one_json(
            fp, strict_drag_fit_success=strict_drag_fit_success
        )
        meta_list.append(meta)
        if meta.skipped or X is None:
            if meta.skip_reason:
                logger.warning("跳过 %s: %s", fp, meta.skip_reason)
            continue
        assert Y_kc is not None and B_one is not None
        X_list.append(X[0].tolist())
        Y_list.append(Y_kc[0].tolist())
        B_list.append(B_one)

    if not X_list:
        raise ValueError(
            f"目录 {root} 中无有效训练样本（请检查 drag_fit.success 与 K,B,C）"
        )

    b_mean = float(np.mean(np.asarray(B_list, dtype=np.float64)))
    return Dataset(
        X=np.asarray(X_list, dtype=np.float64),
        Y=np.asarray(Y_list, dtype=np.float64),
        b_mean=b_mean,
        B_train=np.asarray(B_list, dtype=np.float64),
        meta=meta_list,
    )


def parse_x_star_json(raw: dict[str, Any] | list[Any]) -> np.ndarray:
    """
    解析推理输入：单点 ``{equivalent, al_percent}``、``{points: [...]}`` 或 JSON 数组。
    返回 shape ``(m, 2)``。
    """
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        if "points" in raw:
            rows = raw["points"]
        elif "equivalent" in raw and "al_percent" in raw:
            rows = [raw]
        else:
            raise ValueError("x_star JSON 需为 {equivalent, al_percent}、{points:[]} 或数组")
    else:
        raise TypeError("x_star 需为 JSON object 或 array")

    out: list[list[float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"points[{i}] 需为 object")
        out.append([float(row["equivalent"]), float(row["al_percent"])])
    return np.asarray(out, dtype=np.float64)


def load_x_star_path(path: Path | str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_x_star_json(data)
