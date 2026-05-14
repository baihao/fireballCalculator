"""超参数 JSON：默认值、合并、训练产物读写（见 design §6.3）。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
DEFAULT_TRAINED_FILENAME = "trained_hyperparams.json"
# predict / sweep-k 等 CLI 默认输出目录（相对当前工作目录）
DEFAULT_TRAINING_OUTPUT_DIR = "training_output"


def _default_covariance_block() -> dict[str, Any]:
    return {
        "input_kernel": "RBF",
        "length_scale_init": 1.0,
        "length_scale_init_mode": "span_fraction",
        "length_scale_init_span_fraction": 0.2,
        "length_scale_init_span_floor": 1e-6,
    }


def default_hyperparams() -> dict[str, Any]:
    """与 document/fireball_gp_mogp_module_design.md §6.3 对齐：双单任务 GP。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_family": "dual_single_output_gp",
        "covariance_K": copy.deepcopy(_default_covariance_block()),
        "covariance_C": copy.deepcopy(_default_covariance_block()),
        "optimization": {
            "max_iter": 200,
            "learning_rate": 0.05,
            "num_restarts": 1,
            "restart_base_seed": 0,
        },
        "io": {
            "strict_drag_fit_success": False,
            "predict_plot_filename": "diameter_vs_time.png",
            "predict_point_plot_pattern": "diameter_{index}.png",
        },
        "plot": {
            "time_ms_min": 0.0,
            "time_ms_max": 75.0,
            "num_points": 300,
        },
        "covariance_autoscale": {
            "enabled": True,
            "observation_noise_max_mult": 0.5,
            "observation_noise_init_mult": 0.1,
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge_hyperparams(user: dict[str, Any] | None) -> dict[str, Any]:
    """合并默认；若仅有 legacy `covariance`，复制到 `covariance_K` / `covariance_C`。"""
    u = copy.deepcopy(user) if user else None
    if u and "covariance" in u:
        if "covariance_K" not in u and "covariance_C" not in u:
            c = copy.deepcopy(u["covariance"])
            u["covariance_K"] = c
            u["covariance_C"] = c
    return _deep_merge(default_hyperparams(), u or {})


def load_hyperparams_json(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("超参数文件须为 JSON object")
    return merge_hyperparams(data)


def tensor_state_to_json(obj: Any) -> Any:
    """递归将 torch.Tensor 转为可 JSON 序列化结构。"""
    import torch

    if isinstance(obj, torch.Tensor):
        return {
            "__torch_tensor__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.detach().cpu().reshape(-1).tolist(),
        }
    if isinstance(obj, dict):
        return {k: tensor_state_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_state_to_json(v) for v in obj]
    return obj


def tensor_state_from_json(obj: Any) -> Any:
    import torch

    if isinstance(obj, dict) and obj.get("__torch_tensor__"):
        ds = obj["dtype"].replace("torch.", "")
        dtype = getattr(torch, ds) if hasattr(torch, ds) else torch.float32
        t = torch.tensor(obj["data"], dtype=dtype)
        return t.reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        return {k: tensor_state_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_state_from_json(v) for v in obj]
    return obj


def save_trained_artifact(
    path: Path | str,
    hyperparams: dict[str, Any],
    *,
    training_X: list[list[float]],
    training_Y: list[list[float]],
    data_dir: str | None,
    b_mean: float,
    model_K_state: dict[str, Any],
    likelihood_K_state: dict[str, Any],
    model_C_state: dict[str, Any],
    likelihood_C_state: dict[str, Any],
) -> None:
    """写入训练 JSON（schema 3：K/C 两套 state_dict）。"""
    payload = copy.deepcopy(hyperparams)
    payload["schema_version"] = SCHEMA_VERSION
    payload["model_family"] = "dual_single_output_gp"
    payload["b_mean"] = float(b_mean)
    payload["training"] = {
        "X": training_X,
        "Y": training_Y,
        "data_dir": data_dir,
    }
    payload["torch"] = {
        "model_K_state_dict": tensor_state_to_json(model_K_state),
        "likelihood_K_state_dict": tensor_state_to_json(likelihood_K_state),
        "model_C_state_dict": tensor_state_to_json(model_C_state),
        "likelihood_C_state_dict": tensor_state_to_json(likelihood_C_state),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_trained_artifact(path: Path | str) -> dict[str, Any]:
    """加载训练 JSON；MOGP（单一 model_state_dict）显式报错。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("训练产物须为 JSON object")

    raw_torch = raw.get("torch")
    if isinstance(raw_torch, dict):
        if "model_state_dict" in raw_torch and "model_K_state_dict" not in raw_torch:
            raise ValueError(
                "训练产物为旧版 MOGP（单一 model_state_dict / schema≤2）。"
                "请用当前代码重新 train，生成双单任务 GP（schema 3）。"
            )
        if "model_K_state_dict" not in raw_torch or "model_C_state_dict" not in raw_torch:
            raise ValueError(
                "训练产物 torch 缺少 model_K_state_dict / model_C_state_dict，"
                "请重新 train。"
            )

    merged = merge_hyperparams(
        {
            k: v
            for k, v in raw.items()
            if k
            not in (
                "torch",
                "training",
                "schema_version",
                "b_mean",
                "input_standardization",
                "output_standardization",
            )
        }
    )
    merged.update(raw)
    if "torch" in raw and isinstance(raw["torch"], dict):
        t = raw["torch"]
        merged["torch"] = {
            "model_K_state_dict": tensor_state_from_json(t["model_K_state_dict"]),
            "likelihood_K_state_dict": tensor_state_from_json(t["likelihood_K_state_dict"]),
            "model_C_state_dict": tensor_state_from_json(t["model_C_state_dict"]),
            "likelihood_C_state_dict": tensor_state_from_json(t["likelihood_C_state_dict"]),
        }
    return merged
