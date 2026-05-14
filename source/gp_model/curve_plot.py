"""直径–时间曲线 D(t)=K(1-B exp(-C t^2))，见设计文档 §5。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotConfig:
    time_ms_min: float = 0.0
    time_ms_max: float = 75.0
    num_points: int = 300


def plot_config_from_hyperparams(hp: dict[str, Any]) -> PlotConfig:
    p = hp.get("plot") or {}
    return PlotConfig(
        time_ms_min=float(p.get("time_ms_min", 0.0)),
        time_ms_max=float(p.get("time_ms_max", 75.0)),
        num_points=int(p.get("num_points", 300)),
    )


def diameter_curve(t_ms: np.ndarray, K: float, B: float, C: float) -> np.ndarray:
    """D(t) = K * (1 - B * exp(-C * t^2))，t 与 JSON 一致为 ms。"""
    t = np.asarray(t_ms, dtype=np.float64)
    if C < 0:
        raise ValueError("C < 0 时拖曳公式在实现中未定义，拒绝静默绘图")
    return K * (1.0 - B * np.exp(-C * (t**2)))


def plot_diameter_curve(
    K: float,
    B: float,
    C: float,
    cfg: PlotConfig,
    *,
    equivalent: float | None = None,
    al_percent: float | None = None,
    title_suffix: str | None = None,
) -> plt.Figure:
    t = np.linspace(cfg.time_ms_min, cfg.time_ms_max, cfg.num_points, dtype=np.float64)
    D = diameter_curve(t, K, B, C)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, D, "b-", lw=1.5)
    ax.set_xlabel("时间 t (ms)")
    ax.set_ylabel("直径 D (m)")
    parts = []
    if equivalent is not None and al_percent is not None:
        parts.append(f"equivalent={equivalent}, al%={al_percent}")
    parts.append(f"K={K:.6g}, B={B:.6g}, C={C:.6g}")
    if title_suffix:
        parts.append(title_suffix)
    ax.set_title("火球直径–时间（拖曳近似）\n" + " | ".join(parts))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_kbc_json(path: Path | str) -> dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in ("K", "B", "C"):
        if k not in data:
            raise ValueError(f"kbc.json 缺少 {k}")
    return data
