#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从训练散点截图中提取曲线/散点并重绘为白底高清图（用于报告插图）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import label as nd_label

ROOT = Path(__file__).resolve().parent.parent

CURVE_COLORS = ("#f97316", "#a78bfa", "#34d399")
CURVE_LABELS = ("含铝量 20 %", "含铝量 30 %", "含铝量 40 %")
SCATTER_FACE = "#38bdf8"
SCATTER_EDGE = "#0284c7"


@dataclass(frozen=True)
class ChartSpec:
    path: Path
    title: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    margins: tuple[int, int, int, int]  # left, right, top, bottom (px)


CHARTS = (
    ChartSpec(
        ROOT / "images/截屏2026-06-13 16.27.07.png",
        "火球最大直径 — 炸药当量",
        "最大直径 (m)",
        (0.0, 150.0),
        (2.5, 22.5),
        (78, 20, 48, 62),
    ),
    ChartSpec(
        ROOT / "images/截屏2026-06-13 16.27.20.png",
        "初始状态常数 — 炸药当量",
        "初始状态常数 B",
        (0.0, 150.0),
        (0.550, 0.570),
        (78, 20, 48, 62),
    ),
    ChartSpec(
        ROOT / "images/截屏2026-06-13 16.27.28.png",
        "爆炸时间常数 — 炸药当量",
        "时间常数 (ms)",
        (0.0, 150.0),
        (0.0, 0.0030),
        (78, 20, 48, 62),
    ),
)


def _color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {
        "orange": (r > 170) & (g > 70) & (g < 175) & (b < 95),
        "purple": (r > 115) & (r < 205) & (g > 95) & (g < 185) & (b > 195),
        "green": (r < 105) & (g > 145) & (b > 95) & (b < 205),
        "blue": (b > 95) & (g > 70) & (r < 105) & ~((r > 170) & (g > 70)),
    }


def _pixel_to_data(
    xs: np.ndarray,
    ys: np.ndarray,
    spec: ChartSpec,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    left, right, top, bottom = spec.margins
    plot_w = w - left - right
    plot_h = h - top - bottom
    x0, x1 = spec.xlim
    y0, y1 = spec.ylim
    xd = x0 + (xs - left) / plot_w * (x1 - x0)
    yd = y1 - (ys - top) / plot_h * (y1 - y0)
    return xd, yd


def _extract_curve(mask: np.ndarray, spec: ChartSpec, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    left, right, top, bottom = spec.margins
    # 排除左上角图例区域，避免污染曲线
    legend = np.zeros_like(mask, dtype=bool)
    legend[top : top + 95, left : left + 165] = True
    mask = mask & ~legend

    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.array([]), np.array([])
    xd, yd = _pixel_to_data(xs.astype(float), ys.astype(float), spec, shape)
    order = np.argsort(xd)
    xd, yd = xd[order], yd[order]
    bins = np.linspace(spec.xlim[0], spec.xlim[1], 100)
    idx = np.digitize(xd, bins) - 1
    x_out, y_out = [], []
    y_span = spec.ylim[1] - spec.ylim[0]
    for b in range(len(bins) - 1):
        sel = idx == b
        if np.sum(sel) < 2:
            continue
        y_med = float(np.median(yd[sel]))
        if y_out and abs(y_med - y_out[-1]) > 0.35 * y_span:
            continue
        x_out.append(float(np.median(xd[sel])))
        y_out.append(y_med)
    if len(x_out) < 3:
        return np.array([]), np.array([])
    x_arr = np.asarray(x_out)
    y_arr = np.asarray(y_out)
    # 轻度滑动平均，消除锯齿
    k = 3
    y_smooth = np.convolve(y_arr, np.ones(k) / k, mode="same")
    y_smooth[0] = y_arr[0]
    y_smooth[-1] = y_arr[-1]
    return x_arr, y_smooth


def _extract_scatter(spec: ChartSpec, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _color_masks(rgb)["blue"]
    h, w = rgb.shape[:2]
    left, right, top, bottom = spec.margins
    roi = np.zeros_like(mask)
    roi[top : h - bottom, left : w - right] = mask[top : h - bottom, left : w - right]
    labeled, n = nd_label(roi)
    eq, yy, al = [], [], []
    for k in range(1, n + 1):
        ys, xs = np.where(labeled == k)
        if xs.size < 12:
            continue
        xd, yd = _pixel_to_data(xs.astype(float), ys.astype(float), spec, (h, w))
        eq.append(float(np.median(xd)))
        yy.append(float(np.median(yd)))
        al.append(float(np.sqrt(xs.size) * 3.5))
    if not eq:
        return np.array([]), np.array([]), np.array([])
    eq_a = np.asarray(eq)
    yy_a = np.asarray(yy)
    al_a = np.asarray(al)
    if al_a.max() > al_a.min():
        al_a = 20.0 + 200.0 * (al_a - al_a.min()) / (al_a.max() - al_a.min())
    else:
        al_a = np.full_like(al_a, 80.0)
    return eq_a, yy_a, al_a


def render_light_chart(spec: ChartSpec, rgb: np.ndarray, out_path: Path, *, dpi: int = 150) -> None:
    masks = _color_masks(rgb)
    h, w = rgb.shape[:2]
    fig_w = w / dpi
    fig_h = h / dpi

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    eq, yy, sizes = _extract_scatter(spec, rgb)
    if eq.size:
        ax.scatter(
            eq,
            yy,
            s=sizes,
            c=SCATTER_FACE,
            alpha=0.55,
            edgecolors=SCATTER_EDGE,
            linewidths=0.6,
            zorder=2,
        )

    for key, color in zip(("orange", "purple", "green"), CURVE_COLORS):
        xcv, ycv = _extract_curve(masks[key], spec, (h, w))
        if xcv.size:
            ax.plot(xcv, ycv, color=color, linewidth=1.6, alpha=0.92, zorder=4)

    ax.set_xlim(*spec.xlim)
    ax.set_ylim(*spec.ylim)
    ax.set_xticks(np.arange(spec.xlim[0], spec.xlim[1] + 1, 25))
    ax.set_xlabel("炸药当量 (kg TNT)", color="#111827", fontsize=10)
    ax.set_ylabel(spec.ylabel, color="#111827", fontsize=10)
    ax.set_title(spec.title, color="#0f172a", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#111827", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#64748b")
    ax.grid(True, alpha=0.35, color="#cbd5e1")

    handles = [
        Line2D([0], [0], color=c, linewidth=1.6, label=lbl)
        for c, lbl in zip(CURVE_COLORS, CURVE_LABELS)
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=9,
        title="核回归｜横轴炸药当量",
        framealpha=0.95,
        facecolor="white",
        edgecolor="#94a3b8",
        labelcolor="#111827",
    )
    leg.get_title().set_color("#475569")
    leg.get_title().set_fontsize(9)

    ax.text(
        spec.xlim[0],
        spec.ylim[0],
        "点大小 ∝ 含铝量",
        fontsize=9,
        color="#475569",
        ha="left",
        va="bottom",
    )

    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    for spec in CHARTS:
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        rgb = np.array(Image.open(spec.path).convert("RGB"))
        render_light_chart(spec, rgb, spec.path)
        white_copy = spec.path.with_name(spec.path.stem + "_白底.png")
        render_light_chart(spec, rgb, white_copy)
        print(f"written {spec.path}")


if __name__ == "__main__":
    main()
