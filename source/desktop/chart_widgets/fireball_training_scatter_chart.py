#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练 — 炸药当量与火球参数的散点图

与原型 `machine_vision_ui_prototype.html` 中三张散点一致：横轴为炸药当量，
纵轴为最大直径（K）、初始状态常数（B）、时间常数（C）等之一；第三点维度通过散点大小体现
（通常为含铝量等正相关标量）。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .base_chart import BaseChart, apply_dark_chart_style, FONT_SIZE_BODY, FONT_FAMILY


# 与原型青色散点相近
COLOR_SCATTER_FACE = '#38bdf8'
COLOR_SCATTER_FACE_ALPHA = 0.55
COLOR_SCATTER_EDGE = '#7dd3fc'
SIZE_POINTS_MIN = 25.0
SIZE_POINTS_MAX = 220.0
AXIS_PADDING_RATIO = 0.08


class FireballTrainingScatterChart(BaseChart):
    """当量–指标散点图，点的大小映射第三维标量（如含铝量 %）。"""

    X_LABEL_DEFAULT = '炸药当量 (kg TNT)'

    def __init__(
        self,
        title: str,
        y_label: str,
        size_legend_hint: str = '点大小 ∝ 第三维数值',
        width: float = 4,
        height: float = 2.5,
        dpi: int = 100,
    ):
        super().__init__(
            x_label=self.X_LABEL_DEFAULT,
            y_label=y_label,
            title=title,
            xlim=(0.0, 12.0),
            ylim=(0.0, 16.0),
            placeholder_text=None,
            placeholder_xy=None,
            width=width,
            height=height,
            dpi=dpi,
        )
        self._size_legend_hint = size_legend_hint
        self._polish_chart_frame()

    def _polish_chart_frame(self) -> None:
        """与 model_tab 图表一致：无外框线、无多余留白。"""
        lay = self.layout()
        if lay is not None:
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
        self.canvas.setStyleSheet("border: none; background-color: transparent;")

    # --------------------------- 工厂（对应原型三张图标题） --------------------------- #
    @classmethod
    def for_max_diameter(cls, width: float = 4, height: float = 2.5, dpi: int = 100) -> 'FireballTrainingScatterChart':
        return cls(
            title='火球最大直径 — 炸药当量',
            y_label='最大直径 (m)',
            size_legend_hint='点大小 ∝ 含铝量',
            width=width,
            height=height,
            dpi=dpi,
        )

    @classmethod
    def for_initial_state_constant(cls, width: float = 4, height: float = 2.5, dpi: int = 100) -> 'FireballTrainingScatterChart':
        """拖曳拟合参数 B（初始状态常数），纵轴为 B 数值。"""
        return cls(
            title='初始状态常数 — 炸药当量',
            y_label='初始状态常数 B',
            size_legend_hint='点大小 ∝ 含铝量',
            width=width,
            height=height,
            dpi=dpi,
        )

    @classmethod
    def for_initial_diameter(cls, width: float = 4, height: float = 2.5, dpi: int = 100) -> 'FireballTrainingScatterChart':
        """兼容旧名称，与 `for_initial_state_constant` 相同。"""
        return cls.for_initial_state_constant(width=width, height=height, dpi=dpi)

    @classmethod
    def for_time_constant(cls, width: float = 4, height: float = 2.5, dpi: int = 100) -> 'FireballTrainingScatterChart':
        return cls(
            title='爆炸时间常数 — 炸药当量',
            y_label='时间常数 (ms)',
            size_legend_hint='点大小 ∝ 含铝量',
            width=width,
            height=height,
            dpi=dpi,
        )

    # --------------------------- 内部 --------------------------- #
    @staticmethod
    def _to_float_array(seq: Sequence[float]) -> np.ndarray:
        return np.asarray(seq, dtype=float).ravel()

    @staticmethod
    def _compute_sizes(points_metric: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """将标量映射为 matplotlib scatter 的 s（面积，单位 pt²）。"""
        m = np.asarray(points_metric, dtype=float).ravel()
        if m.size == 0:
            return np.array([], dtype=float), (0.0, 1.0)
        m_min = float(np.nanmin(m))
        m_max = float(np.nanmax(m))
        span = m_max - m_min
        if span <= 0 or not np.isfinite(span):
            t = np.zeros_like(m, dtype=float)
        else:
            t = (m - m_min) / span
        t = np.clip(np.where(np.isfinite(t), t, 0.0), 0.0, 1.0)
        s_pts = SIZE_POINTS_MIN + t * (SIZE_POINTS_MAX - SIZE_POINTS_MIN)
        return s_pts, (m_min, m_max)

    @staticmethod
    def _padded_lim(values: np.ndarray) -> Tuple[float, float]:
        v = values[np.isfinite(values)]
        if v.size == 0:
            return (0.0, 1.0)
        lo, hi = float(np.min(v)), float(np.max(v))
        pad = (hi - lo) * AXIS_PADDING_RATIO if hi > lo else 0.1 * (abs(lo) + 1.0)
        return lo - pad, hi + pad

    def reset(self) -> None:
        super().reset()

    # --------------------------- 对外 API --------------------------- #
    def update_data(  # type: ignore[override]
        self,
        equivalent_kg_tnt: Iterable[float],
        y_values: Iterable[float],
        size_metric: Iterable[float],
        *,
        x_label: Optional[str] = None,
    ) -> None:
        """
        更新散点。

        Args:
            equivalent_kg_tnt: 横轴炸药当量 (kg TNT)
            y_values: 纵轴观测（最大直径、初始直径或时间常数）
            size_metric: 决定散点大小的非负相关标量（如含铝量 %）；越大点越大。
            x_label: 可选覆盖默认横轴标签
        """
        x = self._to_float_array(list(equivalent_kg_tnt))
        y = self._to_float_array(list(y_values))
        z = self._to_float_array(list(size_metric))

        n = min(x.size, y.size, z.size)
        if n == 0:
            self.reset()
            return

        x = x[:n]
        y = y[:n]
        z = z[:n]
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(mask):
            self.reset()
            return

        x = x[mask]
        y = y[mask]
        z = z[mask]
        s_pts, _ = self._compute_sizes(z)

        self.clear()
        try:
            self.figure.set_constrained_layout(True)
        except Exception:
            pass

        ax = self.figure.add_subplot(111)
        xlim = self._padded_lim(x)
        ylim = self._padded_lim(y)

        apply_dark_chart_style(
            ax,
            x_label=x_label or self.X_LABEL_DEFAULT,
            y_label=self._y_label,
            title=self._title,
            xlim=xlim,
            ylim=ylim,
        )

        ax.scatter(
            x,
            y,
            s=s_pts,
            c=COLOR_SCATTER_FACE,
            alpha=COLOR_SCATTER_FACE_ALPHA,
            edgecolors=COLOR_SCATTER_EDGE,
            linewidths=0.6,
        )

        foot = self._size_legend_hint
        ax.text(
            xlim[0],
            ylim[0],
            foot,
            transform=ax.transData,
            ha='left',
            va='bottom',
            color='#94a3b8',
            fontsize=FONT_SIZE_BODY,
            fontfamily=FONT_FAMILY,
        )

        self.canvas.draw()
