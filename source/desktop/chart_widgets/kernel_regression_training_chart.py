#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练 — 核回归带宽 σ 与训练/测试 MSE 曲线

与原型图中一致：横轴带宽 σ（对数刻度），蓝色训练误差、红色测试误差，
绿色叉号标示测试误差最小点附近。
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .base_chart import BaseChart, apply_dark_chart_style, FONT_SIZE_BODY, FONT_FAMILY


SIGMA_MIN = 0.01
SIGMA_MAX = 10.0
COLOR_TRAIN = '#3b82f6'
COLOR_TEST = '#ef4444'
COLOR_MIN_MARKER = '#22c55e'


class KernelRegressionTrainingCurveChart(BaseChart):
    """核回归 σ–MSE；仅在有后端传入序列时绘图，默认不绘制示意曲线。"""

    def __init__(self, width: float = 4, height: float = 2.5, dpi: int = 100):
        super().__init__(
            x_label='σ（带宽）',
            y_label='MSE',
            title='训练曲线（核回归：σ — MSE）',
            xlim=None,
            ylim=None,
            placeholder_text=None,
            placeholder_xy=None,
            width=width,
            height=height,
            dpi=dpi,
        )
        self._polish_chart_frame()

    def _polish_chart_frame(self) -> None:
        lay = self.layout()
        if lay is not None:
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
        self.canvas.setStyleSheet("border: none; background-color: transparent;")

    def set_chart_title(self, title: str) -> None:
        """设置 Matplotlib 图题；需在随后 `update_data` / `reset` 绘制时生效。"""
        self._title = title

    def update_data(  # type: ignore[override]
        self,
        sigma: Optional[Iterable[float]] = None,
        train_mse: Optional[Iterable[float]] = None,
        test_mse: Optional[Iterable[float]] = None,
        *,
        y_max: Optional[float] = None,
    ) -> None:
        """
        绘制误差曲线。

        sigma / train_mse / test_mse 须同时为有效序列且至少两点；否则等价于清空图（不写示意数据）。
        """
        if sigma is None or train_mse is None or test_mse is None:
            self.reset()
            return
        sig = np.asarray(list(sigma), dtype=float).ravel()
        tr = np.asarray(list(train_mse), dtype=float).ravel()
        te = np.asarray(list(test_mse), dtype=float).ravel()
        n = min(sig.size, tr.size, te.size)
        if n < 2:
            self.reset()
            return
        sig, tr, te = sig[:n], tr[:n], te[:n]
        mask = np.isfinite(sig) & np.isfinite(tr) & np.isfinite(te)
        if np.sum(mask) < 2:
            self.reset()
            return
        sig, tr, te = sig[mask], tr[mask], te[mask]
        order = np.argsort(sig)
        sig, tr, te = sig[order], tr[order], te[order]

        y_hi = float(y_max) if y_max is not None else float(max(14.0, np.nanmax(te) * 1.05, np.nanmax(tr) * 1.05))

        self.clear()
        try:
            self.figure.set_constrained_layout(True)
        except Exception:
            pass

        ax = self.figure.add_subplot(111)
        apply_dark_chart_style(
            ax,
            x_label=self._x_label,
            y_label=self._y_label,
            title=self._title,
            ylim=(0.0, y_hi),
        )
        ax.set_xscale('log')
        try:
            s_lo = max(float(np.nanmin(sig[np.isfinite(sig)])) * 0.85, SIGMA_MIN)
            s_hi = min(float(np.nanmax(sig[np.isfinite(sig)])) * 1.15, SIGMA_MAX)
            if s_hi <= s_lo:
                ax.set_xlim(SIGMA_MIN, SIGMA_MAX)
            else:
                ax.set_xlim(s_lo, s_hi)
        except Exception:
            ax.set_xlim(SIGMA_MIN, SIGMA_MAX)

        ax.plot(sig, tr, color=COLOR_TRAIN, linewidth=1.5, label='训练误差')
        ax.plot(sig, te, color=COLOR_TEST, linewidth=1.5, label='测试误差')

        # 标注测试误差最小的 σ（离散网格中取最小）
        j = int(np.nanargmin(te))
        bx, by = float(sig[j]), float(te[j])
        ax.plot(
            [bx],
            [by],
            color=COLOR_MIN_MARKER,
            marker='x',
            markersize=9,
            markeredgewidth=2,
            linestyle='None',
            label='最小测试误差',
        )

        try:
            leg = ax.legend(
                fontsize=FONT_SIZE_BODY,
                prop={'family': FONT_FAMILY},
                loc='upper right',
                framealpha=0.9,
                facecolor='#1f2937',
                edgecolor='#374151',
            )
            for text in leg.get_texts():
                text.set_color('#e5e7eb')
        except Exception:
            pass

        self.canvas.draw()
