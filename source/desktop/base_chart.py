#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用图表样式与基础图表类

提供统一的暗色主题样式配置，及继承自 MatplotlibWidget 的图表基类，
便于具体业务图表（温度、直径）复用与扩展。
"""

from typing import Optional, Tuple, Iterable
from framework import MatplotlibWidget


def apply_dark_chart_style(ax, x_label: str, y_label: str, title: str,
                           xlim: Optional[Tuple[float, float]] = None,
                           ylim: Optional[Tuple[float, float]] = None,
                           placeholder_text: Optional[str] = None,
                           placeholder_xy: Optional[Tuple[float, float]] = None) -> None:
    """
    对给定的坐标轴应用统一的暗色主题样式，并可选设置占位文本。

    Args:
        ax: Matplotlib Axes 对象
        x_label: x轴标签
        y_label: y轴标签
        title: 图表标题
        xlim: x轴范围 (min, max)
        ylim: y轴范围 (min, max)
        placeholder_text: 占位提示文本
        placeholder_xy: 占位文本坐标 (x, y)
    """
    # 画布与坐标轴底色
    fig = ax.figure
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    # 坐标轴与刻度样式
    ax.tick_params(colors='#9ca3af', labelsize=9)
    for spine in ['bottom', 'top', 'left', 'right']:
        ax.spines[spine].set_color('#374151')

    # 轴标签与标题
    ax.set_xlabel(x_label, color='#e5e7eb', fontsize=10)
    ax.set_ylabel(y_label, color='#e5e7eb', fontsize=10)
    ax.set_title(title, color='#38bdf8', fontsize=11, fontweight='bold')

    # 轴范围
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # 网格
    ax.grid(True, alpha=0.3, color='#374151')

    # 占位文本
    if placeholder_text and placeholder_xy:
        ax.text(placeholder_xy[0], placeholder_xy[1], placeholder_text,
                ha='center', va='center',
                color='#9ca3af', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#1f2937', alpha=0.8))


class BaseChart(MatplotlibWidget):
    """统一暗色主题的基础图表，继承自 MatplotlibWidget。"""

    def __init__(self,
                 x_label: str,
                 y_label: str,
                 title: str,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 placeholder_text: Optional[str] = None,
                 placeholder_xy: Optional[Tuple[float, float]] = None,
                 width: float = 4,
                 height: float = 2.5,
                 dpi: int = 100):
        super().__init__(width=width, height=height, dpi=dpi)
        self._x_label = x_label
        self._y_label = y_label
        self._title = title
        self._xlim = xlim
        self._ylim = ylim
        self._placeholder_text = placeholder_text
        self._placeholder_xy = placeholder_xy
        self.reset()

    def reset(self) -> None:
        self.clear()
        ax = self.figure.add_subplot(111)
        apply_dark_chart_style(
            ax,
            x_label=self._x_label,
            y_label=self._y_label,
            title=self._title,
            xlim=self._xlim,
            ylim=self._ylim,
            placeholder_text=self._placeholder_text,
            placeholder_xy=self._placeholder_xy,
        )
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()

    # 子类实现
    def update_data(self, x: Iterable[float], y: Iterable[float]) -> None:  # pragma: no cover - 接口占位
        raise NotImplementedError


# 具体图表子类已移至独立文件：
# - temperature_chart.py: TemperatureChart
# - diameter_chart.py: DiameterChart


