#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用图表样式与基础图表类

提供统一的暗色主题样式配置，及继承自 MatplotlibWidget 的图表基类，
便于具体业务图表（温度、直径）复用与扩展。
"""

from typing import Optional, Tuple, Iterable
import matplotlib
from framework import MatplotlibWidget

# 设置全局字体配置，确保中文正确显示
# 优先使用系统支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 字体大小常量
FONT_SIZE_BODY = 6.5  # 正文字体大小（用于轴标签、刻度标签、占位文本等）
FONT_SIZE_TITLE = 7   # 标题字体大小

# 字体族常量
FONT_FAMILY = 'sans-serif'  # 字体族

# 颜色常量
COLOR_BACKGROUND = '#111827'      # 背景色
COLOR_BORDER = '#374151'           # 边框/网格颜色
COLOR_TEXT = 'white'               # 正文文本颜色
COLOR_TITLE = '#38bdf8'            # 标题颜色
COLOR_PLACEHOLDER = '#9ca3af'      # 占位文本颜色
COLOR_PLACEHOLDER_BG = '#1f2937'   # 占位文本背景色

# 透明度常量
ALPHA_GRID = 0.3                   # 网格透明度
ALPHA_PLACEHOLDER_BG = 0.8         # 占位文本背景透明度

# 布局常量
LAYOUT_PAD = 0.8                   # 仅兜底使用；默认采用 constrained_layout
SUBPLOT_LEFT = 0.12
SUBPLOT_RIGHT = 0.98
SUBPLOT_TOP = 0.88
SUBPLOT_BOTTOM = 0.18
PLACEHOLDER_BBOX_PAD = 0.3         # 占位文本边框内边距
PLACEHOLDER_BBOX_STYLE = 'round'    # 占位文本边框样式


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
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)

    # 坐标轴与刻度样式（使用常量）
    ax.tick_params(colors=COLOR_TEXT, labelsize=FONT_SIZE_BODY)
    for spine in ['bottom', 'top', 'left', 'right']:
        ax.spines[spine].set_color(COLOR_BORDER)

    # 轴标签与标题（使用常量）
    ax.set_xlabel(x_label, color=COLOR_TEXT, fontsize=FONT_SIZE_BODY, fontfamily=FONT_FAMILY)
    ax.set_ylabel(y_label, color=COLOR_TEXT, fontsize=FONT_SIZE_BODY, fontfamily=FONT_FAMILY)
    ax.set_title(title, color=COLOR_TITLE, fontsize=FONT_SIZE_TITLE, fontweight='bold', fontfamily=FONT_FAMILY)

    # 轴范围
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # 网格（使用常量）
    ax.grid(True, alpha=ALPHA_GRID, color=COLOR_BORDER)

    # 占位文本（使用常量）
    if placeholder_text and placeholder_xy:
        ax.text(placeholder_xy[0], placeholder_xy[1], placeholder_text,
                ha='center', va='center',
                color=COLOR_PLACEHOLDER, fontsize=FONT_SIZE_BODY, fontfamily=FONT_FAMILY,
                bbox=dict(boxstyle=f"{PLACEHOLDER_BBOX_STYLE},pad={PLACEHOLDER_BBOX_PAD}",
                         facecolor=COLOR_PLACEHOLDER_BG, alpha=ALPHA_PLACEHOLDER_BG))


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
        # 采用 constrained_layout，确保标题/坐标轴/图例不被裁切
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
            xlim=self._xlim,
            ylim=self._ylim,
            placeholder_text=self._placeholder_text,
            placeholder_xy=self._placeholder_xy,
        )
        # 如遇个别环境仍有裁切，可启用下面的兜底边距
        # self.figure.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
        #                             top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM)
        self.canvas.draw()

    # 子类实现
    def update_data(self, x: Iterable[float], y: Iterable[float]) -> None:  # pragma: no cover - 接口占位
        raise NotImplementedError


# 具体图表子类已移至独立文件：
# - temperature_chart.py: TemperatureChart
# - diameter_chart.py: DiameterChart


