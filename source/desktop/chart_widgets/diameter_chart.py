#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径图表组件

基于 BaseChart，提供直径曲线的渲染与数据更新接口。
"""

from .base_chart import BaseChart, FONT_SIZE_BODY, FONT_FAMILY
from typing import Optional, Tuple
import numpy as np

# 线条样式常量
LINE_WIDTH = 2                    # 线条宽度
SMOOTH_POINTS = 300               # 平滑曲线点数
AXIS_PADDING_RATIO = 0.1         # 坐标轴边距比例（10%）
X_PADDING_DEFAULT = 1.0           # x轴默认边距
Y_PADDING_DEFAULT = 0.1           # y轴默认边距

# 颜色常量
COLOR_DIAMETER = '#f59e0b'        # 直径曲线颜色
COLOR_FIT = 'b'                  # 拟合曲线颜色（蓝色）
COLOR_CUTOFF = 'orange'           # 截断线颜色


class DiameterChart(BaseChart):
    """直径图表。"""

    def __init__(self, width: float = 4, height: float = 2.5, dpi: int = 100):
        super().__init__(
            x_label="时间 (ms)",
            y_label="直径 (m)",
            title="火球直径随时间变化",
            xlim=(0, 140),
            ylim=(0, 2),
            placeholder_text="提取完成后显示",
            placeholder_xy=(70, 1),
            width=width,
            height=height,
            dpi=dpi,
        )
        self._line_color = COLOR_DIAMETER
        self._placeholder_text = "提取完成后显示"

    # --------------------------- 公共API --------------------------- #
    def _compute_axis_limits(self, time_ms, diameter_m, K, B, C):
        """
        依据原始数据与拟合曲线（若提供）计算坐标范围。
        返回 (xlim, ylim)
        """
        xlim = self._xlim
        ylim = self._ylim
        if not time_ms or len(time_ms) == 0:
            return xlim, ylim
        try:
            time_arr = np.array(time_ms, dtype=float)
            x_min, x_max = float(np.min(time_arr)), float(np.max(time_arr))

            y_candidates = []
            if diameter_m is not None and len(diameter_m) > 0:
                diam_arr = np.array(diameter_m, dtype=float)
                valid_mask = np.isfinite(time_arr) & np.isfinite(diam_arr)
                if np.any(valid_mask):
                    y_candidates.append(np.min(diam_arr[valid_mask]))
                    y_candidates.append(np.max(diam_arr[valid_mask]))

            if K is not None and B is not None and C is not None:
                t_smooth = np.linspace(x_min, x_max, SMOOTH_POINTS)
                D_smooth = K * (1.0 - B * np.exp(-C * (t_smooth ** 2)))
                if np.size(D_smooth) > 0 and np.all(np.isfinite(D_smooth)):
                    y_candidates.append(float(np.min(D_smooth)))
                    y_candidates.append(float(np.max(D_smooth)))

            if y_candidates:
                y_min, y_max = float(min(y_candidates)), float(max(y_candidates))
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_padding = x_range * AXIS_PADDING_RATIO if x_range > 0 else X_PADDING_DEFAULT
                y_padding = y_range * AXIS_PADDING_RATIO if y_range > 0 else Y_PADDING_DEFAULT
                xlim = (x_min - x_padding, x_max + x_padding)
                ylim = (max(0.0, y_min - y_padding), y_max + y_padding)
        except Exception:
            pass
        return xlim, ylim

    def set_placeholder(self, text: str, xy: Optional[Tuple[float, float]] = None) -> None:
        """设置占位符文本与位置，并刷新占位图。"""
        self._placeholder_text = text
        if xy is not None:
            self._placeholder_xy = xy
        self.reset()

    def draw_raw_diameter(self, ax, time_ms, diameter_m) -> None:
        if time_ms and diameter_m:
            ax.plot(time_ms, diameter_m, color=self._line_color, linewidth=LINE_WIDTH, label='原始直径')

    def draw_fit(self, ax, time_ms, K: float, B: float, C: float) -> None:
        if time_ms and len(time_ms) > 0 and K is not None and B is not None and C is not None:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                D_smooth = K * (1.0 - B * np.exp(-C * (t_smooth ** 2)))
                ax.plot(t_smooth, D_smooth, f'{COLOR_FIT}-', linewidth=LINE_WIDTH, label='拖曳函数拟合')
            except Exception:
                pass

    def draw_cutoff(self, ax, xlim, cutoff_ms: Optional[float]) -> None:
        if cutoff_ms is None:
            return
        try:
            cutoff_val = float(cutoff_ms)
            if xlim and cutoff_val >= xlim[0] and cutoff_val <= xlim[1]:
                ax.axvline(x=cutoff_val, color=COLOR_CUTOFF, linestyle='--', linewidth=LINE_WIDTH,
                           label=f'数据截断点 ({cutoff_val:.1f}ms)')
        except Exception:
            pass

    def update_data(self, time_ms, diameter_m=None, K: float = None, B: float = None, C: float = None,
                    cutoff_ms: float = None) -> None:
        """
        更新直径图。允许 diameter_m 为空，此时需要提供 K,B,C 绘制拟合曲线。
        """
        # 基础有效性
        if not time_ms or len(time_ms) == 0:
            self.reset()
            return

        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)

        # 计算数据范围（包括原始数据或拟合曲线）
        xlim, ylim = self._compute_axis_limits(time_ms, diameter_m, K, B, C)

        # 应用统一的暗色主题样式
        from .base_chart import apply_dark_chart_style
        apply_dark_chart_style(
            ax,
            x_label=self._x_label,
            y_label=self._y_label,
            title=self._title,
            xlim=xlim,
            ylim=ylim,
        )

        # 如果原始数据为空，则要求存在拟合参数
        if (diameter_m is None or len(diameter_m) == 0) and not (K is not None and B is not None and C is not None):
            # 无法绘制，显示占位符
            self.set_placeholder(self._placeholder_text or "无可绘制数据", self._placeholder_xy)
            return

        # 分别绘制各部分
        self.draw_raw_diameter(ax, time_ms, diameter_m)
        if K is not None and B is not None and C is not None:
            self.draw_fit(ax, time_ms, K, B, C)
        self.draw_cutoff(ax, xlim, cutoff_ms)

        # 刷新图例与画布
        try:
            ax.legend(fontsize=FONT_SIZE_BODY, prop={'family': FONT_FAMILY})
        except Exception:
            pass
        self.canvas.draw()


