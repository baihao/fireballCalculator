#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热通量图表组件

基于 BaseChart，提供多条热通量曲线的渲染与数据更新接口。
显示不同距离下的热通量随时间变化。
"""

from .base_chart import BaseChart, FONT_SIZE_BODY, FONT_FAMILY
from typing import Optional, Tuple, List
import numpy as np
from matplotlib.lines import Line2D

# 线条样式常量
LINE_WIDTH = 2                    # 线条宽度
AXIS_PADDING_RATIO = 0.1          # 坐标轴边距比例（10%）
X_PADDING_DEFAULT = 1.0           # x轴默认边距
Y_PADDING_DEFAULT = 1000.0        # y轴默认边距（热通量数值较大）

# 颜色常量（用于不同距离的曲线）
COLORS = ['#22d3ee', '#38bdf8', '#10b981', '#f59e0b', '#ef4444', '#a855f7', '#ec4899']


class HeatFluxChart(BaseChart):
    """热通量图表（多条曲线，不同距离）。"""

    def __init__(self, width: float = 4, height: float = 2.5, dpi: int = 100):
        super().__init__(
            x_label="时间 (ms)",
            y_label="热通量 (W/m²)",
            title="热通量随时间变化 (不同距离)",
            xlim=(0, 140),
            ylim=(0, 120000),
            placeholder_text="等待预测...",
            placeholder_xy=(70, 60000),
            width=width,
            height=height,
            dpi=dpi,
        )
        self._placeholder_text = "等待预测..."

    # --------------------------- 公共API --------------------------- #
    def _compute_axis_limits(self, time_ms, heat_flux_data: List):
        """
        依据时间序列和所有热通量数据计算坐标范围。
        返回 (xlim, ylim)
        """
        xlim = self._xlim
        ylim = self._ylim
        if time_ms is None or len(time_ms) == 0:
            return xlim, ylim
        try:
            time_arr = np.array(time_ms, dtype=float)
            x_min, x_max = float(np.min(time_arr)), float(np.max(time_arr))

            # 收集所有热通量数据的最小值和最大值
            y_candidates = []
            if heat_flux_data:
                for item in heat_flux_data:
                    if len(item) >= 2:
                        distance = item[0]
                        heat_flux_array = item[1]
                        if heat_flux_array is not None and len(heat_flux_array) > 0:
                            flux_arr = np.array(heat_flux_array, dtype=float)
                            valid_mask = np.isfinite(flux_arr)
                            if np.any(valid_mask):
                                y_candidates.append(float(np.min(flux_arr[valid_mask])))
                                y_candidates.append(float(np.max(flux_arr[valid_mask])))

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

    def update_data(self, time_ms, heat_flux_data: List = None) -> None:
        """
        更新热通量图。绘制多条曲线，每条对应一个距离。

        Args:
            time_ms: 时间序列（毫秒）
            heat_flux_data: 热通量数据列表，每个元素为 [distance(float), heat_flux(array)]
                          例如：[[4.0, [q1, q2, ...]], [4.5, [q1, q2, ...]], ...]
        """
        # 基础有效性
        if time_ms is None or len(time_ms) == 0:
            self.reset()
            return

        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)

        # 计算数据范围
        xlim, ylim = self._compute_axis_limits(time_ms, heat_flux_data or [])

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

        # 如果没有数据，显示占位符
        if heat_flux_data is None or len(heat_flux_data) == 0:
            self.set_placeholder(self._placeholder_text or "无可绘制数据", self._placeholder_xy)
            return

        legend_handles: List[Line2D] = []
        legend_labels: List[str] = []

        # 绘制多条热通量曲线
        for i, item in enumerate(heat_flux_data):
            if len(item) < 2:
                continue
            distance = item[0]
            heat_flux_array = item[1]
            
            if heat_flux_array is None or len(heat_flux_array) == 0:
                continue

            # 确保时间序列和热通量数组长度一致
            if len(time_ms) != len(heat_flux_array):
                continue

            # 选择颜色（循环使用）
            color = COLORS[i % len(COLORS)]
            
            # 绘制曲线
            ax.plot(time_ms, heat_flux_array, color=color, linewidth=LINE_WIDTH)

            legend_handles.append(Line2D([0], [0], color=color, linewidth=LINE_WIDTH))
            legend_labels.append(f'x = {distance:.1f} m')

        # 刷新图例与画布
        if legend_handles:
            try:
                legend = ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc='upper right',
                    bbox_to_anchor=(0.98, 0.98),
                    frameon=True,
                    borderaxespad=0.4,
                    labelspacing=0.4,
                    handlelength=1.8,
                    handletextpad=0.6,
                )
                legend.get_frame().set_facecolor('#1f2937')
                legend.get_frame().set_edgecolor('#374151')
                legend.get_frame().set_alpha(0.85)
                for text in legend.get_texts():
                    text.set_color('white')
                    text.set_fontsize(FONT_SIZE_BODY)
                    text.set_fontfamily(FONT_FAMILY)
            except Exception:
                pass
        self.canvas.draw()

