#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
累积热辐射量图表组件

基于 BaseChart，提供累积热辐射量随距离变化的渲染与数据更新接口。
"""

from .base_chart import BaseChart, FONT_SIZE_BODY, FONT_FAMILY
from typing import Optional, Tuple
import numpy as np

# 线条样式常量
LINE_WIDTH = 2                    # 线条宽度
AXIS_PADDING_RATIO = 0.1          # 坐标轴边距比例（10%）
X_PADDING_DEFAULT = 0.05          # x轴默认边距（距离）
Y_PADDING_DEFAULT = 200.0         # y轴默认边距（热辐射量）

# 颜色常量
COLOR_RADIATION = '#10b981'       # 累积热辐射量曲线颜色（绿色）


class RadiationChart(BaseChart):
    """累积热辐射量图表。"""

    def __init__(self, width: float = 4, height: float = 2.5, dpi: int = 100):
        super().__init__(
            x_label="距离 (m)",
            y_label="热辐射量 (J/m²)",
            title="累积热辐射量随距离分布",
            xlim=(4.0, 6.0),
            ylim=(0, 10000),
            placeholder_text="等待预测...",
            placeholder_xy=(5.0, 5000),
            width=width,
            height=height,
            dpi=dpi,
        )
        self._line_color = COLOR_RADIATION
        self._placeholder_text = "等待预测..."

    # --------------------------- 公共API --------------------------- #
    def _compute_axis_limits(self, distances, heat_radiation):
        """
        依据距离和热辐射量数据计算坐标范围。
        返回 (xlim, ylim)
        """
        xlim = self._xlim
        ylim = self._ylim
        if distances is None or len(distances) == 0:
            return xlim, ylim
        try:
            dist_arr = np.array(distances, dtype=float)
            valid_mask = np.isfinite(dist_arr)
            if not np.any(valid_mask):
                return xlim, ylim
            dist_valid = dist_arr[valid_mask]
            x_min, x_max = float(np.min(dist_valid)), float(np.max(dist_valid))

            y_candidates = []
            if heat_radiation is not None and len(heat_radiation) > 0:
                rad_arr = np.array(heat_radiation, dtype=float)
                valid_rad_mask = np.isfinite(rad_arr)
                if np.any(valid_rad_mask):
                    rad_valid = rad_arr[valid_rad_mask]
                    y_candidates.append(float(np.min(rad_valid)))
                    y_candidates.append(float(np.max(rad_valid)))

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

    def update_data(self, distances, heat_radiation) -> None:
        """
        更新累积热辐射量图。

        Args:
            distances: 距离数组（米）
            heat_radiation: 累积热辐射量数组（J/m²）
        """
        # 基础有效性
        if distances is None or len(distances) == 0:
            self.reset()
            return

        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)

        # 计算数据范围
        xlim, ylim = self._compute_axis_limits(distances, heat_radiation)

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
        if heat_radiation is None or len(heat_radiation) == 0:
            self.set_placeholder(self._placeholder_text or "无可绘制数据", self._placeholder_xy)
            return

        # 确保距离和热辐射量数组长度一致
        if len(distances) != len(heat_radiation):
            self.set_placeholder("数据长度不匹配", self._placeholder_xy)
            return

        # 绘制累积热辐射量曲线
        ax.plot(distances, heat_radiation, color=self._line_color, linewidth=LINE_WIDTH)

        # 刷新画布
        self.canvas.draw()

