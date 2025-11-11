#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
温度图表组件

基于 BaseChart，提供温度曲线的渲染与数据更新接口。
"""

from base_chart import BaseChart
import numpy as np

# 线条样式常量
LINE_WIDTH = 2                    # 线条宽度
AXIS_PADDING_RATIO = 0.1          # 坐标轴边距比例（10%）
X_PADDING_DEFAULT = 1.0           # x轴默认边距
Y_PADDING_DEFAULT = 10.0          # y轴默认边距

# 颜色常量
COLOR_TEMPERATURE = '#38bdf8'     # 温度曲线颜色


class TemperatureChart(BaseChart):
    """温度图表。"""

    def __init__(self, width: float = 4, height: float = 2.5, dpi: int = 100):
        super().__init__(
            x_label="时间 (ms)",
            y_label="温度 (K)",
            title="火球温度随时间变化",
            xlim=(0, 140),
            ylim=(1000, 1600),
            placeholder_text="请加载序列文件",
            placeholder_xy=(70, 1300),
            width=width,
            height=height,
            dpi=dpi,
        )
        self._line_color = COLOR_TEMPERATURE

    def update_data(self, time_ms, temperature_k) -> None:
        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)
        
        # 计算数据范围
        if time_ms and temperature_k and len(time_ms) > 0 and len(temperature_k) > 0:
            try:
                time_arr = np.array(time_ms, dtype=float)
                temp_arr = np.array(temperature_k, dtype=float)
                
                # 过滤掉无效值
                valid_mask = np.isfinite(time_arr) & np.isfinite(temp_arr)
                if np.any(valid_mask):
                    time_valid = time_arr[valid_mask]
                    temp_valid = temp_arr[valid_mask]
                    
                    x_min, x_max = np.min(time_valid), np.max(time_valid)
                    y_min, y_max = np.min(temp_valid), np.max(temp_valid)
                    
                    # 添加边距
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    x_padding = x_range * AXIS_PADDING_RATIO if x_range > 0 else X_PADDING_DEFAULT
                    y_padding = y_range * AXIS_PADDING_RATIO if y_range > 0 else Y_PADDING_DEFAULT
                    
                    xlim = (x_min - x_padding, x_max + x_padding)
                    ylim = (y_min - y_padding, y_max + y_padding)
                    
                    # 确保 y 轴最小值不为负（温度不能为负）
                    if ylim[0] < 0:
                        ylim = (0, ylim[1])
                else:
                    xlim = self._xlim
                    ylim = self._ylim
            except Exception:
                xlim = self._xlim
                ylim = self._ylim
        else:
            xlim = self._xlim
            ylim = self._ylim
        
        # 应用统一的暗色主题样式
        from base_chart import apply_dark_chart_style
        apply_dark_chart_style(
            ax,
            x_label=self._x_label,
            y_label=self._y_label,
            title=self._title,
            xlim=xlim,
            ylim=ylim,
        )
        
        # 绘制温度曲线
        if time_ms and temperature_k:
            ax.plot(time_ms, temperature_k, color=self._line_color, linewidth=LINE_WIDTH)
        # 使用 constrained_layout，无需 tight_layout
        self.canvas.draw()


