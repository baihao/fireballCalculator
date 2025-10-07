#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
温度图表组件

基于 BaseChart，提供温度曲线的渲染与数据更新接口。
"""

from base_chart import BaseChart


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
        self._line_color = '#38bdf8'

    def update_data(self, time_ms, temperature_k) -> None:
        self.plot_line(
            time_ms,
            temperature_k,
            title=self._title,
            xlabel=self._x_label,
            ylabel=self._y_label,
            color=self._line_color,
        )


