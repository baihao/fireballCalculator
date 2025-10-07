#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径图表组件

基于 BaseChart，提供直径曲线的渲染与数据更新接口。
"""

from base_chart import BaseChart
import numpy as np


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
        self._line_color = '#f59e0b'

    def update_data(self, time_ms, diameter_m, K: float = None, B: float = None, C: float = None,
                    cutoff_ms: float = None) -> None:
        """
        更新直径数据并可选绘制拖曳函数与截断线。

        Args:
            time_ms: 时间序列（毫秒）
            diameter_m: 直径序列（米）
            K, B, C: 拖曳函数参数（可选）
            cutoff_ms: 有效数据截断时间（毫秒，可选）
        """
        # 先绘制原始直径曲线
        self.plot_line(
            time_ms,
            diameter_m,
            title=self._title,
            xlabel=self._x_label,
            ylabel=self._y_label,
            color=self._line_color,
        )

        # 补充绘制在同一坐标轴上
        if not self.figure.axes:
            ax = self.figure.add_subplot(111)
        else:
            ax = self.figure.axes[0]

        # 1) 拖曳函数曲线（可选）
        if K is not None and B is not None and C is not None:
            try:
                t_min = float(np.min(time_ms)) if len(time_ms) else 0.0
                t_max = float(np.max(time_ms)) if len(time_ms) else 140.0
                t_smooth = np.linspace(t_min, t_max, 300)
                D_smooth = K * (1.0 - B * np.exp(-C * (t_smooth ** 2)))
                ax.plot(t_smooth, D_smooth, 'b-', linewidth=2, label='拖曳函数拟合')
            except Exception as _:
                pass

        # 2) 截断线（可选）
        if cutoff_ms is not None:
            try:
                cutoff_val = float(cutoff_ms)
                # 在视图范围内再绘制
                x_min, x_max = ax.get_xlim()
                if cutoff_val >= x_min and cutoff_val <= x_max:
                    ax.axvline(x=cutoff_val, color='orange', linestyle='--', linewidth=2,
                               label=f'数据截断点 ({cutoff_val:.1f}ms)')
            except Exception:
                pass

        # 刷新图例与画布
        try:
            ax.legend()
        except Exception:
            pass
        self.canvas.draw()


