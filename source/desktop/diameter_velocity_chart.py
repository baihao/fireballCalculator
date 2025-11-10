#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径变化速率图表组件

显示：
1) 原始直径数据的变化速率 dD/dt
2) 拖曳函数拟合曲线的变化速率 dD/dt
3) 数据截断点
"""

from base_chart import BaseChart
import numpy as np


class DiameterVelocityChart(BaseChart):
    """直径变化速率图表。"""

    def __init__(self, width: float = 4, height: float = 2.5, dpi: int = 100):
        super().__init__(
            x_label="时间 (ms)",
            y_label="直径变化速率 (m/ms)",
            title="火球直径变化速率随时间变化",
            xlim=(0, 140),
            ylim=(0, 0.05),
            placeholder_text="提取完成后显示",
            placeholder_xy=(70, 0.02),
            width=width,
            height=height,
            dpi=dpi,
        )
        self._raw_color = '#22c55e'   # 原始速率曲线（绿色）
        self._fit_color = '#3b82f6'   # 拟合速率曲线（蓝色）

    def update_data(self, time_ms, diameter_m, K: float = None, B: float = None, C: float = None,
                    cutoff_ms: float = None) -> None:
        """
        更新速率数据并可选绘制拟合速率与截断线。

        Args:
            time_ms: 时间序列（毫秒）
            diameter_m: 直径序列（米）
            K, B, C: 拖曳函数参数（可选）
            cutoff_ms: 有效数据截断时间（毫秒，可选）
        """
        # 计算并绘制原始数据的速率（使用 numpy.gradient，非均匀时间间隔也可）
        try:
            t = np.array(time_ms, dtype=float)
            d = np.array(diameter_m, dtype=float)
            if len(t) >= 2 and len(d) == len(t):
                # numpy.gradient 支持传入坐标，单位：m/ms
                ddt_raw = np.gradient(d, t)
                self.plot_line(
                    t,
                    ddt_raw,
                    title=self._title,
                    xlabel=self._x_label,
                    ylabel=self._y_label,
                    color=self._raw_color,
                    label='原始速率'
                )
            else:
                # 没有足够数据则仅清空/占位
                self.plot_line([], [], title=self._title, xlabel=self._x_label, ylabel=self._y_label,
                               color=self._raw_color, label='原始速率')
        except Exception:
            # 出错时尽量不影响后续绘制
            pass

        # 获取坐标轴
        if not self.figure.axes:
            ax = self.figure.add_subplot(111)
        else:
            ax = self.figure.axes[0]

        # 计算并绘制拟合曲线的速率：
        # D(t) = K * (1 - B * exp(-C * t^2))
        # dD/dt = 2 * K * B * C * t * exp(-C * t^2)
        if K is not None and B is not None and C is not None and len(time_ms) > 0:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, 300)
                ddt_fit = 2.0 * float(K) * float(B) * float(C) * t_smooth * np.exp(-float(C) * (t_smooth ** 2))
                ax.plot(t_smooth, ddt_fit, '-', color=self._fit_color, linewidth=2, label='拟合速率')
            except Exception:
                pass

        # 截断线（可选）
        if cutoff_ms is not None:
            try:
                cutoff_val = float(cutoff_ms)
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


