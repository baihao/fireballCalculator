#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径变化速率图表组件

显示：
1) 原始直径数据的变化速率 dD/dt
2) 拖曳函数拟合曲线的变化速率 dD/dt
3) 数据截断点
"""

from base_chart import BaseChart, FONT_SIZE_BODY, FONT_FAMILY, LAYOUT_PAD
import numpy as np

# 线条样式常量
LINE_WIDTH = 2                    # 线条宽度
SMOOTH_POINTS = 300               # 平滑曲线点数
AXIS_PADDING_RATIO = 0.1          # 坐标轴边距比例（10%）
X_PADDING_DEFAULT = 1.0            # x轴默认边距
Y_PADDING_DEFAULT = 0.001         # y轴默认边距

# 颜色常量
COLOR_RAW = '#22c55e'             # 原始速率曲线颜色（绿色）
COLOR_FIT = '#3b82f6'             # 拟合速率曲线颜色（蓝色）
COLOR_CUTOFF = 'orange'           # 截断线颜色


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
        self._raw_color = COLOR_RAW
        self._fit_color = COLOR_FIT

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
        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)
        
        # 计算数据范围（包括原始速率和拟合速率）
        xlim = self._xlim
        ylim = self._ylim
        
        ddt_raw = None
        ddt_fit = None
        
        # 计算原始数据的速率
        if time_ms and diameter_m:
            try:
                t = np.array(time_ms, dtype=float)
                d = np.array(diameter_m, dtype=float)
                if len(t) >= 2 and len(d) == len(t):
                    # numpy.gradient 支持传入坐标，单位：m/ms
                    ddt_raw = np.gradient(d, t)
            except Exception:
                pass
        
        # 计算拟合曲线的速率
        if K is not None and B is not None and C is not None and time_ms and len(time_ms) > 0:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                ddt_fit = 2.0 * float(K) * float(B) * float(C) * t_smooth * np.exp(-float(C) * (t_smooth ** 2))
            except Exception:
                pass
        
        # 根据所有数据计算范围
        if time_ms and len(time_ms) > 0:
            try:
                time_arr = np.array(time_ms, dtype=float)
                valid_mask = np.isfinite(time_arr)
                if np.any(valid_mask):
                    time_valid = time_arr[valid_mask]
                    x_min, x_max = np.min(time_valid), np.max(time_valid)
                    
                    # 收集所有 y 值
                    all_y_values = []
                    if ddt_raw is not None:
                        valid_raw = ddt_raw[np.isfinite(ddt_raw)]
                        if len(valid_raw) > 0:
                            all_y_values.extend(valid_raw)
                    if ddt_fit is not None:
                        valid_fit = ddt_fit[np.isfinite(ddt_fit)]
                        if len(valid_fit) > 0:
                            all_y_values.extend(valid_fit)
                    
                    if len(all_y_values) > 0:
                        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
                        
                        # 添加边距
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        x_padding = x_range * AXIS_PADDING_RATIO if x_range > 0 else X_PADDING_DEFAULT
                        y_padding = y_range * AXIS_PADDING_RATIO if y_range > 0 else Y_PADDING_DEFAULT
                        
                        xlim = (x_min - x_padding, x_max + x_padding)
                        ylim = (y_min - y_padding, y_max + y_padding)
                        
                        # 确保 y 轴最小值不为负（速率可以为负，但通常显示从0开始更合理）
                        if ylim[0] < 0 and y_min >= 0:
                            ylim = (0, ylim[1])
            except Exception:
                pass
        
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
        
        # 绘制原始数据的速率
        if ddt_raw is not None and time_ms:
            ax.plot(time_ms, ddt_raw, color=self._raw_color, linewidth=LINE_WIDTH, label='原始速率')

        # 绘制拟合曲线的速率
        if ddt_fit is not None:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                ax.plot(t_smooth, ddt_fit, '-', color=self._fit_color, linewidth=LINE_WIDTH, label='拟合速率')
            except Exception:
                pass

        # 截断线（可选）
        if cutoff_ms is not None:
            try:
                cutoff_val = float(cutoff_ms)
                if xlim and cutoff_val >= xlim[0] and cutoff_val <= xlim[1]:
                    ax.axvline(x=cutoff_val, color=COLOR_CUTOFF, linestyle='--', linewidth=LINE_WIDTH,
                               label=f'数据截断点 ({cutoff_val:.1f}ms)')
            except Exception:
                pass

        # 刷新图例与画布（使用常量）
        try:
            ax.legend(fontsize=FONT_SIZE_BODY, prop={'family': FONT_FAMILY})
        except Exception:
            pass
        self.figure.tight_layout(pad=LAYOUT_PAD)
        self.canvas.draw()


