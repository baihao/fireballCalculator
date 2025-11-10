#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径图表组件

基于 BaseChart，提供直径曲线的渲染与数据更新接口。
"""

from base_chart import BaseChart, FONT_SIZE_BODY, FONT_FAMILY, LAYOUT_PAD
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
        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)
        
        # 计算数据范围（包括原始数据和拟合曲线）
        xlim = self._xlim
        ylim = self._ylim
        
        if time_ms and diameter_m and len(time_ms) > 0 and len(diameter_m) > 0:
            try:
                time_arr = np.array(time_ms, dtype=float)
                diam_arr = np.array(diameter_m, dtype=float)
                
                # 过滤掉无效值
                valid_mask = np.isfinite(time_arr) & np.isfinite(diam_arr)
                if np.any(valid_mask):
                    time_valid = time_arr[valid_mask]
                    diam_valid = diam_arr[valid_mask]
                    
                    x_min, x_max = np.min(time_valid), np.max(time_valid)
                    y_min, y_max = np.min(diam_valid), np.max(diam_valid)
                    
                    # 如果有拟合曲线，也要考虑拟合曲线的范围
                    if K is not None and B is not None and C is not None:
                        try:
                            t_smooth = np.linspace(x_min, x_max, SMOOTH_POINTS)
                            D_smooth = K * (1.0 - B * np.exp(-C * (t_smooth ** 2)))
                            y_min = min(y_min, np.min(D_smooth))
                            y_max = max(y_max, np.max(D_smooth))
                        except Exception:
                            pass
                    
                    # 添加边距
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    x_padding = x_range * AXIS_PADDING_RATIO if x_range > 0 else X_PADDING_DEFAULT
                    y_padding = y_range * AXIS_PADDING_RATIO if y_range > 0 else Y_PADDING_DEFAULT
                    
                    xlim = (x_min - x_padding, x_max + x_padding)
                    ylim = (max(0, y_min - y_padding), y_max + y_padding)  # 直径不能为负
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
        
        # 绘制原始直径曲线
        if time_ms and diameter_m:
            ax.plot(time_ms, diameter_m, color=self._line_color, linewidth=LINE_WIDTH, label='原始直径')

        # 1) 拖曳函数曲线（可选）
        if K is not None and B is not None and C is not None and time_ms and len(time_ms) > 0:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                D_smooth = K * (1.0 - B * np.exp(-C * (t_smooth ** 2)))
                ax.plot(t_smooth, D_smooth, f'{COLOR_FIT}-', linewidth=LINE_WIDTH, label='拖曳函数拟合')
            except Exception as _:
                pass

        # 2) 截断线（可选）
        if cutoff_ms is not None:
            try:
                cutoff_val = float(cutoff_ms)
                # 在视图范围内再绘制
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


