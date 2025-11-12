#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径变化速率图表组件

显示：
1) 原始直径数据的变化速率 dD/dt
2) 拖曳函数拟合曲线的变化速率 dD/dt
3) 数据截断点
"""

from .base_chart import BaseChart, FONT_SIZE_BODY, FONT_FAMILY
from typing import Optional, Tuple
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
        self._placeholder_text = "提取完成后显示"

    # --------------------------- 公共API --------------------------- #
    def _compute_axis_limits(self, time_ms, ddt_raw, ddt_fit):
        """
        依据原始/拟合速率计算坐标范围，返回 (xlim, ylim)。
        """
        xlim = self._xlim
        ylim = self._ylim
        if time_ms is None or len(time_ms) == 0:
            return xlim, ylim
        try:
            time_arr = np.array(time_ms, dtype=float)
            valid_mask = np.isfinite(time_arr)
            if not np.any(valid_mask):
                return xlim, ylim
            time_valid = time_arr[valid_mask]
            x_min, x_max = np.min(time_valid), np.max(time_valid)

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
                y_min, y_max = float(np.min(all_y_values)), float(np.max(all_y_values))
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_padding = x_range * AXIS_PADDING_RATIO if x_range > 0 else X_PADDING_DEFAULT
                y_padding = y_range * AXIS_PADDING_RATIO if y_range > 0 else Y_PADDING_DEFAULT
                xlim = (x_min - x_padding, x_max + x_padding)
                ylim = (y_min - y_padding, y_max + y_padding)
                if ylim[0] < 0 and y_min >= 0:
                    ylim = (0, ylim[1])
        except Exception:
            pass
        return xlim, ylim

    def set_placeholder(self, text: str, xy: Optional[Tuple[float, float]] = None) -> None:
        """设置占位符文本与位置，并刷新占位图。"""
        self._placeholder_text = text
        if xy is not None:
            self._placeholder_xy = xy
        self.reset()

    def draw_raw_velocity(self, ax, time_ms, diameter_m) -> Optional[np.ndarray]:
        """绘制原始数据速率，返回 ddt_raw（若可计算）。"""
        if time_ms is None or diameter_m is None:
            return None
        try:
            t = np.array(time_ms, dtype=float)
            d = np.array(diameter_m, dtype=float)
            if len(t) >= 2 and len(d) == len(t):
                ddt_raw = np.gradient(d, t)
                ax.plot(time_ms, ddt_raw, color=self._raw_color, linewidth=LINE_WIDTH, label='原始速率')
                return ddt_raw
        except Exception:
            return None
        return None

    def draw_fit_velocity(self, ax, time_ms, K: float, B: float, C: float) -> Optional[np.ndarray]:
        """绘制拟合速率，返回 ddt_fit（若可计算）。"""
        if time_ms is None or len(time_ms) == 0 or K is None or B is None or C is None:
            return None
        try:
            t_min = float(np.min(time_ms))
            t_max = float(np.max(time_ms))
            t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
            ddt_fit = 2.0 * float(K) * float(B) * float(C) * t_smooth * np.exp(-float(C) * (t_smooth ** 2))
            ax.plot(t_smooth, ddt_fit, '-', color=self._fit_color, linewidth=LINE_WIDTH, label='拟合速率')
            return ddt_fit
        except Exception:
            return None

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

    def update_data(self, time_ms, diameter_m, K: float = None, B: float = None, C: float = None,
                    cutoff_ms: float = None) -> None:
        """
        更新速率数据并可选绘制拟合速率与截断线。
        允许 diameter_m 为空，此时需要提供 K,B,C 绘制拟合速率。

        Args:
            time_ms: 时间序列（毫秒）
            diameter_m: 直径序列（米，可为空）
            K, B, C: 拖曳函数参数（可选）
            cutoff_ms: 有效数据截断时间（毫秒，可选）
        """
        if time_ms is None or len(time_ms) == 0:
            self.reset()
            return

        # 清空并重新绘制
        self.clear()
        ax = self.figure.add_subplot(111)
        
        # 计算数据范围（包括原始速率和拟合速率）
        xlim = self._xlim
        ylim = self._ylim
        
        ddt_raw = None
        ddt_fit = None
        
        # 原始速率（可选）
        ddt_raw = None
        if diameter_m is not None:
            try:
                t = np.array(time_ms, dtype=float)
                d = np.array(diameter_m, dtype=float)
                if len(t) >= 2 and len(d) == len(t):
                    ddt_raw = np.gradient(d, t)
            except Exception:
                ddt_raw = None
        
        # 计算拟合曲线的速率
        if K is not None and B is not None and C is not None and time_ms is not None and len(time_ms) > 0:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                ddt_fit = 2.0 * float(K) * float(B) * float(C) * t_smooth * np.exp(-float(C) * (t_smooth ** 2))
            except Exception:
                pass
        
        # 根据所有数据计算范围
        xlim, ylim = self._compute_axis_limits(time_ms, ddt_raw, ddt_fit)
        
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
        
        # 如果原始数据为空且无拟合参数，则显示占位符并返回
        if (diameter_m is None or len(diameter_m) == 0) and ddt_fit is None:
            self.set_placeholder(self._placeholder_text or "无可绘制数据", self._placeholder_xy)
            return
        
        # 分别绘制
        if ddt_raw is not None:
            ax.plot(time_ms, ddt_raw, color=self._raw_color, linewidth=LINE_WIDTH, label='原始速率')
        if ddt_fit is not None:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                ax.plot(t_smooth, ddt_fit, '-', color=self._fit_color, linewidth=LINE_WIDTH, label='拟合速率')
            except Exception:
                pass

        # 截断线（可选）
        self.draw_cutoff(ax, xlim, cutoff_ms)

        # 刷新图例与画布（使用常量）
        try:
            ax.legend(fontsize=FONT_SIZE_BODY, prop={'family': FONT_FAMILY})
        except Exception:
            pass
        # 使用 constrained_layout，无需 tight_layout
        self.canvas.draw()


