#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直径变化速率图表组件

显示：
1) 直径滑动平滑后的变化速率 dD/dt（图例：原始速率）
2) 拖曳函数拟合曲线的变化速率 dD/dt
3) 数据截断点
"""

from .base_chart import BaseChart, FONT_SIZE_BODY, chart_font_family
from typing import Optional, Tuple
import numpy as np

# 线条样式常量
LINE_WIDTH = 2                    # 线条宽度
LINE_WIDTH_FIT = 2.5              # 拟合速率略加粗，避免被原始噪声淹没
SMOOTH_POINTS = 300               # 平滑曲线点数
AXIS_PADDING_RATIO = 0.1          # 坐标轴边距比例（10%）
X_PADDING_DEFAULT = 1.0            # x轴默认边距
Y_PADDING_DEFAULT = 0.001         # y轴默认边距
# 平滑速率纵轴：分位数稳健范围，避免残尖峰撑开刻度
RAW_Y_PERCENTILE_LOW = 5.0
RAW_Y_PERCENTILE_HIGH = 95.0
# 有拟合时：平滑速率稳健范围最多扩到拟合幅度的该倍数
RAW_VS_FIT_Y_EXPAND = 2.5
# 计算「平滑速率」前对直径做滑动平均的窗口（点数）
DIAMETER_SMOOTH_WINDOW = 11

# 颜色常量
COLOR_RAW = '#22c55e'             # 原始速率曲线颜色（绿色）
COLOR_FIT = '#3b82f6'             # 拟合速率曲线颜色（蓝色）
COLOR_CUTOFF = 'orange'           # 截断线颜色


class DiameterVelocityChart(BaseChart):
    """直径变化速率图表。"""

    def __init__(
        self,
        width: float = 4,
        height: float = 2.5,
        dpi: int = 100,
        *,
        title: str = "火球直径变化速率随时间变化",
        y_label: str = "直径变化速率 (m/ms)",
        raw_label: str = "原始速率",
    ):
        super().__init__(
            x_label="时间 (ms)",
            y_label=y_label,
            title=title,
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
        self._raw_label = raw_label
        self._placeholder_text = "提取完成后显示"

    # --------------------------- 公共API --------------------------- #
    @staticmethod
    def _smooth_diameter(diameter_m: np.ndarray, window: int = DIAMETER_SMOOTH_WINDOW) -> np.ndarray:
        """对直径序列做滑动平均，再用于数值求导。"""
        d = np.asarray(diameter_m, dtype=float)
        n = d.size
        if n == 0:
            return d.copy()
        w = int(window)
        if w < 3 or n < w:
            return d.copy()
        if w % 2 == 0:
            w += 1
        half = w // 2
        out = np.empty(n, dtype=float)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            seg = d[lo:hi]
            seg = seg[np.isfinite(seg)]
            out[i] = float(np.mean(seg)) if seg.size else d[i]
        return out

    @staticmethod
    def _rate_from_diameter(
        time_ms: np.ndarray, diameter_m: np.ndarray, *, smooth: bool = True
    ) -> Optional[np.ndarray]:
        """由直径求 dD/dt；默认先滑动平滑直径。"""
        t = np.asarray(time_ms, dtype=float)
        d = np.asarray(diameter_m, dtype=float)
        if t.size < 2 or d.size != t.size:
            return None
        if smooth:
            d = DiameterVelocityChart._smooth_diameter(d)
        return np.gradient(d, t)

    @staticmethod
    def _robust_y_span(values: np.ndarray) -> Optional[Tuple[float, float]]:
        """用分位数得到稳健 y 范围；样本过少则退回 min/max。"""
        valid = np.asarray(values, dtype=float)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return None
        if valid.size < 8:
            return float(np.min(valid)), float(np.max(valid))
        lo, hi = np.percentile(
            valid, [RAW_Y_PERCENTILE_LOW, RAW_Y_PERCENTILE_HIGH]
        )
        return float(lo), float(hi)

    def _compute_axis_limits(self, time_ms, ddt_raw, ddt_fit):
        """
        计算坐标范围。有拟合时以拟合速率为主，平滑速率用稳健范围辅助。
        纵轴不强制从 0 起。
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
            x_range = x_max - x_min
            x_padding = x_range * AXIS_PADDING_RATIO if x_range > 0 else X_PADDING_DEFAULT
            xlim = (x_min - x_padding, x_max + x_padding)

            fit_span = None
            if ddt_fit is not None:
                valid_fit = np.asarray(ddt_fit, dtype=float)
                valid_fit = valid_fit[np.isfinite(valid_fit)]
                if valid_fit.size > 0:
                    fit_span = (float(np.min(valid_fit)), float(np.max(valid_fit)))

            raw_span = None
            if ddt_raw is not None:
                raw_span = self._robust_y_span(ddt_raw)

            if fit_span is not None:
                y_min, y_max = fit_span
                fit_amp = max(abs(y_min), abs(y_max), Y_PADDING_DEFAULT)
                if raw_span is not None:
                    cap = RAW_VS_FIT_Y_EXPAND * fit_amp
                    y_min = min(y_min, max(raw_span[0], -cap))
                    y_max = max(y_max, min(raw_span[1], cap))
            elif raw_span is not None:
                y_min, y_max = raw_span
            else:
                return xlim, ylim

            if y_max < y_min:
                y_min, y_max = y_max, y_min
            y_range = y_max - y_min
            y_padding = y_range * AXIS_PADDING_RATIO if y_range > 0 else Y_PADDING_DEFAULT
            ylim = (y_min - y_padding, y_max + y_padding)
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
        """绘制平滑直径后的速率，返回 ddt（若可计算）。"""
        if time_ms is None or diameter_m is None:
            return None
        try:
            ddt = self._rate_from_diameter(
                np.asarray(time_ms, dtype=float),
                np.asarray(diameter_m, dtype=float),
                smooth=True,
            )
            if ddt is not None:
                ax.plot(
                    time_ms,
                    ddt,
                    color=self._raw_color,
                    linewidth=LINE_WIDTH,
                    label=self._raw_label,
                )
                return ddt
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
        
        # 平滑直径后再求导（可选）
        ddt_raw = None
        if diameter_m is not None:
            try:
                ddt_raw = self._rate_from_diameter(
                    np.asarray(time_ms, dtype=float),
                    np.asarray(diameter_m, dtype=float),
                    smooth=True,
                )
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
        
        # 分别绘制：原始半透明置底，拟合加粗置顶，便于辨认
        if ddt_raw is not None:
            ax.plot(
                time_ms,
                ddt_raw,
                color=self._raw_color,
                linewidth=LINE_WIDTH,
                alpha=0.45,
                zorder=2,
                label=self._raw_label,
            )
        if ddt_fit is not None:
            try:
                t_min = float(np.min(time_ms))
                t_max = float(np.max(time_ms))
                t_smooth = np.linspace(t_min, t_max, SMOOTH_POINTS)
                ax.plot(
                    t_smooth,
                    ddt_fit,
                    '-',
                    color=self._fit_color,
                    linewidth=LINE_WIDTH_FIT,
                    zorder=3,
                    label='拟合速率',
                )
            except Exception:
                pass

        # 截断线（可选）
        self.draw_cutoff(ax, xlim, cutoff_ms)

        # 刷新图例与画布（使用常量）
        try:
            from .base_chart import chart_font_properties

            ax.legend(fontsize=FONT_SIZE_BODY, prop=chart_font_properties(FONT_SIZE_BODY))
        except Exception:
            pass
        # 使用 constrained_layout，无需 tight_layout
        self.canvas.draw()


