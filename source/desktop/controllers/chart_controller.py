#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChartController

负责管理特征提取模块中的温度图与直径图：
- 提供统一的重置、更新接口
- 缓存最近一次绘制的数据及拖曳拟合结果
- 对输入长度与拟合参数进行基础校验
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

from temperature_chart import TemperatureChart
from diameter_chart import DiameterChart


class ChartController:
    """温度图与直径图的统一控制器。"""

    def __init__(self) -> None:
        self._temp_chart: Optional[TemperatureChart] = None
        self._diam_chart: Optional[DiameterChart] = None

        self._last_temperature: Optional[List[Tuple[float, float]]] = None
        self._last_diameter: Optional[List[Tuple[float, float]]] = None
        self._last_drag_fit: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # 依赖注入与重置
    # ------------------------------------------------------------------ #
    def set_widgets(self,
                    temp_chart: TemperatureChart,
                    diam_chart: DiameterChart) -> None:
        """注入温度与直径图组件。"""
        self._temp_chart = temp_chart
        self._diam_chart = diam_chart
        self.reset()

    def reset(self) -> None:
        """重置图表显示与缓存。"""
        if self._temp_chart:
            self._temp_chart.reset()
        if self._diam_chart:
            self._diam_chart.reset()
        self._last_temperature = None
        self._last_diameter = None
        self._last_drag_fit = None

    # ------------------------------------------------------------------ #
    # 温度图相关
    # ------------------------------------------------------------------ #
    def update_temperature(self,
                           time_ms: Sequence[float],
                           temperature_k: Sequence[float]) -> None:
        """更新温度曲线。"""
        self._ensure_chart_initialized()
        self._validate_length(time_ms, temperature_k, "温度数据")
        self._temp_chart.update_data(time_ms, temperature_k)  # type: ignore[arg-type]
        self._last_temperature = list(zip(map(float, time_ms),
                                          map(float, temperature_k)))

    # ------------------------------------------------------------------ #
    # 直径图相关
    # ------------------------------------------------------------------ #
    def update_diameter_raw(self,
                            time_ms: Sequence[float],
                            diameter_m: Sequence[float]) -> None:
        """仅绘制直径原始数据。"""
        self._ensure_chart_initialized()
        self._validate_length(time_ms, diameter_m, "直径数据")
        self._diam_chart.update_data(time_ms, diameter_m)  # type: ignore[arg-type]
        self._last_diameter = list(zip(map(float, time_ms),
                                       map(float, diameter_m)))
        self._last_drag_fit = None

    def update_diameter_with_fit(self,
                                 time_ms: Sequence[float],
                                 diameter_m: Sequence[float],
                                 fit_result: Optional[Dict[str, Any]]) -> None:
        """
        绘制直径数据并叠加拖曳拟合结果。

        Args:
            time_ms: 时间序列（毫秒）
            diameter_m: 直径序列（米）
            fit_result: 拖曳拟合结果字典，可包含以下字段：
                - success: bool
                - K, B, C: float
                - data_filtering.cutoff_time: float
                - expression: str
        """
        self._ensure_chart_initialized()
        self._validate_length(time_ms, diameter_m, "直径数据")

        K = B = C = None
        cutoff_ms = None
        prepared_fit: Optional[Dict[str, Any]] = None

        if fit_result:
            K = fit_result.get('K')
            B = fit_result.get('B')
            C = fit_result.get('C')
            data_filtering = fit_result.get('data_filtering') or {}
            cutoff_ms = data_filtering.get('cutoff_time')

            prepared_fit = {
                'success': bool(fit_result.get('success', False)),
                'K': K,
                'B': B,
                'C': C,
                'expression': fit_result.get(
                    'expression',
                    'D(t) = K * (1 - B * exp(-C * t^2))'
                ),
                'data_filtering': data_filtering,
            }

        self._diam_chart.update_data(  # type: ignore[arg-type]
            time_ms,
            diameter_m,
            K=K,
            B=B,
            C=C,
            cutoff_ms=cutoff_ms,
        )
        self._last_diameter = list(zip(map(float, time_ms),
                                       map(float, diameter_m)))
        self._last_drag_fit = prepared_fit

    def clear_diameter(self) -> None:
        """清空直径图显示与缓存。"""
        if self._diam_chart:
            self._diam_chart.reset()
        self._last_diameter = None
        self._last_drag_fit = None

    # ------------------------------------------------------------------ #
    # 状态查询
    # ------------------------------------------------------------------ #
    def has_analysis_results(self) -> bool:
        """是否同时具备直径曲线与拖曳拟合参数。"""
        has_curve = bool(self._last_diameter)
        has_fit = isinstance(self._last_drag_fit, dict) and \
            self._last_drag_fit.get('K') is not None
        return has_curve and has_fit

    def get_cached_temperature(self) -> Optional[List[Tuple[float, float]]]:
        return self._last_temperature

    def get_cached_diameter(self) -> Optional[List[Tuple[float, float]]]:
        return self._last_diameter

    def get_cached_drag_fit(self) -> Optional[Dict[str, Any]]:
        return self._last_drag_fit

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _ensure_chart_initialized(self) -> None:
        if not (self._temp_chart and self._diam_chart):
            raise RuntimeError("ChartController 尚未通过 set_widgets 初始化图表组件")

    @staticmethod
    def _validate_length(x: Sequence[Any],
                         y: Sequence[Any],
                         label: str) -> None:
        if len(x) != len(y):
            raise ValueError(f"{label}的时间与数值序列长度不一致: {len(x)} vs {len(y)}")


