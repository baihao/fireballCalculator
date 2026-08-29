#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChartController

负责管理特征提取模块中的直径图与速率图：
- 提供统一的重置、更新接口
- 缓存最近一次绘制的数据及拖曳拟合结果
- 对输入长度与拟合参数进行基础校验
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from chart_widgets.diameter_chart import DiameterChart
from chart_widgets.diameter_velocity_chart import DiameterVelocityChart


class ChartController:
    """直径图与速率图的统一控制器。"""

    def __init__(self, ui_builder) -> None:
        ui_components = ui_builder.get_ui_components()
        self._diam_chart: Optional[DiameterChart] = ui_components.get('diam_chart')
        self._diam_vel_chart: Optional[DiameterVelocityChart] = ui_components.get('diam_vel_chart')

        self._last_diameter: Optional[List[Tuple[float, float]]] = None
        self._last_drag_fit: Optional[Dict[str, Any]] = None

        self.reset()

    def set_widgets(
        self,
        diam_chart: DiameterChart,
        diam_vel_chart: DiameterVelocityChart,
    ) -> None:
        """注入直径与直径速率图组件（向后兼容，已废弃）。"""
        self._diam_chart = diam_chart
        self._diam_vel_chart = diam_vel_chart
        self.reset()

    def reset(self) -> None:
        """重置图表显示与缓存。"""
        if self._diam_chart:
            self._diam_chart.reset()
        if self._diam_vel_chart:
            self._diam_vel_chart.reset()
        self._last_diameter = None
        self._last_drag_fit = None

    @staticmethod
    def _prepare_drag_fit(fit_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not fit_result:
            return None
        data_filtering = fit_result.get('data_filtering') or {}
        return {
            'success': bool(fit_result.get('success', False)),
            'K': fit_result.get('K'),
            'B': fit_result.get('B'),
            'C': fit_result.get('C'),
            'expression': fit_result.get(
                'expression',
                'D(t) = K * (1 - B * exp(-C * t^2))',
            ),
            'r_squared': fit_result.get('r_squared'),
            'rmse': fit_result.get('rmse'),
            'mae': fit_result.get('mae'),
            'max_relative_error': fit_result.get('max_relative_error'),
            'data_filtering': data_filtering,
            'error': fit_result.get('error'),
        }

    def update_diameter_raw(
        self,
        time_ms: Sequence[float],
        diameter_m: Sequence[float],
    ) -> None:
        """仅绘制直径原始数据。"""
        self._ensure_chart_initialized()
        self._validate_length(time_ms, diameter_m, "直径数据")
        self._diam_chart.update_data(time_ms, diameter_m)  # type: ignore[arg-type]
        if self._diam_vel_chart:
            self._diam_vel_chart.update_data(time_ms, diameter_m)
        self._last_diameter = list(zip(map(float, time_ms), map(float, diameter_m)))
        self._last_drag_fit = None

    def update_diameter_with_fit(
        self,
        time_ms: Sequence[float],
        diameter_m: Sequence[float],
        fit_result: Optional[Dict[str, Any]],
    ) -> None:
        """绘制直径数据并叠加拖曳拟合结果。"""
        self._ensure_chart_initialized()
        self._validate_length(time_ms, diameter_m, "直径数据")

        prepared_fit = self._prepare_drag_fit(fit_result)
        K = B = C = None
        cutoff_ms = None
        if prepared_fit:
            K = prepared_fit.get('K')
            B = prepared_fit.get('B')
            C = prepared_fit.get('C')
            cutoff_ms = (prepared_fit.get('data_filtering') or {}).get('cutoff_time')

        self._diam_chart.update_data(  # type: ignore[arg-type]
            time_ms,
            diameter_m,
            K=K,
            B=B,
            C=C,
            cutoff_ms=cutoff_ms,
        )
        if self._diam_vel_chart:
            self._diam_vel_chart.update_data(  # type: ignore[arg-type]
                time_ms,
                diameter_m,
                K=K,
                B=B,
                C=C,
                cutoff_ms=cutoff_ms,
            )
        self._last_diameter = list(zip(map(float, time_ms), map(float, diameter_m)))
        self._last_drag_fit = prepared_fit

    def clear_diameter(self) -> None:
        """清空直径图显示与缓存。"""
        if self._diam_chart:
            self._diam_chart.reset()
        if self._diam_vel_chart:
            self._diam_vel_chart.reset()
        self._last_diameter = None
        self._last_drag_fit = None

    def has_analysis_results(self) -> bool:
        """是否同时具备直径曲线与拖曳拟合参数。"""
        has_curve = bool(self._last_diameter)
        has_fit = isinstance(self._last_drag_fit, dict) and \
            self._last_drag_fit.get('K') is not None
        return has_curve and has_fit

    def get_cached_diameter(self) -> Optional[List[Tuple[float, float]]]:
        return self._last_diameter

    def get_cached_drag_fit(self) -> Optional[Dict[str, Any]]:
        return self._last_drag_fit

    def _ensure_chart_initialized(self) -> None:
        if not (self._diam_chart and self._diam_vel_chart):
            raise RuntimeError(
                "ChartController 的图表组件未正确初始化，请检查 ui_builder 是否包含所需的图表组件"
            )

    @staticmethod
    def _validate_length(x: Sequence[Any], y: Sequence[Any], label: str) -> None:
        if len(x) != len(y):
            raise ValueError(f"{label}的时间与数值序列长度不一致: {len(x)} vs {len(y)}")
