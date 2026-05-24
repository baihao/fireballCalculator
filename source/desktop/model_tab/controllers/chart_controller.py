#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelTabChartController

负责管理建模与预测标签页中的各类图表组件；在用户点击「开始仿真」并完成
``ModelTab.generate_prediction_curves`` 后，由 ``ModelTab`` 调用本控制器刷新四张图，
展示基于模型与侧栏参数（当量、含铝量等）的预测与工程计算结果：
- 火球直径随时间变化图
- 火球温度随时间变化图
- 热通量随时间变化图（不同距离）
- 累积热辐射量随距离分布图
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

from chart_widgets.diameter_chart import DiameterChart
from chart_widgets.temperature_chart import TemperatureChart
from chart_widgets.heat_flux_chart import HeatFluxChart
from chart_widgets.radiation_chart import RadiationChart


class ModelTabChartController:
    """建模与预测模块的图表控制器。"""

    def __init__(self, ui_builder) -> None:
        """
        初始化控制器，并从 UI 构建器中获取所需的图表组件。

        Args:
            ui_builder: ModelTabUI 实例，用于提供 UI 组件引用。
        """
        ui_components = ui_builder.get_ui_components()
        self._diam_chart: Optional[DiameterChart] = ui_components.get('diam_chart')
        self._temp_chart: Optional[TemperatureChart] = ui_components.get('temp_chart')
        self._heat_flux_chart: Optional[HeatFluxChart] = ui_components.get('heat_flux_chart')
        self._heat_radiation_chart: Optional[RadiationChart] = ui_components.get('heat_radiation_chart')

        self.reset()

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """重置所有图表为占位状态。"""
        if self._diam_chart:
            self._diam_chart.reset()
        if self._temp_chart:
            self._temp_chart.reset()
        if self._heat_flux_chart:
            self._heat_flux_chart.reset()
        if self._heat_radiation_chart:
            self._heat_radiation_chart.reset()

    def update_diameter(self,
                        time_ms: Sequence[float],
                        diameter_m: Sequence[float]) -> None:
        """更新直径曲线。"""
        self._ensure_initialized(self._diam_chart, 'diam_chart')
        self._diam_chart.update_data(time_ms, diameter_m)  # type: ignore[arg-type]

    def update_temperature(self,
                           time_ms: Sequence[float],
                           temperature_k: Sequence[float]) -> None:
        """更新温度曲线。"""
        self._ensure_initialized(self._temp_chart, 'temp_chart')
        self._temp_chart.update_data(time_ms, temperature_k)  # type: ignore[arg-type]

    def update_heat_flux(self,
                         time_ms: Sequence[float],
                         heat_flux_series: Sequence[Sequence[Any]]) -> None:
        """
        更新热通量曲线。

        Args:
            time_ms: 时间序列（毫秒）
            heat_flux_series: 热通量数据列表，元素形如 [distance(float), heat_flux(array-like)]
        """
        self._ensure_initialized(self._heat_flux_chart, 'heat_flux_chart')
        self._heat_flux_chart.update_data(time_ms, heat_flux_series)  # type: ignore[arg-type]

    def update_heat_radiation(self,
                              distances: Sequence[float],
                              heat_radiation: Sequence[float]) -> None:
        """更新累积热辐射量曲线。"""
        self._ensure_initialized(self._heat_radiation_chart, 'heat_radiation_chart')
        self._heat_radiation_chart.update_data(distances, heat_radiation)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_initialized(chart, name: str) -> None:
        if chart is None:
            raise RuntimeError(f"ModelTabChartController: 图表组件 '{name}' 未正确初始化")


