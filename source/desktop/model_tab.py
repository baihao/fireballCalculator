#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建模与预测模块标签页
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QGridLayout)
from framework import MatplotlibWidget
from fireball_radius_calculator import FireballCalculator
from fireball_temperature_calculator import FireballTemperatureCalculator
from transmissivity_calculator import TransmissivityParams
from fireball_heat_radiation_calculator import (compute_heat_flux_over_time,
                                               integrate_heat_radiation)


class ModelTab(QWidget):
    """建模与预测模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("仿真预测结果"))
        toolbar.addStretch()
        self.modeling_status = QLabel("未开始")
        self.modeling_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.modeling_status)
        layout.addLayout(toolbar)
        
        # 四个图表网格
        charts_widget = QWidget()
        charts_layout = QGridLayout()
        
        # 火球直径随时间变化
        self.diam_chart = MatplotlibWidget(width=5, height=3)
        charts_layout.addWidget(self.diam_chart, 0, 0)
        
        # 火球温度随时间变化
        self.temp_chart = MatplotlibWidget(width=5, height=3)
        charts_layout.addWidget(self.temp_chart, 0, 1)
        
        # 热通量随时间变化 (不同距离)
        self.heat_flux_chart = MatplotlibWidget(width=5, height=3)
        charts_layout.addWidget(self.heat_flux_chart, 1, 0)
        
        # 累积热辐射量随距离分布
        self.heat_radiation_chart = MatplotlibWidget(width=5, height=3)
        charts_layout.addWidget(self.heat_radiation_chart, 1, 1)
        
        charts_widget.setLayout(charts_layout)
        layout.addWidget(charts_widget)
        
        self.setLayout(layout)
        
    def update_prediction_charts(self):
        """更新预测图表"""
        try:
            # 火球直径随时间变化
            t_ms = np.linspace(0, 140, 800)
            radius_calc = FireballCalculator()
            t_s = t_ms / 1000.0
            D_m = radius_calc.calculate_diameter(t_s, '40%Al/Rubber')
            
            self.diam_chart.plot_line(
                t_ms, D_m,
                title="火球直径随时间变化",
                xlabel="时间 (ms)",
                ylabel="直径 (m)",
                color='#22d3ee'
            )
            
            # 火球温度随时间变化
            temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
            T_K = temp_calc.temperature_modified(t_ms)
            
            self.temp_chart.plot_line(
                t_ms, T_K,
                title="火球温度随时间变化",
                xlabel="时间 (ms)",
                ylabel="温度 (K)",
                color='#38bdf8'
            )
            
            # 热通量随时间变化 (不同距离)
            distances = [4.0, 4.5, 5.0, 5.5, 6.0]
            heat_flux_data = {}
            
            for dist in distances:
                q_t = compute_heat_flux_over_time(dist, t_ms, T_K, D_m, TransmissivityParams())
                heat_flux_data[f'x = {dist:.1f} m'] = (t_ms, q_t)
            
            self.heat_flux_chart.plot_multiple_lines(
                heat_flux_data,
                title="热通量随时间变化 (不同距离)",
                xlabel="时间 (ms)",
                ylabel="热通量 (W/m²)"
            )
            
            # 累积热辐射量随距离分布
            x_values = np.linspace(4.0, 6.0, 50)
            H_values = []
            
            for x in x_values:
                q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, TransmissivityParams())
                H = integrate_heat_radiation(q_t, t_ms)
                H_values.append(H)
            
            self.heat_radiation_chart.plot_line(
                x_values, H_values,
                title="累积热辐射量随距离分布",
                xlabel="距离 (m)",
                ylabel="热辐射量 (J/m²)",
                color='#10b981'
            )
            
            self.modeling_status.setText("已生成示例预测曲线")
            
        except Exception as e:
            print(f"更新预测图表失败: {e}")
