#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建模与预测模块UI构建器
负责创建和配置所有UI组件
"""

from typing import Dict
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QListWidget,
    QAbstractItemView,
    QScrollArea,
    QFormLayout,
    QComboBox,
)
from PySide6.QtCore import Qt
from chart_widgets import (
    DiameterChart,
    TemperatureChart,
    HeatFluxChart,
    RadiationChart,
)


class ModelTabUI:
    """建模与预测模块UI构建器"""
    
    def __init__(self):
        """初始化UI构建器"""
        self.ui_components = {}
    
    def create_main_layout(self, parent_widget: QWidget) -> QVBoxLayout:
        """
        创建主界面布局
        
        Args:
            parent_widget: 父控件
            
        Returns:
            QVBoxLayout: 主布局
        """
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("仿真预测结果"))
        toolbar.addStretch()
        
        self.ui_components['modeling_status'] = QLabel("未开始")
        self.ui_components['modeling_status'].setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.ui_components['modeling_status'])
        layout.addLayout(toolbar)
        
        # 四个图表网格
        charts_widget = QWidget()
        charts_layout = QGridLayout()
        charts_layout.setAlignment(Qt.AlignTop)
        
        # 火球直径随时间变化
        self.ui_components['diam_chart'] = DiameterChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components['diam_chart'], 0, 0)
        
        # 火球温度随时间变化
        self.ui_components['temp_chart'] = TemperatureChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components['temp_chart'], 0, 1)
        
        # 热通量随时间变化 (不同距离)
        self.ui_components['heat_flux_chart'] = HeatFluxChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components['heat_flux_chart'], 1, 0)
        
        # 累积热辐射量随距离分布
        self.ui_components['heat_radiation_chart'] = RadiationChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components['heat_radiation_chart'], 1, 1)
        
        charts_widget.setLayout(charts_layout)
        layout.addWidget(charts_widget)
        
        parent_widget.setLayout(layout)
        return layout
    
    def create_sidebar_widget(self) -> QGroupBox:
        """创建侧边栏组件"""
        sidebar_widget = QGroupBox("机器学习")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(16)
        
        # 模型训练区域
        training_group = QGroupBox("模型训练")
        training_layout = QVBoxLayout()
        training_layout.setAlignment(Qt.AlignTop)
        training_layout.addWidget(QLabel("选择训练时间序列（可多选）"))
        
        self.ui_components['train_series_btn'] = QPushButton("选择训练文件")
        training_layout.addWidget(self.ui_components['train_series_btn'])
        
        self.ui_components['train_file_list'] = QListWidget()
        self.ui_components['train_file_list'].setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ui_components['train_file_list'].setVisible(False)
        self.ui_components['train_file_list'].setMaximumHeight(120)
        training_layout.addWidget(self.ui_components['train_file_list'])
        
        self.ui_components['train_params_btn'] = QPushButton("训练参数")
        training_layout.addWidget(self.ui_components['train_params_btn'])
        
        self.ui_components['train_btn'] = QPushButton("开始训练")
        self.ui_components['train_btn'].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )
        training_layout.addWidget(self.ui_components['train_btn'])
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # 仿真预测区域
        simulate_group = QGroupBox("仿真预测")
        simulate_layout = QVBoxLayout()
        simulate_layout.setAlignment(Qt.AlignTop)
        
        simulate_layout.addWidget(QLabel("模型选择"))
        self.ui_components['model_list'] = QComboBox()
        self.ui_components['model_list'].addItems(["示例模型 v1"])
        simulate_layout.addWidget(self.ui_components['model_list'])
        
        simulate_layout.addWidget(QLabel("仿真参数"))
        params_container = QWidget()
        params_form = QFormLayout()
        params_form.setLabelAlignment(Qt.AlignLeft)
        params_form.setFormAlignment(Qt.AlignTop)
        
        self.ui_components['p_eq'] = QLineEdit("10")
        params_form.addRow("当量 (kg TNT)", self.ui_components['p_eq'])
        
        self.ui_components['p_al'] = QLineEdit("30")
        params_form.addRow("含铝量 (%)", self.ui_components['p_al'])
        
        self.ui_components['p_env_temp'] = QLineEdit("24")
        params_form.addRow("环境温度 (°C)", self.ui_components['p_env_temp'])
        
        self.ui_components['p_env_humidity'] = QLineEdit("48")
        params_form.addRow("相对湿度 (%)", self.ui_components['p_env_humidity'])
        
        self.ui_components['p_env_pressure'] = QLineEdit("2987.87")
        params_form.addRow("水饱和气压 (Pa)", self.ui_components['p_env_pressure'])
        
        self.ui_components['p_step'] = QLineEdit("1")
        params_form.addRow("仿真步长 (ms)", self.ui_components['p_step'])
        
        self.ui_components['p_duration'] = QLineEdit("140")
        params_form.addRow("仿真时长 (ms)", self.ui_components['p_duration'])
        
        params_container.setLayout(params_form)
        
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_container)
        self.ui_components['params_scroll_area'] = params_scroll
        simulate_layout.addWidget(params_scroll)
        
        self.ui_components['predict_btn'] = QPushButton("开始仿真")
        self.ui_components['predict_btn'].setStyleSheet(
            "QPushButton { background-color: #10b981; color: white; }"
        )
        simulate_layout.addWidget(self.ui_components['predict_btn'])
        
        self.ui_components['export_btn'] = QPushButton("导出结果")
        self.ui_components['export_btn'].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )
        self.ui_components['export_btn'].setEnabled(False)
        simulate_layout.addWidget(self.ui_components['export_btn'])
        
        simulate_group.setLayout(simulate_layout)
        layout.addWidget(simulate_group)
        
        layout.addStretch()
        sidebar_widget.setLayout(layout)
        return sidebar_widget
    
    def get_ui_components(self) -> Dict:
        """
        获取所有UI组件的引用
        
        Returns:
            dict: UI组件字典
        """
        return self.ui_components.copy()

