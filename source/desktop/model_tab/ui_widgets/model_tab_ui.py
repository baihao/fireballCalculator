#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建模与预测模块UI构建器
负责创建和配置所有UI组件
"""

from typing import Dict
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QGridLayout, QPushButton, QComboBox, QLineEdit, QGroupBox)
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
        sidebar_widget = QGroupBox("建模与预测")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # 训练部分
        layout.addWidget(QLabel("建模 - 训练"))
        layout.addWidget(QLabel("选择训练时间序列（可多选）"))
        
        self.ui_components['train_series_btn'] = QPushButton("选择训练文件")
        layout.addWidget(self.ui_components['train_series_btn'])
        
        layout.addWidget(QLabel("算法"))
        self.ui_components['algo'] = QComboBox()
        self.ui_components['algo'].addItems(["T-Transformer"])
        layout.addWidget(self.ui_components['algo'])
        
        # 学习率和轮次
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学习率:"))
        self.ui_components['lr'] = QLineEdit("0.0005")
        lr_layout.addWidget(self.ui_components['lr'])
        lr_layout.addWidget(QLabel("轮次:"))
        self.ui_components['epochs'] = QLineEdit("50")
        lr_layout.addWidget(self.ui_components['epochs'])
        layout.addLayout(lr_layout)
        
        self.ui_components['train_btn'] = QPushButton("开始训练")
        self.ui_components['train_btn'].setStyleSheet("QPushButton { background-color: #0ea5e9; color: white; }")
        layout.addWidget(self.ui_components['train_btn'])
        
        # 预测部分
        layout.addWidget(QLabel("预测 - 运行"))
        layout.addWidget(QLabel("选择已训练模型"))
        
        self.ui_components['model_list'] = QComboBox()
        self.ui_components['model_list'].addItems(["示例模型 v1"])
        layout.addWidget(self.ui_components['model_list'])
        
        # 预测参数
        pred_layout = QHBoxLayout()
        pred_layout.addWidget(QLabel("当量:"))
        self.ui_components['p_eq'] = QLineEdit("10")
        pred_layout.addWidget(self.ui_components['p_eq'])
        pred_layout.addWidget(QLabel("含铝量:"))
        self.ui_components['p_al'] = QLineEdit("30")
        pred_layout.addWidget(self.ui_components['p_al'])
        layout.addLayout(pred_layout)
        
        # 环境参数
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("环境温度:"))
        self.ui_components['p_env_temp'] = QLineEdit("24")
        env_layout.addWidget(self.ui_components['p_env_temp'])
        env_layout.addWidget(QLabel("相对湿度:"))
        self.ui_components['p_env_humidity'] = QLineEdit("48")
        env_layout.addWidget(self.ui_components['p_env_humidity'])
        layout.addLayout(env_layout)
        
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(QLabel("水饱和气压:"))
        self.ui_components['p_env_pressure'] = QLineEdit("2987.87")
        pressure_layout.addWidget(self.ui_components['p_env_pressure'])
        pressure_layout.addStretch()
        layout.addLayout(pressure_layout)
        
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(QLabel("仿真步长:"))
        self.ui_components['p_step'] = QLineEdit("1")
        sim_layout.addWidget(self.ui_components['p_step'])
        sim_layout.addWidget(QLabel("仿真时长:"))
        self.ui_components['p_duration'] = QLineEdit("140")
        sim_layout.addWidget(self.ui_components['p_duration'])
        layout.addLayout(sim_layout)
        
        self.ui_components['predict_btn'] = QPushButton("开始预测")
        self.ui_components['predict_btn'].setStyleSheet("QPushButton { background-color: #10b981; color: white; }")
        layout.addWidget(self.ui_components['predict_btn'])
        
        # 导出结果按钮
        self.ui_components['export_btn'] = QPushButton("导出结果")
        self.ui_components['export_btn'].setStyleSheet("QPushButton { background-color: #0ea5e9; color: white; }")
        self.ui_components['export_btn'].setEnabled(False)  # 初始状态禁用
        layout.addWidget(self.ui_components['export_btn'])
        
        sidebar_widget.setLayout(layout)
        return sidebar_widget
    
    def get_ui_components(self) -> Dict:
        """
        获取所有UI组件的引用
        
        Returns:
            dict: UI组件字典
        """
        return self.ui_components.copy()

