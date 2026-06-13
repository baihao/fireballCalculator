#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建模与预测模块 UI 构建器
负责创建和配置所有 UI 组件
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
    QScrollArea,
    QFormLayout,
    QPlainTextEdit,
)
from PySide6.QtCore import Qt

from chart_widgets import (
    DiameterChart,
    TemperatureChart,
    HeatFluxChart,
    RadiationChart,
)


class ModelTabUI:
    """建模与预测模块 UI 构建器"""

    def __init__(self):
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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("仿真预测结果"))
        toolbar.addStretch()

        self.ui_components["modeling_status"] = QLabel("未开始")
        self.ui_components["modeling_status"].setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.ui_components["modeling_status"])
        layout.addLayout(toolbar)

        # 四个图表网格
        charts_widget = QWidget()
        charts_layout = QGridLayout()
        charts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.ui_components["diam_chart"] = DiameterChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["diam_chart"], 0, 0)

        self.ui_components["temp_chart"] = TemperatureChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["temp_chart"], 0, 1)

        self.ui_components["heat_flux_chart"] = HeatFluxChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["heat_flux_chart"], 1, 0)

        self.ui_components["heat_radiation_chart"] = RadiationChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["heat_radiation_chart"], 1, 1)

        charts_widget.setLayout(charts_layout)
        layout.addWidget(charts_widget)

        log_label = QLabel("仿真日志")
        log_label.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: bold;")
        layout.addWidget(log_label)

        self.ui_components["simulation_log"] = QPlainTextEdit()
        self.ui_components["simulation_log"].setReadOnly(True)
        self.ui_components["simulation_log"].setMinimumHeight(100)
        self.ui_components["simulation_log"].setPlaceholderText(
            "[仿真] 完成一次「开始仿真」后将输出火球直径、温度、热通量与累积热辐射等关键指标…"
        )
        self.ui_components["simulation_log"].setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 8px;
                color: #cbd5e1;
                font-family: ui-monospace, 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
            }
        """
        )
        layout.addWidget(self.ui_components["simulation_log"])

        parent_widget.setLayout(layout)
        return layout

    def create_sidebar_widget(self) -> QGroupBox:
        """创建侧边栏组件"""
        sidebar_widget = QGroupBox("工程计算")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)

        import_group = QGroupBox("模型导入")
        import_layout = QVBoxLayout()
        import_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        import_layout.setSpacing(8)

        self.ui_components["model_select_btn"] = QPushButton("选择模型")
        self.ui_components["model_select_btn"].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )
        import_layout.addWidget(self.ui_components["model_select_btn"])

        import_layout.addWidget(QLabel("模型与参数概要"))
        self.ui_components["model_import_summary"] = QPlainTextEdit()
        self.ui_components["model_import_summary"].setReadOnly(True)
        self.ui_components["model_import_summary"].setMinimumHeight(100)
        self.ui_components["model_import_summary"].setMaximumHeight(220)
        self.ui_components["model_import_summary"].setPlaceholderText(
            "选择模型目录后将显示路径、火球实验 JSON 与核岭回归 artefact 等信息…"
        )
        self.ui_components["model_import_summary"].setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 8px;
                color: #cbd5e1;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """
        )
        import_layout.addWidget(self.ui_components["model_import_summary"])

        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        simulate_group = QGroupBox("仿真预测")
        simulate_layout = QVBoxLayout()
        simulate_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        simulate_layout.addWidget(QLabel("仿真参数"))
        params_container = QWidget()
        params_form = QFormLayout()
        params_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        params_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self.ui_components["p_eq"] = QLineEdit("10")
        params_form.addRow("当量 (kg TNT)", self.ui_components["p_eq"])

        self.ui_components["p_al"] = QLineEdit("30")
        params_form.addRow("含铝量 (%)", self.ui_components["p_al"])

        self.ui_components["p_env_temp"] = QLineEdit("24")
        params_form.addRow("环境温度 (°C)", self.ui_components["p_env_temp"])

        self.ui_components["p_env_humidity"] = QLineEdit("48")
        params_form.addRow("相对湿度 (%)", self.ui_components["p_env_humidity"])

        self.ui_components["p_env_pressure"] = QLineEdit("2987.87")
        params_form.addRow("水饱和气压 (Pa)", self.ui_components["p_env_pressure"])

        self.ui_components["p_step"] = QLineEdit("1")
        params_form.addRow("仿真步长 (ms)", self.ui_components["p_step"])

        self.ui_components["p_duration"] = QLineEdit("140")
        params_form.addRow("仿真时长 (ms)", self.ui_components["p_duration"])

        params_container.setLayout(params_form)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_container)
        self.ui_components["params_scroll_area"] = params_scroll
        simulate_layout.addWidget(params_scroll)

        self.ui_components["predict_btn"] = QPushButton("开始仿真")
        self.ui_components["predict_btn"].setStyleSheet(
            "QPushButton { background-color: #10b981; color: white; }"
        )
        self.ui_components["predict_btn"].setEnabled(False)
        simulate_layout.addWidget(self.ui_components["predict_btn"])

        self.ui_components["export_btn"] = QPushButton("导出结果")
        self.ui_components["export_btn"].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )
        self.ui_components["export_btn"].setEnabled(False)
        simulate_layout.addWidget(self.ui_components["export_btn"])

        simulate_group.setLayout(simulate_layout)
        layout.addWidget(simulate_group)

        layout.addStretch()
        sidebar_widget.setLayout(layout)
        return sidebar_widget

    def get_ui_components(self) -> Dict:
        """获取所有 UI 组件的引用"""
        return self.ui_components.copy()
