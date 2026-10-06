#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数仿真模块 UI 构建器。"""

from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from chart_widgets import (
    DiameterChart,
    DiameterVelocityChart,
    HeatFluxChart,
    RadiationChart,
)


class ParameterSimTabUI:
    """参数仿真 UI：四图 + 仿真结果；侧栏为 K/B/C/峰值温度等。"""

    def __init__(self) -> None:
        self.ui_components: Dict = {}

    def create_main_layout(self, parent_widget: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("参数仿真结果"))
        toolbar.addStretch()
        self.ui_components["modeling_status"] = QLabel("未开始")
        self.ui_components["modeling_status"].setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.ui_components["modeling_status"])
        layout.addLayout(toolbar)

        charts_widget = QWidget()
        charts_widget.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        charts_layout = QGridLayout()
        charts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.ui_components["diam_chart"] = DiameterChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["diam_chart"], 0, 0)

        self.ui_components["velocity_chart"] = DiameterVelocityChart(
            width=5,
            height=3,
            title="火球膨胀速度随时间变化",
            y_label="膨胀速度 (m/ms)",
            raw_label="膨胀速度",
        )
        charts_layout.addWidget(self.ui_components["velocity_chart"], 0, 1)

        self.ui_components["heat_flux_chart"] = HeatFluxChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["heat_flux_chart"], 1, 0)

        self.ui_components["heat_radiation_chart"] = RadiationChart(width=5, height=3)
        charts_layout.addWidget(self.ui_components["heat_radiation_chart"], 1, 1)

        charts_widget.setLayout(charts_layout)

        expanding_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        result_column = QVBoxLayout()
        result_column.setContentsMargins(0, 0, 0, 0)
        result_label = QLabel("仿真结果")
        result_label.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: bold;")
        result_column.addWidget(result_label)

        self.ui_components["formula_reference"] = QPlainTextEdit()
        self.ui_components["formula_reference"].setReadOnly(True)
        self.ui_components["formula_reference"].setSizePolicy(expanding_policy)
        self.ui_components["formula_reference"].setMinimumHeight(160)
        self.ui_components["formula_reference"].setPlaceholderText(
            "完成「开始计算」后显示参数仿真结果…"
        )
        self.ui_components["formula_reference"].setStyleSheet(self._monospace_panel_style())
        result_column.addWidget(self.ui_components["formula_reference"], 1)

        bottom_widget = QWidget()
        bottom_widget.setLayout(result_column)
        bottom_widget.setSizePolicy(expanding_policy)

        layout.addWidget(charts_widget, 2)
        layout.addWidget(bottom_widget, 1)
        parent_widget.setLayout(layout)
        return layout

    @staticmethod
    def _monospace_panel_style() -> str:
        return """
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

    def create_sidebar_widget(self) -> QGroupBox:
        sidebar_widget = QGroupBox("参数仿真")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)

        simulate_group = QGroupBox("仿真参数")
        simulate_layout = QVBoxLayout()
        simulate_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        params_container = QWidget()
        params_form = QFormLayout()
        params_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        params_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self.ui_components["p_k"] = QLineEdit("30")
        params_form.addRow("K (m)", self.ui_components["p_k"])

        self.ui_components["p_b"] = QLineEdit("0.561313")
        params_form.addRow("B", self.ui_components["p_b"])

        self.ui_components["p_c"] = QLineEdit("8.49502e-05")
        params_form.addRow("C", self.ui_components["p_c"])

        self.ui_components["p_peak_temp_k"] = QLineEdit("1600")
        params_form.addRow("火球最大温度 (K)", self.ui_components["p_peak_temp_k"])

        self.ui_components["p_env_temp"] = QLineEdit("24")
        params_form.addRow("环境温度 (°C)", self.ui_components["p_env_temp"])

        self.ui_components["p_env_humidity"] = QLineEdit("48")
        params_form.addRow("相对湿度 (%)", self.ui_components["p_env_humidity"])

        self.ui_components["p_env_pressure"] = QLineEdit("2987.87")
        params_form.addRow("水饱和气压 (Pa)", self.ui_components["p_env_pressure"])

        self.ui_components["p_step"] = QLineEdit("1")
        params_form.addRow("仿真步长 (ms)", self.ui_components["p_step"])

        self.ui_components["p_duration"] = QLineEdit("2000")
        params_form.addRow("仿真时长 (ms)", self.ui_components["p_duration"])

        params_container.setLayout(params_form)
        simulate_layout.addWidget(params_container)

        self.ui_components["predict_btn"] = QPushButton("开始计算")
        self.ui_components["predict_btn"].setStyleSheet(
            "QPushButton { background-color: #10b981; color: white; }"
        )
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
        return self.ui_components.copy()
