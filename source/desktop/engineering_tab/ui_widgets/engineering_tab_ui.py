#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工程计算 Tab — 侧栏输入 + 主区结果。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui_fonts import song_family_qss


class EngineeringTabUI:
    def __init__(self) -> None:
        self.ui_components: dict = {}
        self._build_inputs()
        self._build_main()

    def get_ui_components(self) -> dict:
        return self.ui_components

    def create_sidebar_widget(self) -> QGroupBox:
        return self._sidebar_group

    def create_main_widget(self) -> QWidget:
        return self._main_root

    def _result_panel_style(self) -> str:
        fq = song_family_qss()
        return f"""
            QTextEdit {{
                background-color: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 8px;
                color: #e5e7eb;
                font-family: "Menlo", "Consolas", "Courier New", monospace, {fq};
                font-size: 12px;
                padding: 4px;
            }}
        """

    def _build_inputs(self) -> None:
        self._sidebar_group = QGroupBox("工程计算")
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.setSpacing(12)

        input_group = QGroupBox("输入参数")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.ui_components["eq_kg"] = QLineEdit("2000")
        form.addRow("装药质量 M / TNT 当量 W (kg)", self.ui_components["eq_kg"])
        self.ui_components["al_percent"] = QLineEdit("30")
        form.addRow("含铝率 X (%)", self.ui_components["al_percent"])
        input_group.setLayout(form)
        sidebar_layout.addWidget(input_group)

        self.ui_components["calc_btn"] = QPushButton("开始计算")
        self.ui_components["calc_btn"].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )
        sidebar_layout.addWidget(self.ui_components["calc_btn"])
        sidebar_layout.addStretch()

        self._sidebar_group.setLayout(sidebar_layout)

    def _build_main(self) -> None:
        self._main_root = QWidget()
        layout = QVBoxLayout(self._main_root)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("计算结果（含公式与逐步代入）")
        header.setStyleSheet("color: #38bdf8; font-weight: bold;")
        layout.addWidget(header)

        self.ui_components["result_panel"] = QTextEdit()
        self.ui_components["result_panel"].setReadOnly(True)
        self.ui_components["result_panel"].setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.ui_components["result_panel"].setPlaceholderText(
            "在左侧填写 M（W）、含铝率，点击「开始计算」后在此显示完整计算流程…"
        )
        self.ui_components["result_panel"].setStyleSheet(self._result_panel_style())
        layout.addWidget(self.ui_components["result_panel"], 1)
