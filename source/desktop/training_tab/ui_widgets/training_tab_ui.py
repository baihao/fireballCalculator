#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块 UI 构建器。

左侧操作面板：**全局侧边栏**（`create_sidebar_widget`），与 `extract_tab` 一致；
中间标签页：仅图表区 + 训练日志。
"""

from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from chart_widgets import (
    FireballTrainingScatterChart,
    KernelRegressionTrainingCurveChart,
)


class TrainingTabUI:
    """模型训练模块 UI 构建器。"""

    def __init__(self) -> None:
        self.ui_components: Dict[str, Any] = {}

    @staticmethod
    def _sidebar_section_style() -> str:
        """与 `extract_tab_ui._param_group_style` 一致，侧栏分区与「机器视觉」同源。"""
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #374151;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #111827;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #38bdf8;
            }
        """

    # -------------------------------------------------------------------------
    # 左侧控件占位：由侧边栏或主区域创建入口统一调用，避免控件未实例化。
    # -------------------------------------------------------------------------
    def _ensure_sidebar_controls(self) -> None:
        if self.ui_components.get("train_input_btn") is not None:
            return

        self.ui_components["train_input_btn"] = QPushButton("输入数据")
        self.ui_components["train_input_status"] = QLabel("未加载训练数据")

        self.ui_components["train_model_combo"] = QComboBox()
        self.ui_components["train_model_combo"].addItems(["核回归", "高斯过程"])

        self.ui_components["train_test_ratio_combo"] = QComboBox()
        for pct in (10, 15, 20, 25, 30, 35, 40):
            self.ui_components["train_test_ratio_combo"].addItem(f"{pct}%", pct)
        self.ui_components["train_test_ratio_combo"].setCurrentIndex(2)

        self.ui_components["train_start_btn"] = QPushButton("开始训练")
        self.ui_components["train_start_btn"].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )

        self.ui_components["train_dataset_summary"] = QPlainTextEdit()
        self.ui_components["train_dataset_summary"].setReadOnly(True)
        self.ui_components["train_dataset_summary"].setMinimumHeight(100)
        self.ui_components["train_dataset_summary"].setMaximumHeight(200)
        self.ui_components["train_dataset_summary"].setPlaceholderText(
            "加载数据后将显示样本数及按测试集比例划分后的训练 / 测试集数量…"
        )
        self.ui_components["train_dataset_summary"].setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 8px;
                color: #cbd5e1;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)

    def create_sidebar_widget(self) -> QGroupBox:
        """挂载到 **主窗口左侧边栏**，与 ExtractTab / ModelTab 侧栏结构一致。"""
        self._ensure_sidebar_controls()

        sidebar_widget = QGroupBox("模型训练")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        gb_in = QGroupBox("输入数据")
        gb_in.setStyleSheet(self._sidebar_section_style())
        gv = QVBoxLayout()
        gv.setAlignment(Qt.AlignmentFlag.AlignTop)
        gv.setSpacing(8)
        gv.addWidget(self.ui_components["train_input_btn"])
        hint1 = QLabel("实现时加载多组爆炸实验记录（CSV/JSON/XLSX 等），供右侧散点图与训练曲线使用。")
        hint1.setWordWrap(True)
        gv.addWidget(hint1)
        gv.addWidget(self.ui_components["train_input_status"])
        gb_in.setLayout(gv)
        layout.addWidget(gb_in)

        gb_alg = QGroupBox("模型训练")
        gb_alg.setStyleSheet(self._sidebar_section_style())
        av = QVBoxLayout()
        av.setAlignment(Qt.AlignmentFlag.AlignTop)
        av.setSpacing(8)
        av.addWidget(QLabel("算法"))
        av.addWidget(self.ui_components["train_model_combo"])
        ah = QLabel('默认「核回归」：第四图为带宽 σ 与训练 / 测试 MSE。')
        ah.setWordWrap(True)
        av.addWidget(ah)
        gb_alg.setLayout(av)
        layout.addWidget(gb_alg)

        gb_run = QGroupBox("划分与执行")
        gb_run.setStyleSheet(self._sidebar_section_style())
        rv = QVBoxLayout()
        rv.setAlignment(Qt.AlignmentFlag.AlignTop)
        rv.setSpacing(8)
        rv.addWidget(QLabel("测试集比例"))
        rv.addWidget(self.ui_components["train_test_ratio_combo"])
        rv.addWidget(self.ui_components["train_start_btn"])
        gb_run.setLayout(rv)
        layout.addWidget(gb_run)

        gb_sum = QGroupBox("训练数据集信息")
        gb_sum.setStyleSheet(self._sidebar_section_style())
        sv = QVBoxLayout()
        sv.setAlignment(Qt.AlignmentFlag.AlignTop)
        sv.setSpacing(8)
        sv.addWidget(QLabel("数据与划分概要"))
        sv.addWidget(self.ui_components["train_dataset_summary"])
        sh = QLabel("只读概要；接入后端后以真实数据集为准。")
        sh.setWordWrap(True)
        sv.addWidget(sh)
        gb_sum.setLayout(sv)
        layout.addWidget(gb_sum)

        layout.addStretch()
        sidebar_widget.setLayout(layout)
        return sidebar_widget

    # -------------------------------------------------------------------------
    # 标签页正文：图表 + 日志（无左侧 splitter）
    # -------------------------------------------------------------------------
    def create_main_layout(self, parent_widget: QWidget) -> QVBoxLayout:
        self._ensure_sidebar_controls()

        vl = QVBoxLayout()
        vl.addWidget(self._build_chart_area())
        parent_widget.setLayout(vl)
        return vl

    def _build_chart_area(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)

        toolbar = QHBoxLayout()
        title = QLabel("模型训练视图")
        title.setStyleSheet("font-size:14px;font-weight:bold;color:#38bdf8;")
        toolbar.addWidget(title)
        toolbar.addStretch()
        self.ui_components["train_model_status_label"] = QLabel("算法：核回归")
        self.ui_components["train_model_status_label"].setStyleSheet("color:#9ca3af;font-size:12px;")
        toolbar.addWidget(self.ui_components["train_model_status_label"])
        vl.addLayout(toolbar)

        self.ui_components["train_hint_label"] = QLabel(
            "请先通过侧栏「输入数据」导入训练文件夹后再查看三张散点；第四图为核回归训练曲线（需在训练回填数据后绘制）。"
        )
        self.ui_components["train_hint_label"].setWordWrap(True)
        self.ui_components["train_hint_label"].setStyleSheet("color:#9ca3af;font-size:11px;margin-bottom:6px;")
        vl.addWidget(self.ui_components["train_hint_label"])

        grid = QGridLayout()
        grid.setSpacing(10)

        self.ui_components["scatter_max_chart"] = FireballTrainingScatterChart.for_max_diameter(
            width=5, height=3
        )
        self.ui_components["scatter_init_chart"] = FireballTrainingScatterChart.for_initial_state_constant(
            width=5, height=3
        )
        self.ui_components["scatter_tau_chart"] = FireballTrainingScatterChart.for_time_constant(
            width=5, height=3
        )
        self.ui_components["chart_train_curve"] = KernelRegressionTrainingCurveChart(width=5, height=3)

        self.ui_components["train_gp_curve_placeholder"] = QLabel("")
        self.ui_components["train_gp_curve_placeholder"].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui_components["train_gp_curve_placeholder"].setMinimumHeight(180)
        self.ui_components["train_gp_curve_placeholder"].setStyleSheet(
            "background-color: #111827; color:#9ca3af; font-size:12px; border: none;"
        )
        self.ui_components["train_gp_curve_placeholder"].hide()

        curve_cell = QWidget()
        curve_cell.setStyleSheet("border: none; background: transparent;")
        cl = QVBoxLayout(curve_cell)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.ui_components["chart_train_curve"])
        cl.addWidget(self.ui_components["train_gp_curve_placeholder"])

        # 与 model_tab「机器学习」一致：图表控件直接放入网格，不套带边框外层
        grid.addWidget(self.ui_components["scatter_max_chart"], 0, 0)
        grid.addWidget(self.ui_components["scatter_init_chart"], 0, 1)
        grid.addWidget(self.ui_components["scatter_tau_chart"], 1, 0)
        grid.addWidget(curve_cell, 1, 1)

        vl.addLayout(grid)

        log_l = QLabel("训练日志")
        log_l.setStyleSheet("color:#38bdf8;font-size:12px;font-weight:bold;")
        vl.addWidget(log_l)
        self.ui_components["train_log"] = QPlainTextEdit()
        self.ui_components["train_log"].setReadOnly(True)
        self.ui_components["train_log"].setMinimumHeight(100)
        self.ui_components["train_log"].setPlaceholderText("[训练] 数据导入、训练与评估输出…")
        self.ui_components["train_log"].setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 8px;
                color: #cbd5e1;
                font-family: ui-monospace, 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
            }
        """)
        vl.addWidget(self.ui_components["train_log"])

        return w

    def get_ui_components(self) -> Dict[str, Any]:
        return self.ui_components
