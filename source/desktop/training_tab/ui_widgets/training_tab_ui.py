#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块 UI 构建器。

左侧操作面板：**全局侧边栏**（`create_sidebar_widget`），与 `extract_tab` 一致；
中间标签页：三张散点图与训练日志（无第四训练曲线图）。
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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from chart_widgets import (
    FireballTrainingScatterChart,
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

        self.ui_components["train_model_label"] = QLabel("核岭回归（Kernel Ridge）")
        self.ui_components["train_model_label"].setWordWrap(True)

        self.ui_components["train_split_strategy_combo"] = QComboBox()
        self.ui_components["train_split_strategy_combo"].addItem("留一交叉验证", "loocv")

        self.ui_components["train_start_btn"] = QPushButton("开始训练")
        self.ui_components["train_start_btn"].setEnabled(False)
        self.ui_components["train_start_btn"].setStyleSheet(
            "QPushButton { background-color: #0ea5e9; color: white; }"
        )

        self.ui_components["train_dataset_summary"] = QPlainTextEdit()
        self.ui_components["train_dataset_summary"].setReadOnly(True)
        self.ui_components["train_dataset_summary"].setMinimumHeight(100)
        self.ui_components["train_dataset_summary"].setMaximumHeight(200)
        self.ui_components["train_dataset_summary"].setPlaceholderText(
            "加载数据后将显示样本数、划分策略及样本明细…"
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

        sidebar_widget = QGroupBox("机器学习")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        gb_in = QGroupBox("输入数据")
        gb_in.setStyleSheet(self._sidebar_section_style())
        gv = QVBoxLayout()
        gv.setAlignment(Qt.AlignmentFlag.AlignTop)
        gv.setSpacing(8)
        gv.addWidget(self.ui_components["train_input_btn"])
        hint1 = QLabel(
            "加载包含多组实验数据的文件夹，数据需大于5条，否则无法获得较好的训练效果"
        )
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
        av.addWidget(self.ui_components["train_model_label"])
        ah = QLabel("核岭回归 RBF / KernelRidge，与留一交叉验证（LOOCV）配合训练 K、B、C。")
        ah.setWordWrap(True)
        av.addWidget(ah)
        gb_alg.setLayout(av)
        layout.addWidget(gb_alg)

        gb_run = QGroupBox("划分与执行")
        gb_run.setStyleSheet(self._sidebar_section_style())
        rv = QVBoxLayout()
        rv.setAlignment(Qt.AlignmentFlag.AlignTop)
        rv.setSpacing(8)
        rv.addWidget(QLabel("划分策略"))
        rv.addWidget(self.ui_components["train_split_strategy_combo"])
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
        sh = QLabel("只读概要；需多于 5 条样本时更有利于模型训练。")
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
        title = QLabel("机器学习视图")
        title.setStyleSheet("font-size:14px;font-weight:bold;color:#38bdf8;")
        toolbar.addWidget(title)
        toolbar.addStretch()
        self.ui_components["train_model_status_label"] = QLabel("算法：核岭回归")
        self.ui_components["train_model_status_label"].setStyleSheet("color:#9ca3af;font-size:12px;")
        toolbar.addWidget(self.ui_components["train_model_status_label"])
        vl.addLayout(toolbar)

        self.ui_components["train_hint_label"] = QLabel(
            "请先通过侧栏「输入数据」导入训练文件夹后再查看三张散点（当量–K/B/C，点大小 ∝ 含铝量）。"
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

        grid.addWidget(self.ui_components["scatter_max_chart"], 0, 0)
        grid.addWidget(self.ui_components["scatter_init_chart"], 0, 1)
        grid.addWidget(self.ui_components["scatter_tau_chart"], 0, 2)

        vl.addLayout(grid, 2)

        expanding_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)

        log_column = QVBoxLayout()
        log_column.setContentsMargins(0, 0, 0, 0)
        log_l = QLabel("训练日志")
        log_l.setStyleSheet("color:#38bdf8;font-size:12px;font-weight:bold;")
        log_column.addWidget(log_l)
        self.ui_components["train_log"] = QPlainTextEdit()
        self.ui_components["train_log"].setReadOnly(True)
        self.ui_components["train_log"].setSizePolicy(expanding_policy)
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
        log_column.addWidget(self.ui_components["train_log"], 1)

        summary_column = QVBoxLayout()
        summary_column.setContentsMargins(0, 0, 0, 0)
        summary_l = QLabel("训练摘要")
        summary_l.setStyleSheet("color:#38bdf8;font-size:12px;font-weight:bold;")
        summary_column.addWidget(summary_l)
        self.ui_components["train_summary"] = QPlainTextEdit()
        self.ui_components["train_summary"].setReadOnly(True)
        self.ui_components["train_summary"].setSizePolicy(expanding_policy)
        self.ui_components["train_summary"].setMinimumHeight(100)
        self.ui_components["train_summary"].setPlaceholderText(
            "完成训练后将显示 LOOCV 精度、超参与预测网格概要…"
        )
        self.ui_components["train_summary"].setStyleSheet("""
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
        summary_column.addWidget(self.ui_components["train_summary"], 1)

        bottom_row.addLayout(log_column, 1)
        bottom_row.addLayout(summary_column, 1)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_row)
        bottom_widget.setSizePolicy(expanding_policy)
        vl.addWidget(bottom_widget, 1)

        return w

    def get_ui_components(self) -> Dict[str, Any]:
        return self.ui_components
