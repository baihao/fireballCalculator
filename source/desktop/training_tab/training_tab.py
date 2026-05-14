#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型训练标签页：侧边栏导入训练目录、概要/散点绑定模型；训练执行仍占位。"""

from __future__ import annotations

from datetime import datetime

import os

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from .controllers.training_chart_controller import TrainingChartController
from .training_dataset_model import TrainingDatasetModel
from .ui_widgets.training_tab_ui import TrainingTabUI
from .utils.dataset_io import import_training_folder


class TrainingTab(QWidget):
    """中间区域仅图表与日志；操作面板在 **主窗口左侧边栏**，与 ExtractTab 相同模式。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = TrainingDatasetModel()
        self._ui_builder = TrainingTabUI()
        # 先确保侧边栏控件已创建（与图表共用 ui_components）
        self._sidebar_widget = self._ui_builder.create_sidebar_widget()
        self._ui_builder.create_main_layout(self)
        self.ui_components = self._ui_builder.get_ui_components()

        self._chart_controller = TrainingChartController(self._ui_builder, self._model)
        self._chart_controller.apply_algorithm_visibility()

        self._wire_signals()
        self._refresh_summary()
        self._sync_import_status()
        self._sync_data_hint()

    def get_sidebar_widget(self):
        """供 `FireballAnalysisApp` 挂载到全局左侧边栏。"""
        return self._sidebar_widget

    def _wire_signals(self) -> None:
        c = self.ui_components
        c["train_model_combo"].currentIndexChanged.connect(self._on_algorithm_changed)
        c["train_test_ratio_combo"].currentIndexChanged.connect(self._on_test_ratio_changed)
        c["train_input_btn"].clicked.connect(self._on_input_data_clicked)
        c["train_start_btn"].clicked.connect(self._on_start_training_clicked)

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _append_train_log(self, line: str) -> None:
        te = self.ui_components["train_log"]
        te.appendPlainText(f"[{self._now()}] {line}")

    def _refresh_summary(self) -> None:
        self.ui_components["train_dataset_summary"].setPlainText(self._model.summary_text())

    def _sync_import_status(self) -> None:
        lab = self.ui_components["train_input_status"]
        if self._model.total_samples > 0 and self._model.data_folder:
            lab.setText(
                f"已载入 {self._model.total_samples} 组｜{os.path.basename(self._model.data_folder)}"
            )
            lab.setToolTip(self._model.data_folder)
        else:
            lab.setText("未加载训练数据")
            lab.setToolTip("")

    def _sync_data_hint(self) -> None:
        hint = self.ui_components["train_hint_label"]
        if self._model.total_samples > 0:
            hint.setText(
                "已载入数据散点：纵轴依次为拟合 K（最大直径）、B（初始状态常数）、C（时间常数）；"
                "点大小 ∝ 含铝量。"
            )
        else:
            hint.setText(
                "请先通过侧栏「输入数据」导入训练文件夹；三张散点为当量与各拟合参数，点大小映射含铝量。"
            )

    def _sync_status_labels(self) -> None:
        name = "核回归" if self._model.algorithm == "kernel" else "高斯过程"
        self.ui_components["train_model_status_label"].setText(f"算法：{name}")

    def _on_algorithm_changed(self, index: int) -> None:
        self._model.set_algorithm("kernel" if index == 0 else "gp")
        self._sync_status_labels()
        self._refresh_summary()
        self._chart_controller.apply_algorithm_visibility()

    def _on_test_ratio_changed(self) -> None:
        combo = self.ui_components["train_test_ratio_combo"]
        pct = int(combo.currentData())
        self._model.set_test_pct(pct)
        self._refresh_summary()

    def _on_input_data_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择训练数据文件夹",
            self._model.data_folder or "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not path:
            return

        result = import_training_folder(path, recursive=True, strict_drag_fit_success=False)
        if not result.ok:
            self._append_train_log(f"[模型训练] 导入失败 — {result.error_message}")
            for line in result.diagnostics[:30]:
                self._append_train_log(f"[模型训练] 跳过 — {line}")
            QMessageBox.warning(self, "训练数据导入", result.error_message)
            return

        self._model.set_loaded_training_folder(result.folder_resolved, result.records)
        self._refresh_summary()
        self._sync_import_status()
        self._sync_data_hint()
        self._chart_controller.redraw_scatters_from_training_model()

        log_line = (
            f"[模型训练] 已从目录载入 {len(result.records)} 组有效样本 "
            f"({os.path.basename(result.folder_resolved)})"
        )
        self._append_train_log(log_line)
        for line in result.diagnostics[:25]:
            self._append_train_log(f"[模型训练] 跳过 — {line}")
        if len(result.diagnostics) > 25:
            self._append_train_log(f"[模型训练] … 另有 {len(result.diagnostics) - 25} 条跳过记录")

    def _on_start_training_clicked(self) -> None:
        self._append_train_log(
            '[模型训练] 开始训练 — utils.training_bridge.run_training 尚未实现'
        )
        QMessageBox.information(
            self,
            "占位",
            "训练执行已预留至 utils.training_bridge。\n核回归曲线需训练完成后写入数据时再绘制。",
        )

    def get_ui_components(self):
        return self.ui_components
