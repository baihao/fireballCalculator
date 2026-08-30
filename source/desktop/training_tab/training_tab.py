#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型训练标签页：侧边栏导入目录、KRR 训练与预测曲线叠加散点。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, auto

import os

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from .controllers.training_chart_controller import TrainingChartController
from .training_dataset_model import TrainingDatasetModel
from .ui_widgets.training_tab_ui import TrainingTabUI
from .utils.dataset_io import import_training_folder
from .utils.krr_workflow import (
    KrrPredictGrid,
    KrrTrainingSummary,
    krr_prediction_log_lines,
    krr_training_log_lines,
    run_train_and_predict,
)
from .utils.training_summary import build_training_summary_text

# UI「数据合规」阈值：少于该条数仍可导入，但不进入可训练状态，且弹窗提示
MIN_SAMPLES_UI_READY = 5


class TrainingUiState(Enum):
    """标签页左侧「开始训练」相关交互相位。"""

    INITIAL = auto()  # 未导入数据，训练按钮不可用
    DATA_IMPORTED = auto()  # 已导入但样本 < MIN_SAMPLES_UI_READY，训练按钮不可用
    DATA_PREPARED = auto()  # 样本 >= MIN_SAMPLES_UI_READY，可进行训练
    TRAINING_DONE = auto()  # 已成功训练过一轮，按钮文案为「重新训练」


class TrainingTab(QWidget):
    """中间区域仅图表与日志；操作面板在 **主窗口左侧边栏**，与 ExtractTab 相同模式。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = TrainingDatasetModel()
        self._last_train_summary: KrrTrainingSummary | None = None
        self._last_predict_grid: KrrPredictGrid | None = None
        self._training_summary_status = "未训练"
        self._ui_state = TrainingUiState.INITIAL
        self._ui_builder = TrainingTabUI()
        # 先确保侧边栏控件已创建（与图表共用 ui_components）
        self._sidebar_widget = self._ui_builder.create_sidebar_widget()
        self._ui_builder.create_main_layout(self)
        self.ui_components = self._ui_builder.get_ui_components()

        self._chart_controller = TrainingChartController(self._ui_builder, self._model)

        self._wire_signals()
        self._refresh_summary()
        self._refresh_training_summary_panel()
        self._sync_import_status()
        self._sync_data_hint()
        self._apply_training_action_ui()

    def get_sidebar_widget(self):
        """供 `FireballAnalysisApp` 挂载到全局左侧边栏。"""
        return self._sidebar_widget

    def _wire_signals(self) -> None:
        c = self.ui_components
        c["train_split_strategy_combo"].currentIndexChanged.connect(self._on_split_strategy_changed)
        c["train_input_btn"].clicked.connect(self._on_input_data_clicked)
        c["train_start_btn"].clicked.connect(self._on_start_training_clicked)

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _append_train_log(self, line: str) -> None:
        te = self.ui_components["train_log"]
        te.appendPlainText(f"[{self._now()}] {line}")

    def _refresh_summary(self) -> None:
        self.ui_components["train_dataset_summary"].setPlainText(self._model.summary_text())

    def _refresh_training_summary_panel(self) -> None:
        text = build_training_summary_text(
            status=self._training_summary_status,
            n_samples=self._model.total_samples,
            split_strategy=self._model.split_strategy,
            data_folder=self._model.data_folder,
            artifact_root=self._model.last_krr_artifact_root,
            train_summary=self._last_train_summary,
            predict_grid=self._last_predict_grid,
        )
        self.ui_components["train_summary"].setPlainText(text)

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
                "已载入数据：三张散点纵轴依次为拟合 K、B、C；点大小 ∝ 含铝量。"
                "划分策略为留一交叉验证。"
            )
        else:
            hint.setText(
                "请先通过侧栏「输入数据」导入训练文件夹；需多于 5 条样本更有利于训练效果。"
            )

    def _apply_training_action_ui(self) -> None:
        """根据 ``_ui_state`` 同步「开始训练 / 重新训练」文案与启用状态。"""
        btn = self.ui_components["train_start_btn"]
        if self._ui_state == TrainingUiState.TRAINING_DONE:
            btn.setText("重新训练")
        else:
            btn.setText("开始训练")
        can_act = self._ui_state in (TrainingUiState.DATA_PREPARED, TrainingUiState.TRAINING_DONE)
        btn.setEnabled(can_act)

    def _set_ui_state_after_import(self, sample_count: int) -> None:
        """导入成功且非空样本后调用；按条数写入 DATA_IMPORTED / DATA_PREPARED。"""
        if sample_count <= 0:
            self._ui_state = TrainingUiState.INITIAL
            return
        if sample_count < MIN_SAMPLES_UI_READY:
            self._ui_state = TrainingUiState.DATA_IMPORTED
        else:
            self._ui_state = TrainingUiState.DATA_PREPARED

    def _on_split_strategy_changed(self, index: int) -> None:
        combo = self.ui_components["train_split_strategy_combo"]
        key = combo.currentData()
        self._model.set_split_strategy(str(key))
        self._refresh_summary()

    def input_training_data(self) -> None:
        """导入训练数据文件夹（侧栏「输入数据」与菜单「文件 → 输入数据」共用）。"""
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

        n = len(result.records)
        self._model.set_loaded_training_folder(result.folder_resolved, result.records)
        self._last_train_summary = None
        self._last_predict_grid = None
        self._training_summary_status = "未训练"

        self._set_ui_state_after_import(n)
        if 0 < n < MIN_SAMPLES_UI_READY:
            QMessageBox.warning(
                self,
                "训练数据",
                f"当前有效样本为 {n} 条，少于 {MIN_SAMPLES_UI_READY} 条，难以获得较好训练效果。"
                "请先扩充数据后再开始训练（「开始训练」将保持不可用）。",
            )

        self._refresh_summary()
        self._refresh_training_summary_panel()
        self._sync_import_status()
        self._sync_data_hint()
        self._chart_controller.redraw_scatters_from_training_model()
        self._apply_training_action_ui()

        log_line = (
            f"[模型训练] 已从目录载入 {len(result.records)} 组有效样本 "
            f"({os.path.basename(result.folder_resolved)})"
        )
        self._append_train_log(log_line)
        for line in result.diagnostics[:25]:
            self._append_train_log(f"[模型训练] 跳过 — {line}")
        if len(result.diagnostics) > 25:
            self._append_train_log(f"[模型训练] … 另有 {len(result.diagnostics) - 25} 条跳过记录")

    def _on_input_data_clicked(self) -> None:
        self.input_training_data()

    def _on_start_training_clicked(self) -> None:
        if self._ui_state not in (TrainingUiState.DATA_PREPARED, TrainingUiState.TRAINING_DONE):
            return
        if self._model.total_samples < 2:
            QMessageBox.warning(self, "模型训练", "至少需要 2 条有效样本才能进行留一交叉验证与训练。")
            return
        if not self._model.data_folder:
            QMessageBox.warning(self, "模型训练", "请先通过「输入数据」加载训练文件夹。")
            return

        is_retrain = self._ui_state == TrainingUiState.TRAINING_DONE
        if is_retrain:
            self._chart_controller.redraw_scatters_from_training_model()
            self._append_train_log("[模型训练] 已去除上一轮预测曲线，开始重新训练…")

        btn = self.ui_components["train_start_btn"]
        btn.setEnabled(False)
        self._model.last_krr_artifact_root = None
        self._last_train_summary = None
        self._last_predict_grid = None
        self._training_summary_status = "训练中"
        self._refresh_summary()
        self._refresh_training_summary_panel()

        self._append_train_log("[模型训练] 执行中：依次进行「模型训练（LOOCV）」与「预测网格」…")
        try:
            saved_root, grid, train_summary = run_train_and_predict(self._model)
        except ImportError as e:
            self._append_train_log(f"[模型训练] 模块导入失败：{e}")
            QMessageBox.critical(
                self,
                "训练失败",
                f"无法加载 kernel_regression（请从 source/desktop 启动并确保依赖已安装）：\n{e}",
            )
            self._training_summary_status = "训练失败"
            self._refresh_training_summary_panel()
            self._ui_state = (
                TrainingUiState.DATA_PREPARED
                if self._model.total_samples >= MIN_SAMPLES_UI_READY
                else TrainingUiState.DATA_IMPORTED
            )
            self._apply_training_action_ui()
            return
        except Exception as e:
            self._append_train_log(f"[模型训练] 训练失败：{e}")
            QMessageBox.critical(self, "训练失败", str(e))
            self._training_summary_status = "训练失败"
            self._refresh_training_summary_panel()
            self._ui_state = (
                TrainingUiState.DATA_PREPARED
                if self._model.total_samples >= MIN_SAMPLES_UI_READY
                else TrainingUiState.DATA_IMPORTED
            )
            self._apply_training_action_ui()
            return

        self._model.last_krr_artifact_root = str(saved_root.resolve())
        self._last_train_summary = train_summary
        self._last_predict_grid = grid
        self._training_summary_status = "训练成功"
        self._ui_state = TrainingUiState.TRAINING_DONE
        self._refresh_summary()
        self._refresh_training_summary_panel()

        for line in krr_training_log_lines(train_summary):
            self._append_train_log(line)
        self._append_train_log(f"[模型训练] artefact 目录：{saved_root}")
        for line in krr_prediction_log_lines(grid):
            self._append_train_log(line)

        self._chart_controller.redraw_scatters_with_prediction_curves(
            grid.equiv_grid,
            grid.al_levels,
            grid.K,
            grid.B,
            grid.C,
        )
        self._append_train_log(
            f"[模型训练] 绘图已更新：在散点上叠加预测曲线（当量采样 {len(grid.equiv_grid)} 点 × "
            f"{len(grid.al_levels)} 档含铝）。"
        )
        self._apply_training_action_ui()

    def get_ui_components(self):
        return self.ui_components
