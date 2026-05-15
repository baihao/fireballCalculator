#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练 — 图表控制器（与画布、算法切换视图相关）。
"""

from __future__ import annotations

import numpy as np

from ..training_dataset_model import TrainingDatasetModel
from ..ui_widgets.training_tab_ui import TrainingTabUI


class TrainingChartController:
    """封装四图的刷新与高斯过程占位切换。"""

    def __init__(self, ui_builder: TrainingTabUI, model: TrainingDatasetModel) -> None:
        self._ui = ui_builder
        self._model = model
        self._c = ui_builder.get_ui_components()

    def apply_algorithm_visibility(self) -> None:
        """切换第四图：核回归为空白 Matplotlib（待训练回填曲线），高斯过程为占位标签。"""
        kernel_w = self._c["chart_train_curve"]
        ph = self._c["train_gp_curve_placeholder"]

        if self._model.algorithm == "kernel":
            ph.hide()
            kernel_w.show()
            kernel_w.set_chart_title("训练曲线（核回归：σ — MSE）")
            kernel_w.reset()
        else:
            kernel_w.hide()
            ph.show()

    def redraw_scatters_from_training_model(self) -> None:
        """
        依据已载入数据集更新前三张散点：纵轴依次为 K（最大直径）、B（初始状态常数）、
        C（时间常数）；点大小映射含铝量。
        """
        recs = self._model.records
        if not recs:
            self.clear_training_plots()
            return
        eq = np.array([r.equivalent_kg_tnt for r in recs], dtype=float)
        al = np.array([r.al_percent for r in recs], dtype=float)
        kk = np.array([r.K for r in recs], dtype=float)
        bb = np.array([r.B for r in recs], dtype=float)
        cc = np.array([r.C for r in recs], dtype=float)
        self._c["scatter_max_chart"].update_data(eq, kk, al)
        self._c["scatter_init_chart"].update_data(eq, bb, al)
        self._c["scatter_tau_chart"].update_data(eq, cc, al)
        self.apply_algorithm_visibility()

    def clear_training_plots(self) -> None:
        """散点与第四图复位为空白（无刻板示意数据）。"""
        self._c["scatter_max_chart"].reset()
        self._c["scatter_init_chart"].reset()
        self._c["scatter_tau_chart"].reset()
        self._c["chart_train_curve"].reset()
        self.apply_algorithm_visibility()
