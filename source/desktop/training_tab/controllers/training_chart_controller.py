#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练 — 图表控制器（与画布、数据绑定相关）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..training_dataset_model import TrainingDatasetModel
from ..ui_widgets.training_tab_ui import TrainingTabUI


# 三条预测曲线（不同含铝档）
CURVE_COLORS_DEFAULT: tuple[str, str, str] = ("#f97316", "#a78bfa", "#34d399")


class TrainingChartController:
    """封装三张散点图的刷新及核回归预测曲线叠加。"""

    def __init__(self, ui_builder: TrainingTabUI, model: TrainingDatasetModel) -> None:
        self._ui = ui_builder
        self._model = model
        self._c = ui_builder.get_ui_components()

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

    def redraw_scatters_with_prediction_curves(
        self,
        equiv_grid: np.ndarray,
        al_levels: np.ndarray,
        K_pred: np.ndarray,
        B_pred: np.ndarray,
        C_pred: np.ndarray,
        *,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        """
        在散点之上叠加核回归预测曲线：行 = 含铝档，列 = 当量采样点；
        ``K_pred`` / ``B_pred`` / ``C_pred`` 形状 ``(len(al_levels), len(equiv_grid))``。
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

        cols = list(colors) if colors is not None else list(CURVE_COLORS_DEFAULT)
        n_al = int(al_levels.shape[0])
        while len(cols) < n_al:
            cols.append(CURVE_COLORS_DEFAULT[len(cols) % len(CURVE_COLORS_DEFAULT)])
        cols = cols[:n_al]

        def _curves_for(Y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
            out: list[tuple[np.ndarray, np.ndarray, str]] = []
            for i in range(n_al):
                out.append((equiv_grid, Y[i], cols[i]))
            return out

        ck = _curves_for(K_pred)
        cb = _curves_for(B_pred)
        ccv = _curves_for(C_pred)

        curve_legend_title = "核回归｜横轴炸药当量"
        curve_labels = [f"含铝量 {float(al_levels[i]):g} %" for i in range(n_al)]

        kw = {
            "curve_legend_labels": curve_labels,
            "curve_legend_title": curve_legend_title,
        }
        self._c["scatter_max_chart"].update_data(eq, kk, al, curves=ck, **kw)
        self._c["scatter_init_chart"].update_data(eq, bb, al, curves=cb, **kw)
        self._c["scatter_tau_chart"].update_data(eq, cc, al, curves=ccv, **kw)

    def clear_training_plots(self) -> None:
        """散点复位为空白（无刻板示意数据）。"""
        self._c["scatter_max_chart"].reset()
        self._c["scatter_init_chart"].reset()
        self._c["scatter_tau_chart"].reset()
