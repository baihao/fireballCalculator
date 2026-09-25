#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工程计算标签页 — 侧栏输入，主区结果。"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from .ui_widgets.engineering_tab_ui import EngineeringTabUI
from .utils.symbols import CHI_R_PLAIN, EPSILON, KAPPA_A_PLAIN, SIGMA
from .utils.image_formula_engineering import (
    EngineeringEstimateInputs,
    compute_engineering_estimate,
    format_engineering_result,
    parse_al_fraction,
)


class EngineeringTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ui = EngineeringTabUI()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._ui.create_main_widget())

        c = self._ui.get_ui_components()
        self.calc_btn = c["calc_btn"]
        self.result_panel = c["result_panel"]
        self._sidebar_widget = self._ui.create_sidebar_widget()
        self.calc_btn.clicked.connect(self._on_calculate)

    def get_sidebar_widget(self) -> QWidget:
        return self._sidebar_widget

    @staticmethod
    def _parse_float(text: str, name: str) -> float:
        t = (text or "").strip()
        if not t:
            raise ValueError(f"{name} 不能为空")
        return float(t)

    def _on_calculate(self) -> None:
        c = self._ui.get_ui_components()
        try:
            w_kg = self._parse_float(c["eq_kg"].text(), "TNT 当量 W")
            al_pct = self._parse_float(c["al_percent"].text(), "含铝率")
            t_amb_c = self._parse_float(c["t_amb_c"].text(), "环境温度")
            chi_r = self._parse_float(c["chi_r"].text(), CHI_R_PLAIN)
            epsilon = self._parse_float(c["epsilon"].text(), EPSILON)
            kappa_a = self._parse_float(c["kappa_a"].text(), KAPPA_A_PLAIN)
            sigma = self._parse_float(c["sigma"].text(), SIGMA)
            if w_kg <= 0:
                raise ValueError("W 必须大于 0")
            if al_pct < 0:
                raise ValueError("含铝率不能为负")

            inputs = EngineeringEstimateInputs(
                equivalent_kg=w_kg,
                al_fraction=parse_al_fraction(al_pct),
                t_amb_k=t_amb_c + 273.15,
                chi_r=chi_r,
                epsilon=epsilon,
                kappa_a=kappa_a,
                sigma=sigma,
            )
            result = compute_engineering_estimate(inputs)
            self.result_panel.setHtml(format_engineering_result(inputs, result))
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
