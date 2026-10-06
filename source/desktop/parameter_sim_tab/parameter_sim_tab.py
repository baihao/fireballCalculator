#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数仿真标签页 — K/B/C + 火球最大温度，图表同参数预测，仅显示仿真结果。"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from model_tab.controllers import ModelTabChartController
from model_tab.utils.calculator import (
    DEFAULT_PARAMETER_SIMULATION_DURATION_MS,
    build_prediction_bundle,
)
from model_tab.utils.formula_reference import (
    build_formula_reference_from_prediction,
    build_formula_reference_text,
)

from .ui_widgets import ParameterSimTabUI

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from fireball_radius_calculator import FireballCalculator

DEFAULT_PEAK_TEMPERATURE_K = 1600.0


class ParameterSimTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ui_builder = ParameterSimTabUI()
        self.ui_builder.create_main_layout(self)
        self.ui_components = self.ui_builder.get_ui_components()

        self.chart_controller = ModelTabChartController(self.ui_builder)
        self.fireball_calculator = FireballCalculator()
        self._simulation_succeeded = False
        self._simulation_running = False
        self._connections_wired = False
        self.prediction_data = None
        self._sidebar_widget = None

        self._bind_ui()
        self.chart_controller.reset()

    def _bind_ui(self) -> None:
        c = self.ui_components
        self.modeling_status = c["modeling_status"]
        self.formula_reference = c["formula_reference"]
        for key in (
            "p_k",
            "p_b",
            "p_c",
            "p_peak_temp_k",
            "p_env_temp",
            "p_env_humidity",
            "p_env_pressure",
            "p_step",
            "p_duration",
            "predict_btn",
            "export_btn",
        ):
            if key in c:
                setattr(self, key, c[key])

    def get_sidebar_widget(self) -> QWidget:
        if self._sidebar_widget is None:
            self._sidebar_widget = self.ui_builder.create_sidebar_widget()
            self.ui_components.update(self.ui_builder.get_ui_components())
            self._bind_ui()
            self._wire_connections()
            self._refresh_formula_reference()
            self._update_predict_btn_state()
        return self._sidebar_widget

    def _wire_connections(self) -> None:
        if self._connections_wired:
            return
        if not hasattr(self, "predict_btn"):
            return
        self.predict_btn.clicked.connect(self.start_simulation)
        self.export_btn.clicked.connect(self.export_results)
        for name in (
            "p_k",
            "p_b",
            "p_c",
            "p_peak_temp_k",
            "p_env_temp",
            "p_env_humidity",
            "p_env_pressure",
            "p_step",
            "p_duration",
        ):
            getattr(self, name).textChanged.connect(self._on_param_changed)
        self._connections_wired = True

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _can_start(self) -> bool:
        if not all(
            hasattr(self, name)
            for name in (
                "p_k",
                "p_b",
                "p_c",
                "p_peak_temp_k",
                "p_env_temp",
                "p_env_humidity",
                "p_env_pressure",
                "p_step",
                "p_duration",
            )
        ):
            return False
        k = self._parse_float(self.p_k.text())
        b = self._parse_float(self.p_b.text())
        c = self._parse_float(self.p_c.text())
        peak = self._parse_float(self.p_peak_temp_k.text())
        fields = (
            self.p_env_temp,
            self.p_env_humidity,
            self.p_env_pressure,
            self.p_step,
            self.p_duration,
        )
        if any(self._parse_float(w.text()) is None for w in fields):
            return False
        return (
            k is not None
            and k > 0
            and b is not None
            and b > 0
            and c is not None
            and c > 0
            and peak is not None
            and peak > 0
        )

    def _update_predict_btn_state(self) -> None:
        if not hasattr(self, "predict_btn"):
            return
        if self._simulation_running:
            return
        self.predict_btn.setEnabled(self._can_start())

    def _on_param_changed(self) -> None:
        self._update_predict_btn_state()
        self._refresh_formula_reference()

    def _collect(self, attr: str) -> Optional[float]:
        if not hasattr(self, attr):
            return None
        return self._parse_float(getattr(self, attr).text())

    def _refresh_formula_reference(self) -> None:
        if not hasattr(self, "formula_reference"):
            return
        if self.prediction_data is not None:
            text = build_formula_reference_from_prediction(
                self.prediction_data,
                al_percent=30.0,
                kbc_source="explicit_kbc",
            )
        else:
            text = build_formula_reference_text(
                is_equivalent_mode=False,
                k=self._collect("p_k"),
                b=self._collect("p_b"),
                c=self._collect("p_c"),
                env_temp=self._collect("p_env_temp"),
                env_humidity=self._collect("p_env_humidity"),
                env_pressure=self._collect("p_env_pressure"),
                duration=self._collect("p_duration"),
                preview_equivalent_kg=None,
                peak_temperature_k=self._collect("p_peak_temp_k"),
            )
        self.formula_reference.setPlainText(text)

    def start_simulation(self) -> None:
        if not self._can_start():
            QMessageBox.warning(self, "计算", "请填写完整且合法的仿真参数后再开始计算。")
            return
        if self._simulation_running:
            return

        self._simulation_succeeded = False
        self.export_btn.setEnabled(False)
        try:
            self.modeling_status.setText("正在计算…")
            self._simulation_running = True
            self.predict_btn.setEnabled(False)

            k_value = float(self.p_k.text())
            b_value = float(self.p_b.text())
            c_value = float(self.p_c.text())
            peak_k = float(self.p_peak_temp_k.text())
            duration = (
                float(self.p_duration.text())
                if self.p_duration.text()
                else DEFAULT_PARAMETER_SIMULATION_DURATION_MS
            )
            env_temp = float(self.p_env_temp.text()) if self.p_env_temp.text() else 24.0
            env_humidity = (
                float(self.p_env_humidity.text()) if self.p_env_humidity.text() else 48.0
            )
            env_pressure = (
                float(self.p_env_pressure.text()) if self.p_env_pressure.text() else 2987.87
            )

            time_points = int(duration / 1.0) + 1
            t_ms = np.linspace(0.0, duration, time_points)
            material_name = "30%Al/Rubber"
            kbc = (k_value, b_value, c_value)

            bundle = build_prediction_bundle(
                t_ms=t_ms,
                duration_ms=float(duration),
                equivalent=1.0,
                material_name=material_name,
                env_temp=env_temp,
                env_humidity=env_humidity,
                env_pressure=env_pressure,
                calculator=self.fireball_calculator,
                use_explicit_kbc=True,
                kbc=kbc,
                training_equivalent=None,
                training_temperature_data=None,
                al_fraction=0.30,
                parameter_simulation=True,
                peak_temperature_k=peak_k,
            )
            heat_series = bundle.pop("_heat_flux_series_chart")
            self.prediction_data = bundle

            self.chart_controller.update_diameter(
                self.prediction_data["time_ms"], self.prediction_data["diameter_data"]
            )
            self.chart_controller.update_expansion_velocity(
                self.prediction_data["time_ms"], self.prediction_data["diameter_data"]
            )
            self.chart_controller.update_heat_flux(
                self.prediction_data["time_ms"], heat_series
            )
            rad = self.prediction_data["heat_radiation_data"]
            self.chart_controller.update_heat_radiation(
                rad["distances"], rad["heat_radiation"]
            )

            self._refresh_formula_reference()
            self.modeling_status.setText("计算完成")
            self._simulation_succeeded = True
        except Exception as e:
            self.modeling_status.setText("计算失败")
            self._simulation_succeeded = False
            self.prediction_data = None
            QMessageBox.critical(self, "错误", f"参数仿真失败:\n{e}")
        finally:
            self._simulation_running = False
            self._update_predict_btn_state()
            self.export_btn.setEnabled(self._simulation_succeeded)

    def export_results(self) -> None:
        if not self._simulation_succeeded or self.prediction_data is None:
            QMessageBox.warning(self, "警告", "请先成功完成一次仿真后再导出。")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存参数仿真结果",
            f"parameter_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)",
        )
        if not file_path:
            return
        try:
            pd = self.prediction_data
            t_ms = np.asarray(pd["time_ms"])
            d_m = np.asarray(pd["diameter_data"])
            v = np.asarray(pd.get("expansion_velocity_data", np.gradient(d_m, t_ms)))
            t_k = np.asarray(pd["temperature_data"])
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["参数仿真结果"])
                writer.writerow(["K", pd["kbc_display"][0]])
                writer.writerow(["B", pd["kbc_display"][1]])
                writer.writerow(["C", pd["kbc_display"][2]])
                writer.writerow(["火球最大温度_K", pd.get("temperature_peak_k")])
                writer.writerow(["仿真时长_ms", pd.get("duration")])
                writer.writerow([])
                writer.writerow(["时间_ms", "直径_m", "膨胀速度_m_per_ms", "温度_K"])
                for i in range(len(t_ms)):
                    writer.writerow([t_ms[i], d_m[i], v[i], t_k[i]])
            QMessageBox.information(self, "成功", f"已导出到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
