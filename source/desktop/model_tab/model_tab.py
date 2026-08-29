#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建模与预测模块标签页
"""

import numpy as np
import sys
import os
import csv
from datetime import datetime
from typing import Optional, Tuple
from PySide6.QtWidgets import QWidget, QMessageBox, QFileDialog
from .ui_widgets.model_tab_ui import ModelTabUI
from .controllers import ModelTabChartController, ModelController
from .utils.calculator import build_prediction_bundle, default_simulation_duration_ms, REFERENCE_EQUIVALENT_KG
from .utils.formula_reference import (
    build_formula_reference_from_prediction,
    build_formula_reference_text,
)
from .utils.simulation_log import build_simulation_log_lines

# 添加路径以导入计算器
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from fireball_radius_calculator import FireballCalculator


class ModelTab(QWidget):
    """建模与预测模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化UI构建器并创建界面
        self.ui_builder = ModelTabUI()
        self.ui_builder.create_main_layout(self)
        self.ui_components = self.ui_builder.get_ui_components()
        
        # 初始化控制器
        self.chart_controller = ModelTabChartController(self.ui_builder)
        self.fireball_calculator = FireballCalculator()
        self.model_ctrl = ModelController(self.fireball_calculator)
        self._model_import_ok = False
        self._simulation_succeeded = False
        self._duration_user_edited = False
        self._syncing_duration_widget = False
        self._eq_duration_sync_wired = False
        self._simulation_running = False
        
        # 设置UI组件引用（向后兼容）
        self._setup_ui_component_references()
        
        # 初始化图表
        self.init_empty_charts()
        
        self.setup_connections()
        self._refresh_formula_reference()
    
    def _collect_sidebar_float(self, attr: str) -> Optional[float]:
        widget = getattr(self, attr, None)
        if widget is None:
            return None
        return self._parse_float_field(widget.text())

    def _refresh_formula_reference(self) -> None:
        if not hasattr(self, "formula_reference"):
            return
        if hasattr(self, "prediction_data") and self.prediction_data is not None:
            al = self._collect_sidebar_float("p_al")
            if al is None:
                al = 30.0
            text = build_formula_reference_from_prediction(
                self.prediction_data,
                is_equivalent_mode=self._is_equivalent_sim_mode(),
                al_percent=float(al),
                kbc_source=self.model_ctrl.last_kbc_source,
            )
        else:
            equivalent = self._collect_sidebar_float("p_eq")
            std_eq = self.model_ctrl.training_equivalent
            if std_eq is None and equivalent is not None and hasattr(self, "p_al"):
                al = self._collect_sidebar_float("p_al")
                if al is not None:
                    material = self.model_ctrl.get_material_by_al_content(al)
                    std_eq = self.fireball_calculator.get_standard_equivalent(material)
            text = build_formula_reference_text(
                is_equivalent_mode=self._is_equivalent_sim_mode(),
                equivalent=equivalent,
                al_percent=self._collect_sidebar_float("p_al"),
                k=self._collect_sidebar_float("p_k"),
                b=self._collect_sidebar_float("p_b"),
                c=self._collect_sidebar_float("p_c"),
                env_temp=self._collect_sidebar_float("p_env_temp"),
                env_humidity=self._collect_sidebar_float("p_env_humidity"),
                env_pressure=self._collect_sidebar_float("p_env_pressure"),
                duration=self._collect_sidebar_float("p_duration"),
                kbc_source=self.model_ctrl.last_kbc_source,
                standard_equivalent=std_eq,
            )
        self.formula_reference.setPlainText(text)

    def _setup_ui_component_references(self):
        """设置UI组件引用（向后兼容）"""
        # 状态显示控件
        self.modeling_status = self.ui_components['modeling_status']
        
        if "model_select_btn" in self.ui_components:
            self.model_select_btn = self.ui_components["model_select_btn"]
        if "model_import_summary" in self.ui_components:
            self.model_import_summary = self.ui_components["model_import_summary"]

        # 按钮控件（可能尚未创建侧边栏）
        if "predict_btn" in self.ui_components:
            self.predict_btn = self.ui_components["predict_btn"]
        if "export_btn" in self.ui_components:
            self.export_btn = self.ui_components["export_btn"]

        # 参数输入控件
        if 'p_eq' in self.ui_components:
            self.p_eq = self.ui_components['p_eq']
        if 'p_al' in self.ui_components:
            self.p_al = self.ui_components['p_al']
        if 'p_step' in self.ui_components:
            self.p_step = self.ui_components['p_step']
        if 'p_duration' in self.ui_components:
            self.p_duration = self.ui_components['p_duration']
        if 'p_env_temp' in self.ui_components:
            self.p_env_temp = self.ui_components['p_env_temp']
        if 'p_env_humidity' in self.ui_components:
            self.p_env_humidity = self.ui_components['p_env_humidity']
        if 'p_env_pressure' in self.ui_components:
            self.p_env_pressure = self.ui_components['p_env_pressure']
        if 'p_k' in self.ui_components:
            self.p_k = self.ui_components['p_k']
        if 'p_b' in self.ui_components:
            self.p_b = self.ui_components['p_b']
        if 'p_c' in self.ui_components:
            self.p_c = self.ui_components['p_c']
        if 'sim_mode_equivalent' in self.ui_components:
            self.sim_mode_equivalent = self.ui_components['sim_mode_equivalent']
        if 'sim_mode_parameter' in self.ui_components:
            self.sim_mode_parameter = self.ui_components['sim_mode_parameter']
        if 'params_form_layout' in self.ui_components:
            self.params_form_layout = self.ui_components['params_form_layout']
        if 'param_form_row_labels' in self.ui_components:
            self.param_form_row_labels = self.ui_components['param_form_row_labels']
        
        if "params_scroll_area" in self.ui_components:
            self.params_scroll_area = self.ui_components["params_scroll_area"]
        if "simulation_log" in self.ui_components:
            self.simulation_log = self.ui_components["simulation_log"]
        if "formula_reference" in self.ui_components:
            self.formula_reference = self.ui_components["formula_reference"]

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _append_simulation_log(self, line: str) -> None:
        if not hasattr(self, "simulation_log"):
            return
        self.simulation_log.appendPlainText(f"[{self._now()}] {line}")

    def _clear_simulation_log(self) -> None:
        if hasattr(self, "simulation_log"):
            self.simulation_log.clear()

    def _write_simulation_summary_log(
        self,
        al_content: float,
        *,
        failed: bool = False,
        error_message: str = "",
    ) -> None:
        if failed:
            self._append_simulation_log(f"仿真失败 — {error_message}")
            return
        if not hasattr(self, "prediction_data") or self.prediction_data is None:
            return
        self._append_simulation_log("仿真完成，关键指标如下：")
        for block_line in build_simulation_log_lines(
            self.prediction_data,
            al_percent=float(al_content),
            kbc_source_label=self.model_ctrl.last_kbc_source,
        ):
            if block_line == "":
                self.simulation_log.appendPlainText("")
            else:
                self.simulation_log.appendPlainText(block_line)

    def _is_equivalent_sim_mode(self) -> bool:
        return not hasattr(self, "sim_mode_parameter") or not self.sim_mode_parameter.isChecked()

    @staticmethod
    def _parse_float_field(text: str) -> Optional[float]:
        cleaned = text.strip() if text else ""
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _common_sim_params_valid(self) -> bool:
        fields = (
            getattr(self, "p_env_temp", None),
            getattr(self, "p_env_humidity", None),
            getattr(self, "p_env_pressure", None),
            getattr(self, "p_step", None),
            getattr(self, "p_duration", None),
        )
        for widget in fields:
            if widget is None or self._parse_float_field(widget.text()) is None:
                return False
        return True

    def _equivalent_mode_params_valid(self) -> bool:
        if not hasattr(self, "p_eq") or not hasattr(self, "p_al"):
            return False
        eq = self._parse_float_field(self.p_eq.text())
        al = self._parse_float_field(self.p_al.text())
        return eq is not None and eq > 0 and al is not None and al >= 0

    def _parameter_mode_params_valid(self) -> bool:
        if not all(hasattr(self, name) for name in ("p_k", "p_b", "p_c")):
            return False
        k = self._parse_float_field(self.p_k.text())
        b = self._parse_float_field(self.p_b.text())
        c = self._parse_float_field(self.p_c.text())
        return (
            k is not None and k > 0
            and b is not None and b > 0
            and c is not None and c > 0
            and self._common_sim_params_valid()
        )

    def _can_start_simulation(self) -> bool:
        if self._is_equivalent_sim_mode():
            return self._model_import_ok and self._equivalent_mode_params_valid() and self._common_sim_params_valid()
        return self._parameter_mode_params_valid()

    def _update_predict_btn_state(self) -> None:
        if not hasattr(self, "predict_btn"):
            return
        if not self.predict_btn.isEnabled() and getattr(self, "_simulation_running", False):
            return
        self.predict_btn.setEnabled(self._can_start_simulation())

    def _apply_sim_mode_visibility(self) -> None:
        equivalent_mode = self._is_equivalent_sim_mode()
        if hasattr(self, "params_form_layout") and hasattr(self, "param_form_row_labels"):
            for key in ("eq", "al"):
                self.params_form_layout.setRowVisible(self.param_form_row_labels[key], equivalent_mode)
            for key in ("k", "b", "c"):
                self.params_form_layout.setRowVisible(self.param_form_row_labels[key], not equivalent_mode)
        self._update_predict_btn_state()

    def _on_sim_mode_changed(self) -> None:
        self._apply_sim_mode_visibility()
        self._refresh_formula_reference()

    def _on_sim_param_changed(self) -> None:
        self._update_predict_btn_state()
        self._refresh_formula_reference()

    def _parse_equivalent_kg(self) -> float:
        try:
            text = self.p_eq.text().strip() if hasattr(self, "p_eq") else ""
            return float(text) if text else 10.0
        except ValueError:
            return 10.0

    def _sync_simulation_duration_from_equivalent(self) -> None:
        """按当量更新侧栏「仿真时长」；若用户已手动改过则不再覆盖。"""
        if not self._is_equivalent_sim_mode():
            return
        if self._duration_user_edited or not hasattr(self, "p_duration"):
            return
        duration_ms = default_simulation_duration_ms(self._parse_equivalent_kg())
        self._syncing_duration_widget = True
        self.p_duration.setText(f"{duration_ms:.6g}")
        self._syncing_duration_widget = False

    def _on_equivalent_changed(self) -> None:
        self._sync_simulation_duration_from_equivalent()

    def _on_duration_edited_by_user(self) -> None:
        if self._syncing_duration_widget:
            return
        self._duration_user_edited = True

    def _wire_equivalent_duration_sync(self) -> None:
        if self._eq_duration_sync_wired:
            return
        if not hasattr(self, "p_eq") or not hasattr(self, "p_duration"):
            return
        self.p_eq.editingFinished.connect(self._on_equivalent_changed)
        self.p_duration.editingFinished.connect(self._on_duration_edited_by_user)
        self._eq_duration_sync_wired = True
        self._sync_simulation_duration_from_equivalent()
    
    def init_empty_charts(self):
        """初始化空图表，显示默认占位内容"""
        try:
            self.chart_controller.reset()
        except Exception as e:
            print(f"初始化空图表失败: {e}")
    
    def setup_connections(self):
        """设置信号连接"""
        try:
            if hasattr(self, "model_select_btn"):
                self.model_select_btn.clicked.connect(self.select_model_folder)
            if hasattr(self, "predict_btn"):
                self.predict_btn.clicked.connect(self.start_prediction)
            if hasattr(self, "export_btn"):
                self.export_btn.clicked.connect(self.export_results)
            if hasattr(self, "sim_mode_equivalent"):
                self.sim_mode_equivalent.toggled.connect(self._on_sim_mode_changed)
            if hasattr(self, "sim_mode_parameter"):
                self.sim_mode_parameter.toggled.connect(self._on_sim_mode_changed)
            for widget_name in (
                "p_eq", "p_al", "p_k", "p_b", "p_c",
                "p_env_temp", "p_env_humidity", "p_env_pressure", "p_step", "p_duration",
            ):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.textChanged.connect(self._on_sim_param_changed)
            self._wire_equivalent_duration_sync()
            self._apply_sim_mode_visibility()
        except Exception:
            pass
        
    def start_prediction(self):
        """开始预测"""
        if not self._can_start_simulation():
            if self._is_equivalent_sim_mode() and not self._model_import_ok:
                QMessageBox.warning(
                    self,
                    "仿真",
                    "请先通过「选择模型」导入合法模型目录：\n"
                    "需包含完整核岭回归 artefact（manifest.json 与三套 kbc_krr_*.joblib），\n"
                    "或至少一个可解析的火球实验 JSON。",
                )
            else:
                QMessageBox.warning(self, "计算", "请填写完整且合法的仿真参数后再开始计算。")
            return
        if not self.predict_btn.isEnabled():
            print("⚠️ 预测正在进行中，请勿重复点击...")
            return

        self._simulation_succeeded = False
        if hasattr(self, "export_btn"):
            self.export_btn.setEnabled(False)

        al_content = 30.0
        try:
            print("🔥 开始预测...")
            self.modeling_status.setText("正在计算预测结果...")
            self._simulation_running = True
            self.predict_btn.setEnabled(False)

            step = float(self.p_step.text()) if self.p_step.text() else 1.0
            env_temp = float(self.p_env_temp.text()) if self.p_env_temp.text() else 24.0
            env_humidity = float(self.p_env_humidity.text()) if self.p_env_humidity.text() else 48.0
            env_pressure = float(self.p_env_pressure.text()) if self.p_env_pressure.text() else 2987.87

            if self._is_equivalent_sim_mode():
                equivalent = self._parse_equivalent_kg()
                al_content = float(self.p_al.text()) if self.p_al.text() else 30.0
                self._sync_simulation_duration_from_equivalent()
                duration = float(self.p_duration.text()) if self.p_duration.text() else default_simulation_duration_ms(equivalent)
                material_name = self.model_ctrl.get_material_by_al_content(al_content)
                print(f"预测参数: 当量={equivalent}, 含铝量={al_content}%, 步长={step}, 时长={duration}ms")
                print(f"环境参数: 温度={env_temp}°C, 湿度={env_humidity}%, 气压={env_pressure}Pa")
                print(f"选择材料: {material_name}")
                self.generate_prediction_curves(
                    material_name, duration, equivalent, al_content, env_temp, env_humidity, env_pressure
                )
            else:
                k_value = float(self.p_k.text())
                b_value = float(self.p_b.text())
                c_value = float(self.p_c.text())
                duration = float(self.p_duration.text())
                equivalent = REFERENCE_EQUIVALENT_KG
                al_content = 30.0
                material_name = self.model_ctrl.get_material_by_al_content(al_content)
                print(
                    f"参数仿真: K={k_value}, B={b_value}, C={c_value}, "
                    f"步长={step}, 时长={duration}ms"
                )
                print(f"环境参数: 温度={env_temp}°C, 湿度={env_humidity}%, 气压={env_pressure}Pa")
                self.generate_prediction_curves(
                    material_name,
                    duration,
                    equivalent,
                    al_content,
                    env_temp,
                    env_humidity,
                    env_pressure,
                    explicit_kbc=(k_value, b_value, c_value),
                )

            self._write_simulation_summary_log(al_content)

            self.modeling_status.setText("预测完成")
            self._simulation_succeeded = True
            print("✅ 预测完成！")

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            self.modeling_status.setText("预测失败")
            self._simulation_succeeded = False
            self._write_simulation_summary_log(al_content, failed=True, error_message=str(e))
            QMessageBox.critical(self, "错误", f"预测失败:\n{str(e)}")
        finally:
            self._simulation_running = False
            self._update_predict_btn_state()
            if hasattr(self, "export_btn"):
                self.export_btn.setEnabled(self._simulation_succeeded)
    
    def generate_prediction_curves(
        self,
        material_name: str,
        duration: float,
        equivalent: float,
        al_content: float,
        env_temp: float = 24.0,
        env_humidity: float = 48.0,
        env_pressure: float = 2987.87,
        explicit_kbc: Optional[Tuple[float, float, float]] = None,
    ):
        """使用 ``ModelController`` 解析 K/B/C，``utils.calculator`` 组装直径、膨胀速度、热通量与累积辐射。"""
        print(f"生成 {material_name} 材料的预测曲线...")
        time_points = int(duration / 1.0) + 1
        t_ms = np.linspace(0, duration, time_points)

        if explicit_kbc is not None:
            kbc_tuple = tuple(float(v) for v in explicit_kbc)
            use_explicit = True
            self.model_ctrl.last_sim_kbc = kbc_tuple
            self.model_ctrl.last_kbc_source = "explicit_kbc"
            print(f"✓ 用户输入 K,B,C → K={kbc_tuple[0]:g}, B={kbc_tuple[1]:g}, C={kbc_tuple[2]:g}")
        else:
            _, kbc_tuple, use_explicit = self.model_ctrl.resolve_kbc_for_simulation(
                float(equivalent), float(al_content), material_name
            )

        bundle = build_prediction_bundle(
            t_ms=t_ms,
            duration_ms=float(duration),
            equivalent=float(equivalent),
            material_name=material_name,
            env_temp=env_temp,
            env_humidity=env_humidity,
            env_pressure=env_pressure,
            calculator=self.fireball_calculator,
            use_explicit_kbc=use_explicit,
            kbc=kbc_tuple if use_explicit else None,
            training_equivalent=self.model_ctrl.training_equivalent,
            training_temperature_data=self.model_ctrl.training_temperature_data,
        )
        heat_series = bundle.pop("_heat_flux_series_chart")
        self.prediction_data = bundle

        self.chart_controller.update_diameter(self.prediction_data["time_ms"], self.prediction_data["diameter_data"])
        self.chart_controller.update_expansion_velocity(
            self.prediction_data["time_ms"], self.prediction_data["diameter_data"]
        )
        self.chart_controller.update_heat_flux(self.prediction_data["time_ms"], heat_series)
        rad = self.prediction_data["heat_radiation_data"]
        self.chart_controller.update_heat_radiation(rad["distances"], rad["heat_radiation"])
        if explicit_kbc is not None:
            self.prediction_data["simulation_mode"] = "parameter"
        self._refresh_formula_reference()
        print("✅ 所有预测曲线生成完成！")

    def get_material_by_al_content(self, al_content: float) -> str:
        return self.model_ctrl.get_material_by_al_content(al_content)
    
    def select_model_folder(self) -> None:
        """选择模型所在文件夹，并解析目录内 JSON / 核岭回归 artefact。"""
        try:
            start = self.model_ctrl.model_folder_path or ""
            path = QFileDialog.getExistingDirectory(
                self,
                "选择模型文件所在文件夹",
                start,
                QFileDialog.Option.ShowDirsOnly,
            )
            if not path:
                return
            self._load_model_folder(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"选择模型目录失败:\n{str(e)}")

    def select_training_files(self) -> None:
        """与文件菜单「导入训练文件」一致，等同于选择模型目录。"""
        self.select_model_folder()

    def _load_model_folder(self, folder: str) -> None:
        self.prediction_data = None
        self._simulation_succeeded = False
        self._clear_simulation_log()
        self._append_simulation_log("已切换模型目录，请重新「开始计算」以更新图表与日志。")
        if hasattr(self, "export_btn"):
            self.export_btn.setEnabled(False)

        result = self.model_ctrl.load_folder(folder, parent=self)
        self._model_import_ok = result.can_run_simulation
        self._update_predict_btn_state()
        if hasattr(self, "model_import_summary"):
            self.model_import_summary.setPlainText(result.summary_text)
        self.model_ctrl.apply_first_params_to_widgets(self.model_ctrl.last_first_params, self)
        self._duration_user_edited = False
        self._sync_simulation_duration_from_equivalent()
        self._refresh_formula_reference()

        if result.can_run_simulation:
            print(
                f"📁 模型目录已载入：{self.model_ctrl.model_folder_path}"
                f"（有效 JSON {result.applied_json_count} 个；"
                f"可在侧栏填写当量、含铝量后点击「开始计算」）"
            )
        else:
            print(f"📁 已选择目录：{self.model_ctrl.model_folder_path}（当前不可启动仿真）")

    def export_results(self):
        """导出预测结果到CSV文件"""
        try:
            if not hasattr(self, "prediction_data") or self.prediction_data is None:
                QMessageBox.warning(self, "警告", "暂无可导出数据，请先成功完成一次仿真。")
                return
            if not getattr(self, "_simulation_succeeded", False):
                QMessageBox.warning(self, "警告", "请先成功完成一次仿真后再导出。")
                return
            
            print("📊 开始导出预测结果...")
            
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存预测结果", 
                f"fireball_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV文件 (*.csv)"
            )
            
            if not file_path:
                return
            
            # 获取预测参数
            equivalent = self._parse_equivalent_kg()
            al_content = float(self.p_al.text()) if self.p_al.text() else 30.0
            step = float(self.p_step.text()) if self.p_step.text() else 1.0
            duration = float(self.p_duration.text()) if self.p_duration.text() else default_simulation_duration_ms(equivalent)
            
            # 获取环境参数
            env_temp = float(self.p_env_temp.text()) if self.p_env_temp.text() else 24.0
            env_humidity = float(self.p_env_humidity.text()) if self.p_env_humidity.text() else 48.0
            env_pressure = float(self.p_env_pressure.text()) if self.p_env_pressure.text() else 2987.87
            
            # 写入CSV文件
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入预测参数
                writer.writerow(['# 火球爆炸预测结果'])
                writer.writerow(['# 导出时间:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([''])
                writer.writerow(['# 预测参数'])
                writer.writerow(['参数名称', '数值', '单位'])
                writer.writerow(['爆炸当量', equivalent, 'kg TNT'])
                writer.writerow(['含铝量', al_content, '%'])
                writer.writerow(['仿真步长', step, 'ms'])
                writer.writerow(['仿真时长', duration, 'ms'])
                writer.writerow(['材料类型', self.prediction_data['material_name'], ''])
                writer.writerow([''])
                
                # 写入时间序列数据
                writer.writerow(['# 时间序列数据'])
                writer.writerow(['时间(ms)', '时间(s)', '火球直径(m)', '膨胀速度(m/ms)'])
                
                t_ms = self.prediction_data['time_ms']
                t_s = self.prediction_data['time_s']
                diameter = self.prediction_data['diameter_data']
                velocity = self.prediction_data.get('expansion_velocity_data')
                if velocity is None:
                    velocity = np.gradient(diameter, t_ms)
                
                for i in range(len(t_ms)):
                    writer.writerow([
                        f"{t_ms[i]:.3f}",
                        f"{t_s[i]:.6f}",
                        f"{diameter[i]:.6f}",
                        f"{velocity[i]:.6f}"
                    ])
                
                writer.writerow([''])
                
                # 写入热通量数据
                writer.writerow(['# 热通量随时间变化数据 (W/m²)'])
                header = ['时间(ms)']
                for dist in ['6.0', '7.0', '8.0', '9.0', '10.0']:
                    header.append(f'热通量_x={dist}m')
                writer.writerow(header)
                
                for i in range(len(t_ms)):
                    row = [f"{t_ms[i]:.3f}"]
                    for dist in ['6.0', '7.0', '8.0', '9.0', '10.0']:
                        if dist in self.prediction_data['heat_flux_data']:
                            row.append(f"{self.prediction_data['heat_flux_data'][dist][i]:.2f}")
                        else:
                            row.append("")
                    writer.writerow(row)
                
                writer.writerow([''])
                
                # 写入累积热辐射数据
                writer.writerow(['# 累积热辐射量随距离分布数据'])
                writer.writerow(['距离(m)', '热辐射量(kJ/m²)'])
                
                distances = self.prediction_data['heat_radiation_data']['distances']
                heat_radiation = self.prediction_data['heat_radiation_data']['heat_radiation']
                
                for i in range(len(distances)):
                    writer.writerow([
                        f"{distances[i]:.3f}",
                        f"{heat_radiation[i]:.2f}"  # kJ/m²单位，使用2位小数
                    ])
                
                writer.writerow([''])
                
                # 写入环境参数
                writer.writerow(['# 环境参数'])
                writer.writerow(['参数名称', '数值', '单位'])
                writer.writerow(['环境温度', env_temp, '°C'])
                writer.writerow(['相对湿度', env_humidity, '%'])
                writer.writerow(['水饱和气压', env_pressure, 'Pa'])
                writer.writerow(['火球表面比辐射率', '0.9', ''])
                writer.writerow(['斯蒂芬-波尔茨曼常数', '5.67e-8', 'W/(m²·K⁴)'])
            
            print(f"✅ 预测结果已导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"预测结果已成功导出到:\n{file_path}")
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", f"导出失败:\n{str(e)}")
        
    def get_sidebar_widget(self):
        """获取建模与预测模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            # 使用UI构建器创建侧边栏
            self._sidebar_widget = self.ui_builder.create_sidebar_widget()
            # 侧边栏创建后更新引用并补充信号连接
            self.ui_components.update(self.ui_builder.get_ui_components())
            self._setup_ui_component_references()
            self.setup_connections()
            self._wire_equivalent_duration_sync()
            self._apply_sim_mode_visibility()
            self._refresh_formula_reference()
        
        return self._sidebar_widget
