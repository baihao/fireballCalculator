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
from PySide6.QtWidgets import QWidget, QMessageBox, QFileDialog
from .ui_widgets.model_tab_ui import ModelTabUI
from .controllers import ModelTabChartController, ModelController
from .utils.calculator import build_prediction_bundle

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
        
        # 设置UI组件引用（向后兼容）
        self._setup_ui_component_references()
        
        # 初始化图表
        self.init_empty_charts()
        
        self.setup_connections()
    
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
        
        if "params_scroll_area" in self.ui_components:
            self.params_scroll_area = self.ui_components["params_scroll_area"]
    
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
        except Exception:
            pass
        
    def start_prediction(self):
        """开始预测"""
        if not self._model_import_ok:
            QMessageBox.warning(
                self,
                "仿真",
                "请先通过「选择模型」导入合法模型目录：\n"
                "需包含完整核岭回归 artefact（manifest.json 与三套 kbc_krr_*.joblib），\n"
                "或至少一个可解析的火球实验 JSON。",
            )
            return
        if not self.predict_btn.isEnabled():
            print("⚠️ 预测正在进行中，请勿重复点击...")
            return

        self._simulation_succeeded = False
        if hasattr(self, "export_btn"):
            self.export_btn.setEnabled(False)

        try:
            print("🔥 开始预测...")
            self.modeling_status.setText("正在计算预测结果...")
            self.predict_btn.setEnabled(False)

            equivalent = float(self.p_eq.text()) if self.p_eq.text() else 10.0
            al_content = float(self.p_al.text()) if self.p_al.text() else 30.0
            step = float(self.p_step.text()) if self.p_step.text() else 1.0
            duration = float(self.p_duration.text()) if self.p_duration.text() else 140.0

            env_temp = float(self.p_env_temp.text()) if self.p_env_temp.text() else 24.0
            env_humidity = float(self.p_env_humidity.text()) if self.p_env_humidity.text() else 48.0
            env_pressure = float(self.p_env_pressure.text()) if self.p_env_pressure.text() else 2987.87

            print(f"预测参数: 当量={equivalent}, 含铝量={al_content}%, 步长={step}, 时长={duration}ms")
            print(f"环境参数: 温度={env_temp}°C, 湿度={env_humidity}%, 气压={env_pressure}Pa")

            material_name = self.model_ctrl.get_material_by_al_content(al_content)
            print(f"选择材料: {material_name}")

            self.generate_prediction_curves(
                material_name, duration, equivalent, al_content, env_temp, env_humidity, env_pressure
            )

            self.modeling_status.setText("预测完成")
            self._simulation_succeeded = True
            print("✅ 预测完成！")

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            self.modeling_status.setText("预测失败")
            self._simulation_succeeded = False
            QMessageBox.critical(self, "错误", f"预测失败:\n{str(e)}")
        finally:
            if hasattr(self, "predict_btn"):
                self.predict_btn.setEnabled(self._model_import_ok)
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
    ):
        """使用 ``ModelController`` 解析 K/B/C，``utils.calculator`` 组装直径、温度、热通量与累积辐射。"""
        print(f"生成 {material_name} 材料的预测曲线...")
        time_points = int(duration / 1.0) + 1
        t_ms = np.linspace(0, duration, time_points)

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
        self.chart_controller.update_temperature(self.prediction_data["time_ms"], self.prediction_data["temperature_data"])
        self.chart_controller.update_heat_flux(self.prediction_data["time_ms"], heat_series)
        rad = self.prediction_data["heat_radiation_data"]
        self.chart_controller.update_heat_radiation(rad["distances"], rad["heat_radiation"])
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
        if hasattr(self, "export_btn"):
            self.export_btn.setEnabled(False)

        result = self.model_ctrl.load_folder(folder, parent=self)
        self._model_import_ok = result.can_run_simulation
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(result.can_run_simulation)
        if hasattr(self, "model_import_summary"):
            self.model_import_summary.setPlainText(result.summary_text)
        self.model_ctrl.apply_first_params_to_widgets(self.model_ctrl.last_first_params, self)

        if result.can_run_simulation:
            print(
                f"📁 模型目录已载入：{self.model_ctrl.model_folder_path}"
                f"（有效 JSON {result.applied_json_count} 个；"
                f"可在侧栏填写当量、含铝量后点击「开始仿真」）"
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
            equivalent = float(self.p_eq.text()) if self.p_eq.text() else 10.0
            al_content = float(self.p_al.text()) if self.p_al.text() else 30.0
            step = float(self.p_step.text()) if self.p_step.text() else 1.0
            duration = float(self.p_duration.text()) if self.p_duration.text() else 140.0
            
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
                writer.writerow(['时间(ms)', '时间(s)', '火球直径(m)', '火球温度(K)'])
                
                t_ms = self.prediction_data['time_ms']
                t_s = self.prediction_data['time_s']
                diameter = self.prediction_data['diameter_data']
                temperature = self.prediction_data['temperature_data']
                
                for i in range(len(t_ms)):
                    writer.writerow([
                        f"{t_ms[i]:.3f}",
                        f"{t_s[i]:.6f}",
                        f"{diameter[i]:.6f}",
                        f"{temperature[i]:.2f}"
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
        
        return self._sidebar_widget
