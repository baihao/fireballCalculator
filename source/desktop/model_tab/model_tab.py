#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建模与预测模块标签页
"""

import numpy as np
import sys
import os
import csv
import json
import re
from datetime import datetime
from PySide6.QtWidgets import QWidget, QMessageBox, QFileDialog
from .ui_widgets.model_tab_ui import ModelTabUI
from .controllers import ModelTabChartController, TrainConfigController

# 添加路径以导入计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fireball_radius_calculator import FireballCalculator
from fireball_temperature_calculator import FireballTemperatureCalculator
from transmissivity_calculator import TransmissivityParams
from fireball_heat_radiation_calculator import (compute_heat_flux_over_time,
                                               integrate_heat_radiation)


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
        self.train_config_controller = TrainConfigController(self)
        self.fireball_calculator = FireballCalculator()
        self.training_files = []
        # 存储训练文件中的温度数据
        self.training_temperature_data = None  # (time_ms_array, temperature_K_array)
        # 存储训练文件中的K值（用于直接使用，不进行当量缩放）
        self.training_K_value = None  # 从训练文件中提取的K值
        # 存储训练文件中的当量值（作为标准当量）
        self.training_equivalent = None  # 从训练文件中提取的当量值
        
        # 设置UI组件引用（向后兼容）
        self._setup_ui_component_references()
        
        # 初始化图表
        self.init_empty_charts()
        
        self.setup_connections()
    
    def _setup_ui_component_references(self):
        """设置UI组件引用（向后兼容）"""
        # 状态显示控件
        self.modeling_status = self.ui_components['modeling_status']
        
        # 按钮控件（可能尚未创建侧边栏，此时这些键不存在，需做兼容判断）
        if 'predict_btn' in self.ui_components:
            self.predict_btn = self.ui_components['predict_btn']
        if 'export_btn' in self.ui_components:
            self.export_btn = self.ui_components['export_btn']
        if 'train_series_btn' in self.ui_components:
            self.train_series_btn = self.ui_components['train_series_btn']
        if 'train_btn' in self.ui_components:
            self.train_btn = self.ui_components['train_btn']
        if 'train_file_list' in self.ui_components:
            self.train_file_list = self.ui_components['train_file_list']
            self.train_file_list_max_height = self.train_file_list.maximumHeight() or 120
        if 'train_params_btn' in self.ui_components:
            self.train_params_btn = self.ui_components['train_params_btn']
        
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
        
        # 其他控件
        if 'algo' in self.ui_components:
            self.algo = self.ui_components['algo']
        if 'model_list' in self.ui_components:
            self.model_list = self.ui_components['model_list']
        if 'lr' in self.ui_components:
            self.lr = self.ui_components['lr']
        if 'epochs' in self.ui_components:
            self.epochs = self.ui_components['epochs']
        if 'params_scroll_area' in self.ui_components:
            self.params_scroll_area = self.ui_components['params_scroll_area']
    
    def init_empty_charts(self):
        """初始化空图表，显示默认占位内容"""
        try:
            self.chart_controller.reset()
        except Exception as e:
            print(f"初始化空图表失败: {e}")
    
    def setup_connections(self):
        """设置信号连接"""
        # 连接预测、导出与训练按钮（仅在对应控件已创建时）
        try:
            if hasattr(self, 'predict_btn'):
                self.predict_btn.clicked.connect(self.start_prediction)
            if hasattr(self, 'export_btn'):
                self.export_btn.clicked.connect(self.export_results)
            if hasattr(self, 'train_series_btn'):
                self.train_series_btn.clicked.connect(self.select_training_files)
            if hasattr(self, 'train_params_btn'):
                self.train_params_btn.clicked.connect(self.open_train_config_dialog)
            if hasattr(self, 'train_btn'):
                self.train_btn.clicked.connect(self.start_training)
        except Exception:
            pass
        
    def start_prediction(self):
        """开始预测"""
        # 防止重复点击：如果按钮已被禁用，说明正在执行中，直接返回
        if not self.predict_btn.isEnabled():
            print("⚠️ 预测正在进行中，请勿重复点击...")
            return
        
        try:
            print("🔥 开始预测...")
            self.modeling_status.setText("正在计算预测结果...")
            self.predict_btn.setEnabled(False)
            
            # 获取预测参数
            equivalent = float(self.p_eq.text()) if self.p_eq.text() else 10.0
            al_content = float(self.p_al.text()) if self.p_al.text() else 30.0
            step = float(self.p_step.text()) if self.p_step.text() else 1.0
            duration = float(self.p_duration.text()) if self.p_duration.text() else 140.0
            
            # 获取环境参数
            env_temp = float(self.p_env_temp.text()) if self.p_env_temp.text() else 24.0
            env_humidity = float(self.p_env_humidity.text()) if self.p_env_humidity.text() else 48.0
            env_pressure = float(self.p_env_pressure.text()) if self.p_env_pressure.text() else 2987.87
            
            print(f"预测参数: 当量={equivalent}, 含铝量={al_content}%, 步长={step}, 时长={duration}ms")
            print(f"环境参数: 温度={env_temp}°C, 湿度={env_humidity}%, 气压={env_pressure}Pa")
            
            # 根据含铝量选择材料类型
            material_name = self.get_material_by_al_content(al_content)
            print(f"选择材料: {material_name}")
            
            # 生成预测曲线（传入当量值）
            self.generate_prediction_curves(material_name, duration, equivalent, env_temp, env_humidity, env_pressure)
            
            # 更新状态
            self.modeling_status.setText("预测完成")
            self.predict_btn.setEnabled(True)
            self.export_btn.setEnabled(True)  # 启用导出按钮
            
            print("✅ 预测完成！")
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            self.modeling_status.setText("预测失败")
            self.predict_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"预测失败:\n{str(e)}")
    
    def get_material_by_al_content(self, al_content):
        """根据含铝量选择材料类型"""
        if al_content <= 30:
            return '30%Al/Rubber'
        elif al_content <= 40:
            return '40%Al/Rubber'
        elif al_content <= 50:
            return '50%Al/Rubber'
        elif al_content <= 60:
            return '60%Al/Rubber'
        else:
            return '40%Al/Rubber'  # 默认
    
    def generate_prediction_curves(self, material_name, duration, equivalent=10.0, env_temp=24.0, env_humidity=48.0, env_pressure=2987.87):
        """
        生成预测曲线
        
        参数:
        material_name: 材料名称
        duration: 仿真时长 (ms)
        equivalent: 爆炸当量 (kg TNT)，默认10.0 kg TNT（标准当量）
        env_temp: 环境温度 (°C)
        env_humidity: 相对湿度 (%)
        env_pressure: 水饱和气压 (Pa)
        """
        try:
            print(f"生成 {material_name} 材料的预测曲线...")
            
            radius_calc = self.fireball_calculator
            # 确定标准当量：如果有训练文件当量值，使用训练文件的当量值；否则使用计算器中的标准当量
            if self.training_equivalent is not None:
                standard_equivalent = self.training_equivalent
                print(f"✓ 使用训练文件当量值作为标准当量: {standard_equivalent} kg TNT")
            else:
                standard_equivalent = radius_calc.get_standard_equivalent(material_name)
                print(f"✓ 使用计算器标准当量: {standard_equivalent} kg TNT")
            
            # 计算当量比值 M（当前当量/标准当量）
            # 只有当用户输入的当量与标准当量不同时，才进行缩放（m != 1.0）
            m = equivalent / standard_equivalent if standard_equivalent > 0 else 1.0
            
            if abs(m - 1.0) < 1e-6:
                print(f"✓ 当量比值 M = {m:.3f} (当前当量={equivalent} kg TNT = 标准当量={standard_equivalent} kg TNT，不进行缩放)")
            else:
                print(
                    f"✓ 当量比值 M = {m:.3f} (当前当量={equivalent} kg TNT / 标准当量={standard_equivalent} kg TNT，将进行缩放)"
                )
            
            # 如果加载了训练文件且训练文件中有K值，提示用户
            if self.training_K_value is not None:
                print(f"✓ 检测到训练文件K值 ({self.training_K_value:.3f} m)")
            
            # 生成时间序列
            time_points = int(duration / 1.0) + 1  # 1ms步长
            t_ms = np.linspace(0, duration, time_points)
            t_s = t_ms / 1000.0
            
            # 保存预测数据供导出使用
            self.prediction_data = {
                'time_ms': t_ms,
                'time_s': t_s,
                'material_name': material_name,
                'duration': duration,
                'equivalent': equivalent,
                'equivalent_ratio': m,
                'env_temp': env_temp,
                'env_humidity': env_humidity,
                'env_pressure': env_pressure,
                'diameter_data': None,
                'temperature_data': None,
                'heat_flux_data': {},
                'heat_radiation_data': {}
            }
            
            # 1. 火球直径随时间变化
            print("计算火球直径...")
            if abs(m - 1.0) < 1e-6:
                print(f"✓ 当量比值 M = 1.0，使用标准K值（不进行缩放）")
            else:
                print(f"✓ 当量比值 M = {m:.3f}，将根据当量比值缩放K值")
            
            if self.training_K_value is not None:
                # 验证计算器中的K值
                current_K = radius_calc.get_standard_parameters(material_name)['K']
                print(f"  训练文件K值: {self.training_K_value:.3f} m (直径)")
                print(f"  计算器中的K值: {current_K:.3f} m (半径)")
                if abs(current_K - self.training_K_value / 2.0) > 0.01:
                    print(f"  ⚠️ 警告：计算器中的K值 ({current_K:.3f} m) 与训练文件K值 ({self.training_K_value:.3f} m / 2) 不一致")
            
            D_m = []
            for t in t_s:
                diameter = radius_calc.calculate_diameter(t, material_name, m)
                D_m.append(diameter)
            D_m = np.array(D_m)
            self.prediction_data['diameter_data'] = D_m
            
            # 更新直径图表
            self.chart_controller.update_diameter(t_ms, D_m)
            
            # 2. 火球温度随时间变化
            print("计算火球温度...")
            # 检查是否有训练文件中的温度数据
            if self.training_temperature_data is not None:
                train_time_ms, train_temp_K = self.training_temperature_data
                # 使用训练温度数据，插值到预测时间网格（np.interp会自动处理外推）
                T_K = np.interp(t_ms, train_time_ms, train_temp_K)
                train_time_min = train_time_ms.min()
                train_time_max = train_time_ms.max()
                pred_time_min = t_ms.min()
                pred_time_max = t_ms.max()
                
                # 检查时间范围覆盖情况
                if train_time_min <= pred_time_min and train_time_max >= pred_time_max:
                    print(f"✓ 使用训练文件中的温度数据（{len(train_time_ms)} 个数据点，完全覆盖预测时间范围）")
                else:
                    print(f"✓ 使用训练文件中的温度数据（{len(train_time_ms)} 个数据点）")
                    if train_time_min > pred_time_min:
                        print(f"  ⚠️ 注意：预测开始时间 ({pred_time_min:.1f} ms) 早于训练数据 ({train_time_min:.1f} ms)，使用边界值外推")
                    if train_time_max < pred_time_max:
                        print(f"  ⚠️ 注意：预测结束时间 ({pred_time_max:.1f} ms) 晚于训练数据 ({train_time_max:.1f} ms)，使用边界值外推")
            else:
                # 没有训练温度数据，使用默认温度模型
                temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
                T_K = temp_calc.temperature_modified(t_ms)
                print("使用默认温度模型")
            
            self.prediction_data['temperature_data'] = T_K
            
            # 更新温度图表
            self.chart_controller.update_temperature(t_ms, T_K)
            
            # 3. 热通量随时间变化 (不同距离)
            print("计算热通量...")
            distances = [6.0, 7.0, 8.0, 9.0, 10.0]
            heat_flux_series = []
            
            # 创建传输率参数对象
            transmissivity_params = TransmissivityParams(
                Ta_K=env_temp + 273.15,  # 转换为开尔文
                RH_percent=env_humidity,
                PwSat_Pa=env_pressure
            )
            
            for dist in distances:
                q_t = compute_heat_flux_over_time(dist, t_ms, T_K, D_m, transmissivity_params)
                heat_flux_series.append([dist, q_t])
                self.prediction_data['heat_flux_data'][f'{dist:.1f}'] = q_t
            
            # 更新热通量图表
            self.chart_controller.update_heat_flux(t_ms, heat_flux_series)
            
            # 4. 累积热辐射量随距离分布
            print("计算累积热辐射...")
            x_values = np.linspace(6.0, 10.0, 50)
            H_values_kJ_per_m2 = []
            
            # 转换为千焦每平方米：1 kJ = 1000 J
            J_TO_KJ = 1000.0
            
            for x in x_values:
                q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, transmissivity_params)
                H_J_per_m2 = integrate_heat_radiation(q_t, t_ms)
                # 转换为 kJ/m²
                H_kJ_per_m2 = H_J_per_m2 / J_TO_KJ
                H_values_kJ_per_m2.append(H_kJ_per_m2)
            
            self.prediction_data['heat_radiation_data'] = {
                'distances': x_values,
                'heat_radiation': H_values_kJ_per_m2  # 单位：kJ/m²
            }
            
            # 更新累积热辐射图表
            self.chart_controller.update_heat_radiation(x_values, H_values_kJ_per_m2)
            
            print("✅ 所有预测曲线生成完成！")
            
        except Exception as e:
            print(f"❌ 生成预测曲线失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def select_training_files(self):
        """选择训练文件并显示"""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择训练文件",
                "",
                "火球序列/CSV (*.json *.csv);;所有文件 (*.*)"
            )
            if not files:
                self.training_files = []
                self.training_temperature_data = None  # 清空温度数据
                self.training_K_value = None  # 清空K值
                self.training_equivalent = None  # 清空当量值
                self._update_train_file_list_state()
                return
            self.training_files = files
            if hasattr(self, 'train_file_list'):
                self.train_file_list.clear()
                for path in files:
                    self.train_file_list.addItem(os.path.basename(path))
                self._update_train_file_list_state()
            self._apply_training_parameters_from_files(files)
            print(f"📁 已选择训练文件 {len(files)} 个")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"选择训练文件失败:\n{str(e)}")
    
    def open_train_config_dialog(self):
        """打开训练参数配置"""
        config = self.train_config_controller.open_dialog()
        if config:
            print(f"⚙️ 训练参数已更新: {config}")
    
    def start_training(self):
        """开始训练（占位逻辑）"""
        if not self.training_files:
            QMessageBox.warning(self, "提示", "请先选择训练文件。")
            return
        config = self.train_config_controller.get_config()
        QMessageBox.information(
            self,
            "开始训练",
            f"训练文件数: {len(self.training_files)}\n算法: {config['algorithm']}\n学习率: {config['learning_rate']}\n轮次: {config['epochs']}"
        )
    
    def export_results(self):
        """导出预测结果到CSV文件"""
        try:
            if not hasattr(self, 'prediction_data') or self.prediction_data is None:
                QMessageBox.warning(self, "警告", "请先进行预测！")
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

    def _update_train_file_list_state(self):
        """根据文件数量更新训练文件列表显示状态"""
        if not hasattr(self, 'train_file_list'):
            return
        if not self.training_files:
            self.train_file_list.clear()
            self.train_file_list.setVisible(False)
            self.train_file_list.setFixedHeight(0)
            return
        self.train_file_list.setVisible(True)
        row_height = self.train_file_list.sizeHintForRow(0) if self.train_file_list.count() else 28
        target_height = min(
            getattr(self, 'train_file_list_max_height', 120),
            row_height * self.train_file_list.count() + 6
        )
        self.train_file_list.setFixedHeight(int(target_height))

    def _apply_training_parameters_from_files(self, files):
        """从训练文件中读取参数并更新火球计算器"""
        if not files:
            return
        applied = 0
        first_params = None
        for path in files:
            params = self._apply_training_parameters_from_file(path)
            if params:
                applied += 1
                if first_params is None:
                    first_params = params
        if applied:
            print(f"🔧 已从 {applied} 个训练文件更新标准当量与 K/B/C 参数")
        if first_params:
            self._apply_simulation_inputs_from_params(first_params)

    def _apply_training_parameters_from_file(self, file_path):
        """单个文件解析逻辑"""
        if not file_path.lower().endswith('.json'):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"⚠️ 无法读取训练文件 {file_path}: {e}")
            return None
        
        params = data.get('parameters') or {}
        drag_fit = data.get('drag_fit') or {}
        
        equivalent = self._safe_float(params.get('equivalent'))
        al_percent = self._safe_float(params.get('al_percent'))
        duration = self._safe_float(params.get('explosion_duration'))
        k_value = self._safe_float(drag_fit.get('K'))
        b_value = self._safe_float(drag_fit.get('B'))
        c_value = self._safe_float(drag_fit.get('C'))
        
        # 调试信息：打印提取的值
        print(f"从训练文件提取参数:")
        print(f"  equivalent: {equivalent}")
        print(f"  al_percent: {al_percent}")
        print(f"  K值: {k_value}")
        print(f"  B值: {b_value}")
        print(f"  C值: {c_value}")
        
        if equivalent is None or al_percent is None:
            return None
        
        # 提取温度数据（如果存在）
        temperature_data = data.get('temperature', [])
        if temperature_data and len(temperature_data) > 0:
            try:
                time_data = []
                temp_data = []
                for time_temp_pair in temperature_data:
                    if len(time_temp_pair) >= 2:
                        time_data.append(float(time_temp_pair[0]))
                        temp_data.append(float(time_temp_pair[1]))
                if len(time_data) > 0 and len(temp_data) > 0:
                    self.training_temperature_data = (np.array(time_data), np.array(temp_data))
                    print(f"✓ 从训练文件加载温度数据: {len(time_data)} 个数据点")
            except Exception as e:
                print(f"⚠️ 提取训练文件温度数据失败: {e}")
                self.training_temperature_data = None
        else:
            # 如果没有温度数据，清空之前的温度数据
            self.training_temperature_data = None
        
        material_name = self.get_material_by_al_content(al_percent)
        kwargs = {'standard_equivalent': equivalent}
        # 保存训练文件中的当量值（作为标准当量）
        self.training_equivalent = equivalent
        print(f"✓ 保存训练文件当量值: {equivalent} kg TNT（将作为标准当量）")
        if k_value is not None:
            # 重要：训练文件中的K值是直径K值（来自拖曳函数 D(t) = K * (1 - B*exp(-C*t^2))）
            # 而FireballCalculator中的K值是半径K值（来自 R(t) = K * (1 - B*exp(-C*t^2))）
            # 需要将直径K值转换为半径K值：K_radius = K_diameter / 2
            k_radius = k_value / 2.0
            kwargs['K'] = k_radius
            # 保存训练文件中的K值（直径K值），用于预测时直接使用（不进行当量缩放）
            self.training_K_value = k_value  # 保存原始直径K值，用于显示
            print(f"✓ 保存训练文件K值: {k_value:.3f} m (直径K值)")
            print(f"  转换为半径K值: {k_radius:.3f} m，对应最大直径: {k_value:.3f} m")
            # 验证K值是否正确更新到计算器中
            try:
                updated_K = self.fireball_calculator.get_standard_parameters(material_name)['K']
                print(f"  验证：计算器中的K值 = {updated_K:.3f} m (半径)")
            except Exception as e:
                print(f"  ⚠️ 验证K值失败: {e}")
        else:
            # 如果没有K值，清空之前保存的K值
            self.training_K_value = None
            print(f"⚠️ 训练文件中没有K值，将使用当量缩放")
        if b_value is not None:
            kwargs['B'] = b_value
        if c_value is not None:
            kwargs['C'] = c_value
        
        try:
            self.fireball_calculator.set_standard_parameters(material_name, **kwargs)
            return {
                'equivalent': equivalent,
                'al_percent': al_percent,
                'duration': duration,
            }
        except Exception as exc:
            print(f"⚠️ 更新材料 {material_name} 参数失败: {exc}")
            return None

    def _apply_simulation_inputs_from_params(self, params):
        """根据训练文件中的参数更新仿真默认值"""
        if hasattr(self, 'p_eq') and params.get('equivalent') is not None:
            self.p_eq.setText(f"{params['equivalent']:.6g}")
        if hasattr(self, 'p_al') and params.get('al_percent') is not None:
            self.p_al.setText(f"{params['al_percent']:.6g}")
        if hasattr(self, 'p_duration') and params.get('duration') is not None:
            self.p_duration.setText(f"{params['duration']:.6g}")

    @staticmethod
    def _safe_float(value):
        """将各种字符串/数字转换为 float"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            cleaned = cleaned.replace('%', '')
            cleaned = re.sub(r'[^\d\.\-eE+]', '', cleaned)
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None
