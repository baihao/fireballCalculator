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
from datetime import datetime
from PySide6.QtWidgets import QWidget, QMessageBox, QFileDialog
from .ui_widgets.model_tab_ui import ModelTabUI
from .controllers import ModelTabChartController

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
        
        # 初始化图表控制器
        self.chart_controller = ModelTabChartController(self.ui_builder)
        
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
    
    def init_empty_charts(self):
        """初始化空图表，显示默认占位内容"""
        try:
            self.chart_controller.reset()
        except Exception as e:
            print(f"初始化空图表失败: {e}")
    
    def setup_connections(self):
        """设置信号连接"""
        # 连接预测和导出按钮（仅在对应控件已创建时）
        try:
            if hasattr(self, 'predict_btn'):
                self.predict_btn.clicked.connect(self.start_prediction)
            if hasattr(self, 'export_btn'):
                self.export_btn.clicked.connect(self.export_results)
        except Exception:
            pass
        
    def start_prediction(self):
        """开始预测"""
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
            
            # 计算当量比值 M（当前当量/标准当量）
            # 标准当量设为 10.0 kg TNT（与默认值一致）
            STANDARD_EQUIVALENT = 10.0  # kg TNT
            m = equivalent / STANDARD_EQUIVALENT
            print(f"当量比值 M = {m:.3f} (当前当量={equivalent} kg TNT / 标准当量={STANDARD_EQUIVALENT} kg TNT)")
            
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
            
            # 1. 火球直径随时间变化（应用当量缩放）
            print("计算火球直径...")
            radius_calc = FireballCalculator()
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
            temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
            T_K = temp_calc.temperature_modified(t_ms)
            self.prediction_data['temperature_data'] = T_K
            
            # 更新温度图表
            self.chart_controller.update_temperature(t_ms, T_K)
            
            # 3. 热通量随时间变化 (不同距离)
            print("计算热通量...")
            distances = [4.0, 4.5, 5.0, 5.5, 6.0]
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
            x_values = np.linspace(4.0, 6.0, 50)
            H_values = []
            
            for x in x_values:
                q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, transmissivity_params)
                H = integrate_heat_radiation(q_t, t_ms)
                H_values.append(H)
            
            self.prediction_data['heat_radiation_data'] = {
                'distances': x_values,
                'heat_radiation': H_values
            }
            
            # 更新累积热辐射图表
            self.chart_controller.update_heat_radiation(x_values, H_values)
            
            print("✅ 所有预测曲线生成完成！")
            
        except Exception as e:
            print(f"❌ 生成预测曲线失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
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
                for dist in ['4.0', '4.5', '5.0', '5.5', '6.0']:
                    header.append(f'热通量_x={dist}m')
                writer.writerow(header)
                
                for i in range(len(t_ms)):
                    row = [f"{t_ms[i]:.3f}"]
                    for dist in ['4.0', '4.5', '5.0', '5.5', '6.0']:
                        if dist in self.prediction_data['heat_flux_data']:
                            row.append(f"{self.prediction_data['heat_flux_data'][dist][i]:.2f}")
                        else:
                            row.append("")
                    writer.writerow(row)
                
                writer.writerow([''])
                
                # 写入累积热辐射数据
                writer.writerow(['# 累积热辐射量随距离分布数据'])
                writer.writerow(['距离(m)', '热辐射量(J/m²)'])
                
                distances = self.prediction_data['heat_radiation_data']['distances']
                heat_radiation = self.prediction_data['heat_radiation_data']['heat_radiation']
                
                for i in range(len(distances)):
                    writer.writerow([
                        f"{distances[i]:.3f}",
                        f"{heat_radiation[i]:.2f}"
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
