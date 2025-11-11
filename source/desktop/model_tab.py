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
from model_tab_ui import ModelTabUI

# 添加路径以导入计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
        
        # 设置UI组件引用（向后兼容）
        self._setup_ui_component_references()
        
        # 初始化图表
        self.init_empty_charts()
        
        self.setup_connections()
    
    def _setup_ui_component_references(self):
        """设置UI组件引用（向后兼容）"""
        # 状态显示控件
        self.modeling_status = self.ui_components['modeling_status']
        
        # 图表控件
        self.diam_chart = self.ui_components['diam_chart']
        self.temp_chart = self.ui_components['temp_chart']
        self.heat_flux_chart = self.ui_components['heat_flux_chart']
        self.heat_radiation_chart = self.ui_components['heat_radiation_chart']
        
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
        """初始化空图表，显示等待状态"""
        self.init_empty_chart(self.diam_chart, "火球直径随时间变化", "时间 (ms)", "直径 (m)")
        self.init_empty_chart(self.temp_chart, "火球温度随时间变化", "时间 (ms)", "温度 (K)")
        self.init_empty_chart(self.heat_flux_chart, "热通量随时间变化 (不同距离)", "时间 (ms)", "热通量 (W/m²)")
        self.init_empty_chart(self.heat_radiation_chart, "累积热辐射量随距离分布", "距离 (m)", "热辐射量 (J/m²)")
    
    def init_empty_chart(self, chart_widget, title, xlabel, ylabel):
        """初始化空图表，显示等待状态"""
        try:
            # 设置图表样式
            fig = chart_widget.figure
            fig.patch.set_facecolor('#0b1220')  # 深色背景
            
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0b1220')  # 深色背景
            
            # 设置标题和轴标签 - 保持固定样式
            ax.set_title(title, color='#cbd5e1', fontsize=12, pad=30)
            ax.set_xlabel(xlabel, color='#cbd5e1', fontsize=10)
            ax.set_ylabel(ylabel, color='#cbd5e1', fontsize=10)
            
            # 设置坐标轴颜色 - 保持固定样式
            ax.tick_params(colors='#9ca3af', labelsize=9)
            ax.spines['bottom'].set_color('#374151')
            ax.spines['top'].set_color('#374151')
            ax.spines['right'].set_color('#374151')
            ax.spines['left'].set_color('#374151')
            
            # 设置网格 - 保持固定样式
            ax.grid(True, alpha=0.3, color='#374151')
            
            # 显示等待消息
            ax.text(0.5, 0.5, '等待预测...', 
                   horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=14,
                   color='#9ca3af',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='#1f2937', 
                           edgecolor='#374151',
                           alpha=0.8))
            
            # 使用 constrained_layout（已在 framework.py 中启用）
            # 不再调用 tight_layout，避免与 constrained_layout 冲突
            
            # 刷新画布
            chart_widget.canvas.draw()
            
        except Exception as e:
            print(f"初始化空图表失败: {e}")
    
    def apply_chart_style(self, ax, title, xlabel, ylabel):
        """应用统一的图表样式"""
        # 设置背景色
        ax.set_facecolor('#0b1220')
        
        # 设置标题和轴标签 - 调整pad参数确保标题完全显示
        ax.set_title(title, color='#cbd5e1', fontsize=12, pad=30)
        ax.set_xlabel(xlabel, color='#cbd5e1', fontsize=10)
        ax.set_ylabel(ylabel, color='#cbd5e1', fontsize=10)
        
        # 设置坐标轴颜色
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['top'].set_color('#374151')
        ax.spines['right'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        
        # 设置网格
        ax.grid(True, alpha=0.3, color='#374151')
        
        # 使用 constrained_layout（已在 framework.py 中启用）
        # 不再调用 tight_layout，避免与 constrained_layout 冲突
    
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
            
            # 生成预测曲线
            self.generate_prediction_curves(material_name, duration, env_temp, env_humidity, env_pressure)
            
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
    
    def generate_prediction_curves(self, material_name, duration, env_temp=24.0, env_humidity=48.0, env_pressure=2987.87):
        """生成预测曲线"""
        try:
            print(f"生成 {material_name} 材料的预测曲线...")
            
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
            radius_calc = FireballCalculator()
            D_m = []
            for t in t_s:
                diameter = radius_calc.calculate_diameter(t, material_name)
                D_m.append(diameter)
            D_m = np.array(D_m)
            self.prediction_data['diameter_data'] = D_m
            
            # 清除之前的等待状态并绘制新数据
            self.diam_chart.clear()
            # 重新应用样式
            fig = self.diam_chart.figure
            fig.patch.set_facecolor('#0b1220')
            ax = fig.add_subplot(111)
            self.apply_chart_style(ax, "火球直径随时间变化", "时间 (ms)", "直径 (m)")
            # 绘制数据线
            ax.plot(t_ms, D_m, color='#22d3ee', linewidth=2)
            self.diam_chart.canvas.draw()
            
            # 2. 火球温度随时间变化
            print("计算火球温度...")
            temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
            T_K = temp_calc.temperature_modified(t_ms)
            self.prediction_data['temperature_data'] = T_K
            
            # 清除之前的等待状态并绘制新数据
            self.temp_chart.clear()
            # 重新应用样式
            fig = self.temp_chart.figure
            fig.patch.set_facecolor('#0b1220')
            ax = fig.add_subplot(111)
            self.apply_chart_style(ax, "火球温度随时间变化", "时间 (ms)", "温度 (K)")
            # 绘制数据线
            ax.plot(t_ms, T_K, color='#38bdf8', linewidth=2)
            self.temp_chart.canvas.draw()
            
            # 3. 热通量随时间变化 (不同距离)
            print("计算热通量...")
            distances = [4.0, 4.5, 5.0, 5.5, 6.0]
            heat_flux_data = {}
            
            # 创建传输率参数对象
            transmissivity_params = TransmissivityParams(
                Ta_K=env_temp + 273.15,  # 转换为开尔文
                RH_percent=env_humidity,
                PwSat_Pa=env_pressure
            )
            
            for dist in distances:
                q_t = compute_heat_flux_over_time(dist, t_ms, T_K, D_m, transmissivity_params)
                heat_flux_data[f'x = {dist:.1f} m'] = (t_ms, q_t)
                self.prediction_data['heat_flux_data'][f'{dist:.1f}'] = q_t
            
            # 清除之前的等待状态并绘制新数据
            self.heat_flux_chart.clear()
            # 重新应用样式
            fig = self.heat_flux_chart.figure
            fig.patch.set_facecolor('#0b1220')
            ax = fig.add_subplot(111)
            self.apply_chart_style(ax, "热通量随时间变化 (不同距离)", "时间 (ms)", "热通量 (W/m²)")
            # 绘制多条数据线
            colors = ['#22d3ee', '#38bdf8', '#10b981', '#f59e0b', '#ef4444']
            for i, (label, (x_data, y_data)) in enumerate(heat_flux_data.items()):
                ax.plot(x_data, y_data, color=colors[i % len(colors)], linewidth=2, label=label)
            ax.legend(loc='upper right', fontsize=8, facecolor='#1f2937', edgecolor='#374151', labelcolor='#cbd5e1')
            self.heat_flux_chart.canvas.draw()
            
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
            
            # 清除之前的等待状态并绘制新数据
            self.heat_radiation_chart.clear()
            # 重新应用样式
            fig = self.heat_radiation_chart.figure
            fig.patch.set_facecolor('#0b1220')
            ax = fig.add_subplot(111)
            self.apply_chart_style(ax, "累积热辐射量随距离分布", "距离 (m)", "热辐射量 (J/m²)")
            # 绘制数据线
            ax.plot(x_values, H_values, color='#10b981', linewidth=2)
            self.heat_radiation_chart.canvas.draw()
            
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
        
    def update_prediction_charts(self):
        """更新预测图表"""
        try:
            # 火球直径随时间变化
            t_ms = np.linspace(0, 140, 800)
            radius_calc = FireballCalculator()
            t_s = t_ms / 1000.0
            D_m = radius_calc.calculate_diameter(t_s, '40%Al/Rubber')
            
            self.diam_chart.plot_line(
                t_ms, D_m,
                title="火球直径随时间变化",
                xlabel="时间 (ms)",
                ylabel="直径 (m)",
                color='#22d3ee'
            )
            
            # 火球温度随时间变化
            temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
            T_K = temp_calc.temperature_modified(t_ms)
            
            self.temp_chart.plot_line(
                t_ms, T_K,
                title="火球温度随时间变化",
                xlabel="时间 (ms)",
                ylabel="温度 (K)",
                color='#38bdf8'
            )
            
            # 热通量随时间变化 (不同距离)
            distances = [4.0, 4.5, 5.0, 5.5, 6.0]
            heat_flux_data = {}
            
            for dist in distances:
                q_t = compute_heat_flux_over_time(dist, t_ms, T_K, D_m, TransmissivityParams())
                heat_flux_data[f'x = {dist:.1f} m'] = (t_ms, q_t)
            
            self.heat_flux_chart.plot_multiple_lines(
                heat_flux_data,
                title="热通量随时间变化 (不同距离)",
                xlabel="时间 (ms)",
                ylabel="热通量 (W/m²)"
            )
            
            # 累积热辐射量随距离分布
            x_values = np.linspace(4.0, 6.0, 50)
            H_values = []
            
            for x in x_values:
                q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, TransmissivityParams())
                H = integrate_heat_radiation(q_t, t_ms)
                H_values.append(H)
            
            self.heat_radiation_chart.plot_line(
                x_values, H_values,
                title="累积热辐射量随距离分布",
                xlabel="距离 (m)",
                ylabel="热辐射量 (J/m²)",
                color='#10b981'
            )
            
            self.modeling_status.setText("已生成示例预测曲线")
            
        except Exception as e:
            print(f"更新预测图表失败: {e}")
    
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
