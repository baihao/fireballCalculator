#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入模块标签页
"""

import numpy as np
import json
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QComboBox, QSlider, 
                               QGroupBox, QSplitter, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget
from fireball_temperature_calculator import FireballTemperatureCalculator


class InputTab(QWidget):
    """输入模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # 左侧图像预览区域
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # 图像预览、时间轴和温度图表组合
        preview_group = QGroupBox("爆炸流程预览")
        preview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #1f2937;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #111827;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #38bdf8;
            }
        """)
        preview_layout = QVBoxLayout()
        preview_layout.setAlignment(Qt.AlignTop)
        preview_layout.setSpacing(8)  # 设置组件间距
        
        # 图像预览
        self.image_preview = ImagePreviewWidget()
        preview_layout.addWidget(self.image_preview)
        
        # 时间轴
        timeline_layout = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_label = QLabel("t = 0 ms")
        self.time_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        timeline_layout.addWidget(self.time_slider)
        timeline_layout.addWidget(self.time_label)
        preview_layout.addLayout(timeline_layout)
        
        # 温度图表
        self.temp_chart = MatplotlibWidget(width=4, height=3)
        preview_layout.addWidget(self.temp_chart)
        
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group)
        
        left_widget.setLayout(left_layout)
        
        # 右侧参数区域
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)  # 减少组件间距
        right_layout.setAlignment(Qt.AlignTop)  # 顶部对齐
        
        # 材料与环境参数（只读显示）
        params_group = QGroupBox("材料与环境")
        params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #1f2937;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #111827;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #38bdf8;
            }
        """)
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)  # 减少标签间距
        params_layout.setContentsMargins(10, 15, 10, 10)
        params_layout.setAlignment(Qt.AlignTop)  # 标签顶部对齐
        
        # 创建只读标签显示参数
        self.material_label = QLabel("炸药类别: 40% Al / Rubber")
        self.equivalent_label = QLabel("当量: 10 kg TNT")
        self.al_percent_label = QLabel("含铝量: 30%")
        self.env_temp_label = QLabel("环境温度: 24°C")
        self.env_humidity_label = QLabel("相对湿度: 48%")
        self.env_pressure_label = QLabel("水饱和气压: 2987.87 Pa")
        
        # 设置标签样式，参考HTML设计
        label_style = """
            QLabel {
                color: #9ca3af;
                font-size: 12px;
                padding: 2px 0px;
                margin: 0px;
            }
        """
        self.material_label.setStyleSheet(label_style)
        self.equivalent_label.setStyleSheet(label_style)
        self.al_percent_label.setStyleSheet(label_style)
        self.env_temp_label.setStyleSheet(label_style)
        self.env_humidity_label.setStyleSheet(label_style)
        self.env_pressure_label.setStyleSheet(label_style)
        
        params_layout.addWidget(self.material_label)
        params_layout.addWidget(self.equivalent_label)
        params_layout.addWidget(self.al_percent_label)
        params_layout.addWidget(self.env_temp_label)
        params_layout.addWidget(self.env_humidity_label)
        params_layout.addWidget(self.env_pressure_label)
        
        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)
        
        # 保存采样序列按钮
        self.save_sequence_btn = QPushButton("保存采样序列")
        self.save_sequence_btn.setStyleSheet("""
            QPushButton {
                background-color: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 8px 12px;
                color: #e5e7eb;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1f2937;
            }
            QPushButton:pressed {
                background-color: #374151;
            }
        """)
        right_layout.addWidget(self.save_sequence_btn)
        
        right_widget.setLayout(right_layout)
        
        # 添加到主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
        # 初始化图表
        self.update_temperature_chart()
        
    def setup_connections(self):
        """设置信号连接"""
        self.time_slider.valueChanged.connect(self.on_time_changed)
        self.save_sequence_btn.clicked.connect(self.save_sequence)
        
    def on_time_changed(self, value):
        """时间轴变化"""
        self.time_label.setText(f"t = {value} ms")
        
    def update_temperature_chart(self):
        """更新温度图表"""
        try:
            t_ms = np.linspace(0, 140, 800)
            temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
            T_K = temp_calc.temperature_modified(t_ms)
            
            self.temp_chart.plot_line(
                t_ms, T_K,
                title="爆炸温度变化",
                xlabel="时间 (ms)",
                ylabel="温度 (K)",
                color='#38bdf8'
            )
        except Exception as e:
            print(f"更新温度图表失败: {e}")
    
    def update_parameters_display(self, material_type="40% Al / Rubber", equivalent="10", 
                                 al_percent="30", env_temp="24", env_humidity="48", 
                                 env_pressure="2987.87"):
        """更新右侧参数显示"""
        self.material_label.setText(f"炸药类别: {material_type}")
        self.equivalent_label.setText(f"当量: {equivalent} kg TNT")
        self.al_percent_label.setText(f"含铝量: {al_percent}%")
        self.env_temp_label.setText(f"环境温度: {env_temp}°C")
        self.env_humidity_label.setText(f"相对湿度: {env_humidity}%")
        self.env_pressure_label.setText(f"水饱和气压: {env_pressure} Pa")
    
    def save_sequence(self):
        """保存采样序列到JSON文件"""
        try:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存采样序列", "fireball_sequence.json", 
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 准备保存的数据
            sequence_data = {
                "metadata": {
                    "description": "爆炸火球分析采样序列",
                    "version": "1.0",
                    "created_at": str(np.datetime64('now'))
                },
                "parameters": {
                    "material_type": self.material_label.text().split(": ")[1],
                    "equivalent": self.equivalent_label.text().split(": ")[1].split(" ")[0],
                    "al_percent": self.al_percent_label.text().split(": ")[1].replace("%", ""),
                    "env_temp": self.env_temp_label.text().split(": ")[1].replace("°C", ""),
                    "env_humidity": self.env_humidity_label.text().split(": ")[1].replace("%", ""),
                    "env_pressure": self.env_pressure_label.text().split(": ")[1].replace(" Pa", "")
                },
                "files": {
                    "image_folder": "未设置",  # 这里可以从侧边栏获取
                    "temperature_file": "未设置"  # 这里可以从侧边栏获取
                },
                "temperature_data": {
                    "time_range": "0-140 ms",
                    "data_points": 800,
                    "note": "温度数据基于当前参数计算"
                }
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "保存成功", f"采样序列已保存到:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存采样序列时出错:\n{str(e)}")
