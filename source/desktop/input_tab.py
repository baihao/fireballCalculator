#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入模块标签页
"""

import numpy as np
import json
import os
import glob
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QComboBox, QSlider, 
                               QGroupBox, QSplitter, QFileDialog, QMessageBox,
                               QInputDialog)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget
from fireball_temperature_calculator import FireballTemperatureCalculator


class InputTab(QWidget):
    """输入模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化文件路径属性
        self.image_folder_path = None
        self.video_file_path = None
        self.image_files = []  # 存储图像文件列表
        self.current_image_index = 0  # 当前显示的图像索引
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
        # 更新标签显示
        if self.image_files:
            # 计算实际时间（毫秒）
            try:
                duration = int(self.explosion_duration.text())
                total_frames = len(self.image_files)
                if total_frames > 1:
                    time_ms = (value / (total_frames - 1)) * duration
                else:
                    time_ms = 0
                self.time_label.setText(f"t = {time_ms:.1f} ms (帧 {value + 1}/{total_frames})")
            except ValueError:
                self.time_label.setText(f"帧 {value + 1}/{len(self.image_files)}")
            
            # 显示对应的图像
            self.display_image_at_index(value)
        else:
            self.time_label.setText(f"t = {value} ms")
        
    def display_image_at_index(self, index):
        """显示指定索引的图像"""
        if not self.image_files or index < 0 or index >= len(self.image_files):
            return
        
        try:
            image_path = self.image_files[index]
            # 使用 ImagePreviewWidget 的 set_image 方法
            self.image_preview.set_image(image_path)
            self.current_image_index = index
            print(f"显示图像: {image_path} (索引: {index})")
                
        except Exception as e:
            print(f"显示图像失败: {e}")
    
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
                                 env_pressure="2987.87", explosion_duration="140", pixel_length="0.01"):
        """更新右侧参数显示"""
        self.material_label.setText(f"炸药类别: {material_type}")
        self.equivalent_label.setText(f"当量: {equivalent} kg TNT")
        self.al_percent_label.setText(f"含铝量: {al_percent}%")
        self.env_temp_label.setText(f"环境温度: {env_temp}°C")
        self.env_humidity_label.setText(f"相对湿度: {env_humidity}%")
        self.env_pressure_label.setText(f"水饱和气压: {env_pressure} Pa")
        # 注意：这里可以添加新参数的显示，但右侧面板目前没有对应的标签
    
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
                    "env_pressure": self.env_pressure_label.text().split(": ")[1].replace(" Pa", ""),
                    "explosion_duration": self.explosion_duration.text(),
                    "pixel_length": self.pixel_length.text()
                },
                "files": {
                    "image_folder": self.image_folder_path if self.image_folder_path else "未设置",
                    "video_file": self.video_file_path if self.video_file_path else "未设置",
                    "temperature_file": "未设置"  # 这里可以从侧边栏获取
                },
                "temperature_data": {
                    "time_range": "0-140 ms",
                    "data_points": 800,
                    "note": "温度数据基于当前参数计算"
                },
                "image_sequence_info": {
                    "folder_path": self.image_folder_path if self.image_folder_path else "未设置",
                    "note": "图像序列文件夹路径"
                }
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "保存成功", f"采样序列已保存到:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存采样序列时出错:\n{str(e)}")
    
    
    def get_sidebar_widget(self):
        """获取输入模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            self._sidebar_widget = QGroupBox("图像与参数输入")
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)  # 只向上对齐
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(5)
            
            # 文件输入组控件
            self.image_sequence_btn = QPushButton("导入火球图像序列")
            self.video_file_btn = QPushButton("导入火球视频文件")
            self.temp_files_btn = QPushButton("导入火球温度时间序列")
            self.explosion_duration = QLineEdit("140")
            self.pixel_length = QLineEdit("0.01")
            
            # 参数组控件
            self.explosive_type = QComboBox()
            self.explosive_type.addItems(["TNT", "Polyurethane", "30% Al / Rubber", 
                                        "40% Al / Rubber", "50% Al / Rubber", "60% Al / Rubber"])
            self.explosive_type.setCurrentText("40% Al / Rubber")
            
            # 当量和含铝量
            self.equivalent = QLineEdit("10")
            self.al_percent = QLineEdit("30")
            
            # 环境参数
            self.env_temp = QLineEdit("24")
            self.env_humidity = QLineEdit("48")
            self.env_pressure = QLineEdit("2987.87")
            
            # 控制按钮
            self.clear_btn = QPushButton("清空输入")
            self.input_status = QLabel("等待文件导入…")
            self.input_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
            
            # 文件输入组
            file_group = QGroupBox("文件输入")
            file_group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #374151;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 10px;
                    background-color: #1f2937;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #60a5fa;
                }
            """)
            file_layout = QVBoxLayout()
            file_layout.setAlignment(Qt.AlignTop)
            file_layout.setContentsMargins(10, 10, 10, 10)
            
            # 创建标签样式
            label_style = """
                QLabel {
                    background-color: transparent;
                    color: #e5e7eb;
                    font-size: 12px;
                    padding: 2px 0px;
                }
            """
            
            # 文件输入标签
            image_seq_label = QLabel("导入火球图像序列")
            image_seq_label.setStyleSheet(label_style)
            file_layout.addWidget(image_seq_label)
            file_layout.addWidget(self.image_sequence_btn)
            
            video_label = QLabel("导入火球视频文件")
            video_label.setStyleSheet(label_style)
            file_layout.addWidget(video_label)
            file_layout.addWidget(self.video_file_btn)
            
            temp_label = QLabel("导入火球温度时间序列（CSV/JSON）")
            temp_label.setStyleSheet(label_style)
            file_layout.addWidget(temp_label)
            file_layout.addWidget(self.temp_files_btn)
            
            # 时长和像素长度
            duration_layout = QHBoxLayout()
            duration_layout.setAlignment(Qt.AlignTop)
            duration_label = QLabel("爆炸时长(ms):")
            duration_label.setStyleSheet(label_style)
            duration_layout.addWidget(duration_label)
            duration_layout.addWidget(self.explosion_duration)
            file_layout.addLayout(duration_layout)
            
            pixel_layout = QHBoxLayout()
            pixel_layout.setAlignment(Qt.AlignTop)
            pixel_label = QLabel("像素长度(m):")
            pixel_label.setStyleSheet(label_style)
            pixel_layout.addWidget(pixel_label)
            pixel_layout.addWidget(self.pixel_length)
            file_layout.addLayout(pixel_layout)
            
            file_group.setLayout(file_layout)
            layout.addWidget(file_group)
            
            # 参数组
            param_group = QGroupBox("参数设置")
            param_group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #374151;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 10px;
                    background-color: #1f2937;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #60a5fa;
                }
            """)
            param_layout = QVBoxLayout()
            param_layout.setAlignment(Qt.AlignTop)
            param_layout.setContentsMargins(10, 10, 10, 10)
            
            # 参数组标签样式（与文件组相同）
            param_label_style = """
                QLabel {
                    background-color: transparent;
                    color: #e5e7eb;
                    font-size: 12px;
                    padding: 2px 0px;
                }
            """
            
            # 炸药参数
            explosive_label = QLabel("炸药类别")
            explosive_label.setStyleSheet(param_label_style)
            param_layout.addWidget(explosive_label)
            param_layout.addWidget(self.explosive_type)
            
            # 当量和含铝量
            equivalent_label = QLabel("当量（kg TNT）")
            equivalent_label.setStyleSheet(param_label_style)
            param_layout.addWidget(equivalent_label)
            param_layout.addWidget(self.equivalent)
            
            al_percent_label = QLabel("含铝量（%）")
            al_percent_label.setStyleSheet(param_label_style)
            param_layout.addWidget(al_percent_label)
            param_layout.addWidget(self.al_percent)
            
            # 环境参数
            env_temp_label = QLabel("环境温度 Ta（°C）")
            env_temp_label.setStyleSheet(param_label_style)
            param_layout.addWidget(env_temp_label)
            param_layout.addWidget(self.env_temp)
            
            env_humidity_label = QLabel("相对湿度 RH（%）")
            env_humidity_label.setStyleSheet(param_label_style)
            param_layout.addWidget(env_humidity_label)
            param_layout.addWidget(self.env_humidity)
            
            env_pressure_label = QLabel("水饱和气压 PwSat(Ta)（Pa）")
            env_pressure_label.setStyleSheet(param_label_style)
            param_layout.addWidget(env_pressure_label)
            param_layout.addWidget(self.env_pressure)
            
            param_group.setLayout(param_layout)
            layout.addWidget(param_group)
            
            # 控制按钮
            button_layout = QHBoxLayout()
            button_layout.setAlignment(Qt.AlignTop)
            button_layout.addWidget(self.clear_btn)
            button_layout.addStretch()
            button_layout.addWidget(self.input_status)
            layout.addLayout(button_layout)
            
            self._sidebar_widget.setLayout(layout)
            
            # 设置信号连接（只设置一次）
            self.setup_sidebar_connections()
        
        return self._sidebar_widget
    
    def setup_sidebar_connections(self):
        """设置侧边栏信号连接"""
        # 文件选择按钮
        self.image_sequence_btn.clicked.connect(self.select_image_sequence)
        self.video_file_btn.clicked.connect(self.select_video_file)
        self.temp_files_btn.clicked.connect(self.select_temp_files)
        self.clear_btn.clicked.connect(self.clear_inputs)
        
        # 参数变化连接
        self.explosive_type.currentTextChanged.connect(self.on_parameter_changed)
        self.equivalent.textChanged.connect(self.on_parameter_changed)
        self.al_percent.textChanged.connect(self.on_parameter_changed)
        self.env_temp.textChanged.connect(self.on_parameter_changed)
        self.env_humidity.textChanged.connect(self.on_parameter_changed)
        self.env_pressure.textChanged.connect(self.on_parameter_changed)
        self.explosion_duration.textChanged.connect(self.on_parameter_changed)
        self.explosion_duration.textChanged.connect(self.on_duration_changed)
        self.pixel_length.textChanged.connect(self.on_parameter_changed)
    
    
    def select_image_sequence(self):
        """选择图像序列文件夹"""
        # 选择文件夹
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图像序列文件夹", ""
        )
        
        if folder_path:
            # 检查文件夹中的图像文件
            import os
            import glob
            
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            image_files = []
            
            for ext in image_extensions:
                pattern = os.path.join(folder_path, ext)
                image_files.extend(glob.glob(pattern))
                pattern = os.path.join(folder_path, ext.upper())
                image_files.extend(glob.glob(pattern))
            
            if image_files:
                # 按文件名排序
                image_files.sort()
                self.image_files = image_files
                self.image_folder_path = folder_path
                self.current_image_index = 0
                
                # 获取爆炸时长
                try:
                    duration = int(self.explosion_duration.text())
                except ValueError:
                    duration = 140  # 默认值
                
                # 设置时间轴范围
                self.time_slider.setRange(0, len(image_files) - 1)
                self.time_slider.setValue(0)
                
                # 显示第一张图像
                self.display_image_at_index(0)
                
                self.input_status.setText(f"已选择图像序列: {len(image_files)} 个文件，时长: {duration}ms")
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中没有找到图像文件！")
    
    def select_video_file(self):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件",
            "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*)"
        )
        
        if file_path:
            self.input_status.setText(f"已选择视频文件: {os.path.basename(file_path)}")
            # 这里可以保存视频文件路径
            self.video_file_path = file_path
    
    
    def select_temp_files(self):
        """选择温度时间序列文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择火球温度时间序列文件",
            "", "CSV文件 (*.csv);;JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*)"
        )
        if files:
            self.input_status.setText(f"已导入温度序列文件: {files[0]}")
    
    def clear_inputs(self):
        """清空输入"""
        self.input_status.setText("已清空")
        # 清空文件路径和图像序列
        self.image_folder_path = None
        self.video_file_path = None
        self.image_files = []
        self.current_image_index = 0
        
        # 重置时间轴
        self.time_slider.setRange(0, 100)
        self.time_slider.setValue(0)
        self.time_label.setText("t = 0 ms")
        
        # 清空图像预览
        self.image_preview.clear()
    
    def on_duration_changed(self):
        """爆炸时长变化时的回调"""
        if self.image_files:
            # 重新计算当前时间显示
            current_value = self.time_slider.value()
            self.on_time_changed(current_value)
    
    def on_parameter_changed(self):
        """参数变化时的回调"""
        # 更新右侧参数显示
        self.update_parameters_display(
            self.explosive_type.currentText(),
            self.equivalent.text(),
            self.al_percent.text(),
            self.env_temp.text(),
            self.env_humidity.text(),
            self.env_pressure.text(),
            self.explosion_duration.text(),
            self.pixel_length.text()
        )
