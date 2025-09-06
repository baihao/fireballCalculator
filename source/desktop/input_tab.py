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
        self.temperature_file_path = None  # 温度文件路径
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
        
        # 初始化图表（不绘制曲线）
        self.init_temperature_chart()
        
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
    
    def init_temperature_chart(self):
        """初始化温度图表（不绘制曲线）"""
        try:
            # 清除图表
            self.temp_chart.clear()
            
            # 创建一个空的图表，只显示提示信息
            ax = self.temp_chart.figure.add_subplot(111)
            
            # 设置图表样式
            self.temp_chart.figure.patch.set_facecolor('#111827')  # 背景色
            ax.set_facecolor('#111827')  # 坐标轴区域背景色
            
            # 设置坐标轴颜色
            ax.tick_params(colors='#9ca3af', labelsize=10)  # 刻度颜色和大小
            ax.spines['bottom'].set_color('#374151')  # x轴颜色
            ax.spines['top'].set_color('#374151')
            ax.spines['left'].set_color('#374151')  # y轴颜色
            ax.spines['right'].set_color('#374151')
            
            # 设置标签颜色
            ax.set_xlabel("时间 (ms)", color='#e5e7eb', fontsize=11)
            ax.set_ylabel("温度 (K)", color='#e5e7eb', fontsize=11)
            ax.set_title("爆炸温度变化", color='#38bdf8', fontsize=12, fontweight='bold')
            
            # 设置坐标轴范围
            ax.set_xlim(0, 140)
            ax.set_ylim(1000, 1600)
            
            # 显示网格
            ax.grid(True, alpha=0.3, color='#374151')
            
            # 显示提示文本
            ax.text(70, 1300, "请导入温度时间序列数据", 
                   ha='center', va='center', 
                   color='#9ca3af', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1f2937', alpha=0.8))
            
            # 调整布局，确保标签完全显示
            self.temp_chart.figure.tight_layout(pad=1.5)
            
            # 刷新显示
            self.temp_chart.canvas.draw()
            
        except Exception as e:
            print(f"初始化温度图表失败: {e}")
    
    def update_temperature_chart(self, time_data=None, temp_data=None):
        """更新温度图表（绘制曲线）"""
        try:
            print(f"update_temperature_chart 被调用: time_data={time_data is not None}, temp_data={temp_data is not None}")
            
            if time_data is not None and temp_data is not None:
                # 使用导入的数据绘制曲线
                print(f"绘制导入的数据曲线: {len(time_data)} 个点")
                self.temp_chart.plot_line(
                    time_data, temp_data,
                    title="爆炸温度变化",
                    xlabel="时间 (ms)",
                    ylabel="温度 (K)",
                    color='#38bdf8'
                )
            else:
                # 使用默认计算数据绘制曲线
                print("绘制默认计算数据曲线")
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
            
            print("温度图表绘制完成")
            
        except Exception as e:
            print(f"更新温度图表失败: {e}")
            import traceback
            traceback.print_exc()
    
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
                    "temperature_file": getattr(self, 'temperature_file_path', "未设置")
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
            try:
                # 读取温度数据文件
                file_path = files[0]
                print(f"正在读取温度数据文件: {file_path}")
                time_data, temp_data = self.load_temperature_data(file_path)
                
                if time_data is not None and temp_data is not None:
                    print(f"成功读取数据: 时间点 {len(time_data)} 个, 温度点 {len(temp_data)} 个")
                    print(f"时间范围: {min(time_data)} - {max(time_data)} ms")
                    print(f"温度范围: {min(temp_data)} - {max(temp_data)} K")
                    
                    # 保存温度文件路径
                    self.temperature_file_path = file_path
                    
                    # 更新温度图表
                    self.update_temperature_chart(time_data, temp_data)
                    self.input_status.setText(f"已导入温度序列文件: {os.path.basename(file_path)}")
                    print("温度图表已更新")
                else:
                    print("读取温度数据失败: 返回None")
                    QMessageBox.warning(self, "警告", "无法读取温度数据文件！")
                    
            except Exception as e:
                print(f"读取温度数据文件时出错: {e}")
                QMessageBox.critical(self, "错误", f"读取温度数据文件失败:\n{str(e)}")
    
    def load_temperature_data(self, file_path):
        """加载温度数据文件"""
        try:
            if file_path.endswith('.csv'):
                # 使用pandas读取CSV文件
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    
                    # 查找时间和温度列
                    time_col = None
                    temp_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'time' in col_lower or '时间' in col_lower or 'ms' in col_lower:
                            time_col = col
                        elif 'temp' in col_lower or '温度' in col_lower or 'k' in col_lower:
                            temp_col = col
                    
                    if time_col is None or temp_col is None:
                        # 如果找不到列名，使用前两列
                        time_col = df.columns[0]
                        temp_col = df.columns[1]
                    
                    time_data = df[time_col].values
                    temp_data = df[temp_col].values
                    
                    return time_data, temp_data
                    
                except ImportError:
                    # 如果没有pandas，使用标准库读取CSV
                    print("pandas未安装，使用标准库读取CSV文件")
                    return self._read_csv_manual(file_path)
                
            elif file_path.endswith('.json'):
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 假设JSON格式为 {"time": [...], "temperature": [...]}
                if 'time' in data and 'temperature' in data:
                    return data['time'], data['temperature']
                else:
                    return None, None
                    
            else:
                # 尝试读取文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                time_data = []
                temp_data = []
                
                for line in lines:
                    if ',' in line:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            try:
                                time_data.append(float(parts[0]))
                                temp_data.append(float(parts[1]))
                            except ValueError:
                                continue
                
                if time_data and temp_data:
                    return time_data, temp_data
                else:
                    return None, None
                    
        except Exception as e:
            print(f"加载温度数据失败: {e}")
            return None, None
    
    def _read_csv_manual(self, file_path):
        """手动读取CSV文件（不使用pandas）"""
        try:
            time_data = []
            temp_data = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 跳过标题行
            for i, line in enumerate(lines[1:], 1):
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            time_val = float(parts[0])
                            temp_val = float(parts[1])
                            time_data.append(time_val)
                            temp_data.append(temp_val)
                        except ValueError as e:
                            print(f"第{i+1}行数据格式错误: {line}, 错误: {e}")
                            continue
            
            if time_data and temp_data:
                print(f"成功读取 {len(time_data)} 个数据点")
                return time_data, temp_data
            else:
                print("没有找到有效的数据")
                return None, None
                
        except Exception as e:
            print(f"手动读取CSV失败: {e}")
            return None, None
    
    def clear_inputs(self):
        """清空输入"""
        self.input_status.setText("已清空")
        # 清空文件路径和图像序列
        self.image_folder_path = None
        self.video_file_path = None
        self.temperature_file_path = None
        self.image_files = []
        self.current_image_index = 0
        
        # 重置时间轴
        self.time_slider.setRange(0, 100)
        self.time_slider.setValue(0)
        self.time_label.setText("t = 0 ms")
        
        # 清空图像预览
        self.image_preview.clear()
        
        # 重新初始化温度图表
        self.init_temperature_chart()
    
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
