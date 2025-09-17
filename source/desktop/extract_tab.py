#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块标签页
"""

import numpy as np
import json
import os
import glob
import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox,
                               QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QTextEdit, QScrollArea)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget

# 添加路径以导入火球计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fireball_radius_calculator import FireballCalculator


class ExtractTab(QWidget):
    """特征提取模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化图像序列相关属性
        self.image_paths = []  # 图像路径列表（统一使用）
        self.current_image_index = 0  # 当前显示的图像索引
        self.sequence_folder_path = None  # 序列文件夹路径
        self.sequence_data = None  # 序列数据
        self.explosion_duration = 140  # 爆炸时长（毫秒）
        
        # 初始化prompt选择相关属性
        self.is_prompt_selection_mode = False  # 是否处于prompt点选择模式
        self.prompt_data = {}  # prompt数据：{image_index: {"points": [[x,y], ...], "labels": [1,0,1,...]}}
        self.current_prompt_points = []  # 当前图像的prompt点临时存储
        
        # 初始化火球计算器
        self.fireball_calculator = FireballCalculator()
        
        self.init_ui()
        self.setup_connections()
        self.init_charts()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # 左侧图像预览
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)  # 只向上对齐
        
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("提取结果预览与质量检查"))
        toolbar.addStretch()
        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.progress_label)
        left_layout.addLayout(toolbar)
        
        # 图像预览和时间轴组合
        preview_group = QGroupBox("火球爆炸序列预览")
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
        preview_group_layout = QVBoxLayout()
        preview_group_layout.setAlignment(Qt.AlignTop)  # 只向上对齐
        preview_group_layout.setSpacing(8)
        
        self.extract_preview = ImagePreviewWidget()
        # 创建水平布局来居中图像预览
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.extract_preview)
        image_layout.addStretch()
        preview_group_layout.addLayout(image_layout)
        
        # 时间轴
        timeline_layout = QHBoxLayout()
        timeline_layout.setAlignment(Qt.AlignTop)  # 只向上对齐
        self.extract_slider = QSlider(Qt.Horizontal)
        self.extract_slider.setRange(0, 100)
        self.extract_time_label = QLabel("t = 0 ms")
        self.extract_time_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        timeline_layout.addWidget(self.extract_slider)
        timeline_layout.addWidget(self.extract_time_label)
        preview_group_layout.addLayout(timeline_layout)
        
        preview_group.setLayout(preview_group_layout)
        left_layout.addWidget(preview_group)
        
        left_widget.setLayout(left_layout)
        
        # 右侧图表区域
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)  # 只向上对齐
        right_layout.setSpacing(10)
        
        # 温度图表
        temp_group = QGroupBox("火球温度随时间变化")
        temp_group.setStyleSheet("""
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
        temp_layout = QVBoxLayout()
        temp_layout.setAlignment(Qt.AlignTop)
        
        self.temp_chart = MatplotlibWidget(width=4, height=2.5)
        temp_layout.addWidget(self.temp_chart)
        temp_group.setLayout(temp_layout)
        right_layout.addWidget(temp_group)
        
        # 直径图表
        diam_group = QGroupBox("火球直径随时间变化")
        diam_group.setStyleSheet("""
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
        diam_layout = QVBoxLayout()
        diam_layout.setAlignment(Qt.AlignTop)
        
        self.diam_chart = MatplotlibWidget(width=4, height=2.5)
        diam_layout.addWidget(self.diam_chart)
        diam_group.setLayout(diam_layout)
        right_layout.addWidget(diam_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignTop)
        self.save_button = QPushButton("保存提取序列")
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(QLabel("提取完成后可保存"))
        right_layout.addLayout(button_layout)
        
        right_widget.setLayout(right_layout)
        
        # 添加到主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def setup_connections(self):
        """设置信号连接"""
        self.extract_slider.valueChanged.connect(self.on_time_changed)
        
        # 连接特征提取按钮
        if hasattr(self, 'extract_btn'):
            self.extract_btn.clicked.connect(self.start_feature_extraction)
    
    def get_sidebar_widget(self):
        """获取特征提取模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            from PySide6.QtWidgets import QGroupBox, QComboBox, QLineEdit
            
            self._sidebar_widget = QGroupBox("特征提取")
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)  # 只向上对齐
            
            layout.addWidget(QLabel("火球爆炸序列文件（JSON格式）"))
            self.sequence_btn = QPushButton("选择火球爆炸序列文件")
            layout.addWidget(self.sequence_btn)
            
            # Prompt点选择
            layout.addWidget(QLabel("Prompt点选择"))
            self.prompt_btn = QPushButton("开始选择prompt点")
            layout.addWidget(self.prompt_btn)
            
            # 正负点选择单选按钮组
            point_type_layout = QHBoxLayout()
            self.point_type_group = QButtonGroup()
            self.positive_radio = QRadioButton("选择正点")
            self.negative_radio = QRadioButton("选择负点")
            self.positive_radio.setChecked(True)  # 默认选择正点
            
            self.point_type_group.addButton(self.positive_radio)
            self.point_type_group.addButton(self.negative_radio)
            
            point_type_layout.addWidget(self.positive_radio)
            point_type_layout.addWidget(self.negative_radio)
            layout.addLayout(point_type_layout)
            
            button_layout = QHBoxLayout()
            self.extract_btn = QPushButton("开始特征提取")
            self.extract_btn.setStyleSheet("QPushButton { background-color: #0ea5e9; color: white; }")
            self.cancel_extract_btn = QPushButton("取消")
            button_layout.addWidget(self.extract_btn)
            button_layout.addWidget(self.cancel_extract_btn)
            button_layout.addStretch()
            self.extract_status = QLabel("待开始")
            self.extract_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
            button_layout.addWidget(self.extract_status)
            layout.addLayout(button_layout)
            
            # 添加prompt点信息显示区域
            layout.addWidget(QLabel("已选择的Prompt点信息"))
            self.prompt_info_text = QTextEdit()
            self.prompt_info_text.setReadOnly(True)
            self.prompt_info_text.setMaximumHeight(200)
            self.prompt_info_text.setStyleSheet("""
                QTextEdit {
                    background-color: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 5px;
                    color: #e5e7eb;
                    font-family: 'Courier New', monospace;
                    font-size: 11px;
                    padding: 8px;
                }
            """)
            self.prompt_info_text.setPlaceholderText("选择prompt点后，信息将在此显示...")
            layout.addWidget(self.prompt_info_text)
            
            self._sidebar_widget.setLayout(layout)
            
            # 设置信号连接
            self.sequence_btn.clicked.connect(self.select_sequence_folder)
            self.prompt_btn.clicked.connect(self.toggle_prompt_selection)
            self.extract_btn.clicked.connect(self.start_feature_extraction)
        
        return self._sidebar_widget
    
    def on_time_changed(self, value):
        """时间轴变化"""
        if self.image_paths:
            # 计算实际时间（毫秒）
            total_frames = len(self.image_paths)
            if total_frames > 1:
                time_ms = (value / (total_frames - 1)) * self.explosion_duration
            else:
                time_ms = 0
            self.extract_time_label.setText(f"t = {time_ms:.1f} ms (帧 {value + 1}/{total_frames})")
            
            # 显示对应的图像
            self.display_image_at_index(value)
        else:
            self.extract_time_label.setText(f"t = {value} ms")
    
    def display_image_at_index(self, index):
        """显示指定索引的图像"""
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return
        
        try:
            image_path = self.image_paths[index]
            # 使用 ImagePreviewWidget 的 set_image 方法
            self.extract_preview.set_image(image_path)
            self.current_image_index = index
            print(f"显示图像: {image_path} (索引: {index})")
                
        except Exception as e:
            print(f"显示图像失败: {e}")
    
    def select_sequence_folder(self):
        """选择火球爆炸序列JSON文件"""
        # 选择JSON文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择火球爆炸序列文件",
            "", "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.sequence_data = json.load(f)
                
                # 解析图像序列路径
                image_folder_path = self.sequence_data.get('files', {}).get('image_folder', '')
                if not image_folder_path or image_folder_path == "未设置":
                    QMessageBox.warning(self, "警告", "JSON文件中没有有效的图像序列路径！")
                    self.extract_status.setText("无效的图像序列路径")
                    return
                
                # 获取爆炸时长
                self.explosion_duration = int(self.sequence_data.get('parameters', {}).get('explosion_duration', 140))
                
                # 加载图像序列
                success = self.load_image_sequence(image_folder_path)
                
                if success:
                    # 加载温度数据
                    temp_success = self.load_temperature_data()
                    
                    self.extract_status.setText(f"已加载序列: {len(self.image_paths)} 个文件，时长: {self.explosion_duration}ms")
                    print(f"成功加载火球序列: {len(self.image_paths)} 个文件")
                    if temp_success:
                        print("温度数据加载成功")
                    else:
                        print("温度数据加载失败")
                else:
                    self.extract_status.setText("加载图像序列失败")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取序列文件失败:\n{str(e)}")
                self.extract_status.setText("读取文件失败")
                print(f"读取序列文件失败: {e}")
    
    def load_image_sequence(self, folder_path):
        """加载图像序列"""
        try:
            if not os.path.exists(folder_path):
                QMessageBox.warning(self, "警告", f"图像序列文件夹不存在: {folder_path}")
                return False
            
            # 检查文件夹中的图像文件
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
                self.image_paths = image_files  # 统一使用image_paths
                self.sequence_folder_path = folder_path
                self.current_image_index = 0
                
                # 设置时间轴范围
                self.extract_slider.setRange(0, len(self.image_paths) - 1)
                self.extract_slider.setValue(0)
                
                # 显示第一张图像
                self.display_image_at_index(0)
                
                return True
            else:
                QMessageBox.warning(self, "警告", f"文件夹中没有找到图像文件: {folder_path}")
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像序列失败:\n{str(e)}")
            print(f"加载图像序列失败: {e}")
            return False
    
    def init_charts(self):
        """初始化图表"""
        # 初始化温度图表
        self.init_temperature_chart()
        
        # 初始化直径图表
        self.init_diameter_chart()
    
    def init_temperature_chart(self):
        """初始化温度图表"""
        try:
            # 清除图表并设置基本样式
            self.temp_chart.clear()
            
            # 添加子图
            ax = self.temp_chart.figure.add_subplot(111)
            
            # 设置图表样式
            self.temp_chart.figure.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            
            # 设置坐标轴颜色
            ax.tick_params(colors='#9ca3af', labelsize=9)
            ax.spines['bottom'].set_color('#374151')
            ax.spines['top'].set_color('#374151')
            ax.spines['left'].set_color('#374151')
            ax.spines['right'].set_color('#374151')
            
            # 设置标签颜色
            ax.set_xlabel("时间 (ms)", color='#e5e7eb', fontsize=10)
            ax.set_ylabel("温度 (K)", color='#e5e7eb', fontsize=10)
            ax.set_title("火球温度随时间变化", color='#38bdf8', fontsize=11, fontweight='bold')
            
            # 设置坐标轴范围
            ax.set_xlim(0, 140)
            ax.set_ylim(1000, 1600)
            
            # 显示网格
            ax.grid(True, alpha=0.3, color='#374151')
            
            # 显示提示文本
            ax.text(70, 1300, "请加载序列文件", 
                   ha='center', va='center', 
                   color='#9ca3af', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1f2937', alpha=0.8))
            
            # 调整布局
            self.temp_chart.figure.tight_layout(pad=1.0)
            self.temp_chart.canvas.draw()
            
        except Exception as e:
            print(f"初始化温度图表失败: {e}")
    
    def init_diameter_chart(self):
        """初始化直径图表"""
        try:
            # 清除图表并设置基本样式
            self.diam_chart.clear()
            
            # 添加子图
            ax = self.diam_chart.figure.add_subplot(111)
            
            # 设置图表样式
            self.diam_chart.figure.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            
            # 设置坐标轴颜色
            ax.tick_params(colors='#9ca3af', labelsize=9)
            ax.spines['bottom'].set_color('#374151')
            ax.spines['top'].set_color('#374151')
            ax.spines['left'].set_color('#374151')
            ax.spines['right'].set_color('#374151')
            
            # 设置标签颜色
            ax.set_xlabel("时间 (ms)", color='#e5e7eb', fontsize=10)
            ax.set_ylabel("直径 (m)", color='#e5e7eb', fontsize=10)
            ax.set_title("火球直径随时间变化", color='#38bdf8', fontsize=11, fontweight='bold')
            
            # 设置坐标轴范围
            ax.set_xlim(0, 140)
            ax.set_ylim(0, 2)
            
            # 显示网格
            ax.grid(True, alpha=0.3, color='#374151')
            
            # 显示提示文本
            ax.text(70, 1, "提取完成后显示", 
                   ha='center', va='center', 
                   color='#9ca3af', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#1f2937', alpha=0.8))
            
            # 调整布局
            self.diam_chart.figure.tight_layout(pad=1.0)
            self.diam_chart.canvas.draw()
            
        except Exception as e:
            print(f"初始化直径图表失败: {e}")
    
    def load_temperature_data(self):
        """加载温度数据并更新图表"""
        try:
            print("开始加载温度数据...")
            
            if not self.sequence_data:
                print("❌ 没有序列数据，无法加载温度数据")
                return False
            
            # 获取温度文件路径
            temp_file_path = self.sequence_data.get('files', {}).get('temperature_file', '')
            print(f"温度文件路径: {temp_file_path}")
            
            if not temp_file_path or temp_file_path == "未设置":
                print("❌ 没有温度文件路径")
                return False
            
            # 检查文件是否存在
            if not os.path.exists(temp_file_path):
                print(f"❌ 温度文件不存在: {temp_file_path}")
                return False
            
            print(f"✅ 温度文件存在，开始读取...")
            
            # 读取温度数据
            time_data, temp_data = self._read_temperature_csv(temp_file_path)
            if time_data is None or temp_data is None:
                print("❌ 读取温度数据失败")
                return False
            
            print(f"✅ 温度数据读取成功: {len(time_data)} 个点")
            
            # 更新温度图表
            self.update_temperature_chart(time_data, temp_data)
            print("✅ 温度图表更新完成")
            return True
            
        except Exception as e:
            print(f"❌ 加载温度数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _read_temperature_csv(self, file_path):
        """读取温度CSV文件"""
        try:
            print(f"开始读取CSV文件: {file_path}")
            time_data = []
            temp_data = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"文件总行数: {len(lines)}")
            if len(lines) > 0:
                print(f"第一行（标题）: {lines[0].strip()}")
            
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
                    else:
                        print(f"第{i+1}行列数不足: {line}")
                else:
                    print(f"第{i+1}行格式错误: {line}")
            
            print(f"成功解析 {len(time_data)} 个数据点")
            
            if time_data and temp_data:
                print(f"✅ 成功读取温度数据: {len(time_data)} 个点")
                print(f"   时间范围: {min(time_data)} - {max(time_data)} ms")
                print(f"   温度范围: {min(temp_data)} - {max(temp_data)} K")
                return time_data, temp_data
            else:
                print("❌ 没有找到有效的温度数据")
                return None, None
                
        except Exception as e:
            print(f"❌ 读取温度CSV失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def update_temperature_chart(self, time_data, temp_data):
        """更新温度图表"""
        try:
            print(f"开始更新温度图表: {len(time_data)} 个数据点")
            
            # 使用 MatplotlibWidget 的 plot_line 方法
            print(f"绘制温度曲线: 时间范围 {min(time_data)}-{max(time_data)} ms, 温度范围 {min(temp_data)}-{max(temp_data)} K")
            
            self.temp_chart.plot_line(
                time_data, temp_data,
                title="火球温度随时间变化",
                xlabel="时间 (ms)",
                ylabel="温度 (K)",
                color='#38bdf8'
            )
            
            print("✅ 温度图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新温度图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_diameter_chart(self, time_data, diameter_data):
        """更新直径图表（提取完成后调用）"""
        try:
            print(f"开始更新直径图表: {len(time_data)} 个数据点")
            
            # 使用 MatplotlibWidget 的 plot_line 方法
            self.diam_chart.plot_line(
                time_data, diameter_data,
                title="火球直径随时间变化",
                xlabel="时间 (ms)",
                ylabel="直径 (m)",
                color='#f59e0b'
            )
            
            print("✅ 直径图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新直径图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def start_feature_extraction(self):
        """开始特征提取"""
        try:
            print("🔥 开始特征提取...")
            self.extract_status.setText("正在提取特征...")
            self.extract_btn.setEnabled(False)
            
            # 检查是否有序列数据
            if not self.sequence_data:
                QMessageBox.warning(self, "警告", "请先加载火球爆炸序列文件！")
                self.extract_status.setText("请先加载序列文件")
                self.extract_btn.setEnabled(True)
                return
            
            # 获取材料类型（从序列数据中获取）
            material_type = self.sequence_data.get('parameters', {}).get('material_type', '40%Al/Rubber')
            
            # 映射材料类型到计算器中的名称
            material_mapping = {
                '40% Al / Rubber': '40%Al/Rubber',
                '30% Al / Rubber': '30%Al/Rubber', 
                '50% Al / Rubber': '50%Al/Rubber',
                '60% Al / Rubber': '60%Al/Rubber',
                'Polyurethane': 'Polyurethane'
            }
            
            material_name = material_mapping.get(material_type, '40%Al/Rubber')
            print(f"使用材料类型: {material_name}")
            
            # 计算火球直径随时间变化
            self.calculate_fireball_diameter_curve(material_name)
            
            # 更新状态
            self.extract_status.setText("特征提取完成")
            self.save_button.setEnabled(True)
            self.extract_btn.setEnabled(True)
            
            print("✅ 特征提取完成！")
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            self.extract_status.setText("特征提取失败")
            self.extract_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"特征提取失败:\n{str(e)}")
    
    def calculate_fireball_diameter_curve(self, material_name):
        """计算火球直径随时间变化的曲线"""
        try:
            print(f"计算 {material_name} 材料的火球直径曲线...")
            
            # 生成时间序列 (0-140ms)
            time_points = 100  # 时间点数量
            t_ms = np.linspace(0, self.explosion_duration, time_points)  # 毫秒
            t_s = t_ms / 1000.0  # 转换为秒
            
            # 计算直径
            diameter_data = []
            for t in t_s:
                diameter = self.fireball_calculator.calculate_diameter(t, material_name)
                diameter_data.append(diameter)
            
            print(f"✅ 火球直径计算完成: {len(diameter_data)} 个数据点")
            print(f"   时间范围: {min(t_ms)} - {max(t_ms)} ms")
            print(f"   直径范围: {min(diameter_data):.3f} - {max(diameter_data):.3f} m")
            
            # 更新直径图表
            self.update_diameter_chart(t_ms, diameter_data)
            
            # 保存计算结果
            self.extraction_results = {
                'time_ms': t_ms.tolist(),
                'diameter_m': diameter_data,
                'material': material_name,
                'explosion_duration': self.explosion_duration
            }
            
        except Exception as e:
            print(f"❌ 计算火球直径曲线失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def toggle_prompt_selection(self):
        """切换prompt点选择模式"""
        try:
            if not self.is_prompt_selection_mode:
                # 开始选择prompt点
                self.is_prompt_selection_mode = True
                self.prompt_btn.setText("选择prompt点完成")
                self.extract_status.setText("正在选择prompt点...")
                print("🎯 开始prompt点选择模式")
            else:
                # 完成选择prompt点
                self.is_prompt_selection_mode = False
                self.prompt_btn.setText("开始选择prompt点")
                self.extract_status.setText("prompt点选择完成")
                print("✅ prompt点选择完成")
                
        except Exception as e:
            print(f"❌ 切换prompt选择模式失败: {e}")
            QMessageBox.critical(self, "错误", f"切换prompt选择模式失败:\n{str(e)}")
    
    def get_current_point_type(self):
        """获取当前选择的点类型"""
        if hasattr(self, 'positive_radio') and self.positive_radio.isChecked():
            return 'positive'
        elif hasattr(self, 'negative_radio') and self.negative_radio.isChecked():
            return 'negative'
        else:
            return 'positive'  # 默认返回正点
    
    def add_prompt_point(self, image_index: int, x: int, y: int, is_positive: bool = True):
        """
        添加prompt点到数据结构
        
        Args:
            image_index: 图像索引
            x, y: 点坐标
            is_positive: 是否为正点
        """
        try:
            # 确保数据结构存在
            if image_index not in self.prompt_data:
                self.prompt_data[image_index] = {
                    "points": [],
                    "labels": []
                }
            
            # 添加点坐标和标签
            self.prompt_data[image_index]["points"].append([x, y])
            self.prompt_data[image_index]["labels"].append(1 if is_positive else 0)
            
            print(f"添加prompt点: 图像{image_index}, 坐标({x}, {y}), 类型: {'正点' if is_positive else '负点'}")
            
            # 更新显示
            self.update_prompt_info_display()
            
        except Exception as e:
            print(f"❌ 添加prompt点失败: {e}")
    
    def remove_last_prompt_point(self, image_index: int):
        """
        移除指定图像的最后一个prompt点
        
        Args:
            image_index: 图像索引
        """
        try:
            if image_index in self.prompt_data:
                if self.prompt_data[image_index]["points"]:
                    removed_point = self.prompt_data[image_index]["points"].pop()
                    self.prompt_data[image_index]["labels"].pop()
                    print(f"移除prompt点: 图像{image_index}, 坐标{removed_point}")
                    
                    # 如果没有点了，删除整个条目
                    if not self.prompt_data[image_index]["points"]:
                        del self.prompt_data[image_index]
                    
                    # 更新显示
                    self.update_prompt_info_display()
                    
        except Exception as e:
            print(f"❌ 移除prompt点失败: {e}")
    
    def clear_prompt_data(self):
        """清空所有prompt数据"""
        self.prompt_data = {}
        self.update_prompt_info_display()
        print("🗑️ 已清空所有prompt数据")
    
    def update_prompt_info_display(self):
        """更新prompt点信息显示"""
        try:
            if not self.prompt_data:
                self.prompt_info_text.setPlainText("暂无选择的prompt点\n\n提示：\n1. 先加载图像序列\n2. 点击'开始选择prompt点'\n3. 在图像上点击选择正负点")
                return
            
            # 生成用户可读的信息
            info_lines = []
            
            for image_idx in sorted(self.prompt_data.keys()):
                points = self.prompt_data[image_idx]["points"]
                labels = self.prompt_data[image_idx]["labels"]
                
                # 分离正点和负点
                positive_points = []
                negative_points = []
                
                for point, label in zip(points, labels):
                    if label == 1:
                        positive_points.append(f"({point[0]}, {point[1]})")
                    else:
                        negative_points.append(f"({point[0]}, {point[1]})")
                
                # 格式化显示
                info_lines.append(f"第 {image_idx + 1} 张图片特征点：")
                
                if positive_points:
                    info_lines.append(f"  - 正点坐标：{', '.join(positive_points)}")
                
                if negative_points:
                    info_lines.append(f"  - 负点坐标：{', '.join(negative_points)}")
                
                info_lines.append("")  # 空行分隔
            
            # 添加统计信息
            total_images_with_prompts = len(self.prompt_data)
            total_points = sum(len(data["points"]) for data in self.prompt_data.values())
            total_positive = sum(sum(1 for label in data["labels"] if label == 1) for data in self.prompt_data.values())
            total_negative = total_points - total_positive
            
            info_lines.append("=" * 30)
            info_lines.append("统计信息：")
            info_lines.append(f"  - 有prompt点的图像：{total_images_with_prompts} 张")
            info_lines.append(f"  - 总点数：{total_points} 个")
            info_lines.append(f"  - 正点：{total_positive} 个")
            info_lines.append(f"  - 负点：{total_negative} 个")
            
            # 更新文本显示
            self.prompt_info_text.setPlainText("\n".join(info_lines))
            
        except Exception as e:
            print(f"❌ 更新prompt信息显示失败: {e}")
            self.prompt_info_text.setPlainText(f"显示错误: {str(e)}")
    
    def export_prompt_data_to_json(self, file_path: str):
        """
        导出prompt数据到JSON文件
        
        Args:
            file_path: 导出文件路径
        """
        try:
            export_data = {
                "image_paths": self.image_paths,
                "prompt_data": {str(k): v for k, v in self.prompt_data.items()}  # 键转换为字符串
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            
            print(f"✅ prompt数据已导出到: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出prompt数据失败: {e}")
            return False
    
    def test_prompt_data_functionality(self):
        """测试prompt数据功能"""
        try:
            print("🧪 测试prompt数据功能...")
            
            # 模拟添加一些prompt点
            self.add_prompt_point(0, 100, 100, True)   # 第1张图片，正点
            self.add_prompt_point(0, 90, 100, True)    # 第1张图片，正点
            self.add_prompt_point(0, 50, 50, False)    # 第1张图片，负点
            
            self.add_prompt_point(2, 120, 100, True)   # 第3张图片，正点
            self.add_prompt_point(2, 60, 100, False)   # 第3张图片，负点
            
            print("✅ prompt数据功能测试完成")
            
        except Exception as e:
            print(f"❌ prompt数据功能测试失败: {e}")
