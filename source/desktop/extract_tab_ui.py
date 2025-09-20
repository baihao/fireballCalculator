#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块UI构建器
负责创建和配置所有UI组件
"""

import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox,
                               QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QTextEdit, QScrollArea)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget
from interactive_image_widget import create_interactive_image_widget


class ExtractTabUI:
    """特征提取模块UI构建器"""
    
    def __init__(self):
        """初始化UI构建器"""
        self.ui_components = {}
        
    def create_main_layout(self, parent_widget: QWidget) -> QHBoxLayout:
        """
        创建主界面布局
        
        Args:
            parent_widget: 父控件
            
        Returns:
            QHBoxLayout: 主布局
        """
        layout = QHBoxLayout()
        
        # 创建左侧图像预览区域
        left_widget = self._create_left_panel()
        
        # 创建右侧图表区域
        right_widget = self._create_right_panel()
        
        # 创建侧边栏组件（为了确保所有组件都被创建）
        self._create_sidebar_components()
        
        # 添加到分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        parent_widget.setLayout(layout)
        
        return layout
    
    def _create_sidebar_components(self):
        """创建侧边栏组件（不添加到布局，只创建组件引用）"""
        # 序列文件选择按钮
        self.ui_components['sequence_btn'] = QPushButton("选择火球爆炸序列文件")
        
        # Prompt相关组件
        self.ui_components['prompt_btn'] = QPushButton("开始选择参考点")
        self.ui_components['point_type_group'] = QButtonGroup()
        self.ui_components['positive_radio'] = QRadioButton("选择正点")
        self.ui_components['negative_radio'] = QRadioButton("选择负点")
        self.ui_components['ignition_radio'] = QRadioButton("选择起爆点")
        self.ui_components['positive_radio'].setChecked(True)
        
        self.ui_components['point_type_group'].addButton(self.ui_components['positive_radio'])
        self.ui_components['point_type_group'].addButton(self.ui_components['negative_radio'])
        self.ui_components['point_type_group'].addButton(self.ui_components['ignition_radio'])
        
        self.ui_components['cancel_extract_btn'] = QPushButton("清除当前图片上参考点")
        
        # Prompt信息显示
        self.ui_components['prompt_info_text'] = QTextEdit()
        self.ui_components['prompt_info_text'].setReadOnly(True)
        self.ui_components['prompt_info_text'].setMaximumHeight(200)
        self.ui_components['prompt_info_text'].setStyleSheet("""
            QTextEdit {
                background-color: #111827;
                border: 1px solid #374151;
                border-radius: 5px;
                color: #e5e7eb;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.ui_components['prompt_info_text'].setPlaceholderText("选择参考点后，信息将在此显示...")
        
        # 特征提取按钮
        self.ui_components['extract_btn'] = QPushButton("开始特征提取")
        self.ui_components['extract_btn'].setStyleSheet("QPushButton { background-color: #0ea5e9; color: white; }")
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧图像预览面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        
        # 顶部工具栏
        toolbar = self._create_toolbar()
        left_layout.addLayout(toolbar)
        
        # 图像预览组
        preview_group = self._create_preview_group()
        left_layout.addWidget(preview_group)
        
        # 添加弹性空间，将状态标签推到最下面
        left_layout.addStretch()
        
        # 状态标签
        self.ui_components['extract_status'] = QLabel("待开始")
        self.ui_components['extract_status'].setStyleSheet("color: #9ca3af; font-size: 12px; padding: 10px; text-align: center;")
        self.ui_components['extract_status'].setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.ui_components['extract_status'])
        
        left_widget.setLayout(left_layout)
        return left_widget
    
    def _create_toolbar(self) -> QHBoxLayout:
        """创建顶部工具栏"""
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("提取结果预览与质量检查"))
        toolbar.addStretch()
        
        self.ui_components['progress_label'] = QLabel("0%")
        self.ui_components['progress_label'].setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.ui_components['progress_label'])
        
        return toolbar
    
    def _create_preview_group(self) -> QGroupBox:
        """创建图像预览组"""
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
        preview_group_layout.setAlignment(Qt.AlignTop)
        preview_group_layout.setSpacing(8)
        
        # 交互式图像控件
        self.ui_components['extract_preview'] = create_interactive_image_widget()
        
        # 创建水平布局来居中图像预览
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.ui_components['extract_preview'])
        image_layout.addStretch()
        preview_group_layout.addLayout(image_layout)
        
        # 图片导航控件
        self._add_image_navigation_controls(preview_group_layout)
        
        # 时间轴
        timeline_layout = self._create_timeline()
        preview_group_layout.addLayout(timeline_layout)
        
        preview_group.setLayout(preview_group_layout)
        return preview_group
    
    def _add_image_navigation_controls(self, parent_layout: QVBoxLayout):
        """添加图片导航控件"""
        # 图片索引信息标签
        self.ui_components['image_index_label'] = QLabel("0/0")
        self.ui_components['image_index_label'].setAlignment(Qt.AlignCenter)
        self.ui_components['image_index_label'].setStyleSheet("""
            QLabel {
                color: #9ca3af;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 5px;
            }
        """)
        parent_layout.addWidget(self.ui_components['image_index_label'])
        
        # 图片跳转控件
        jump_layout = QHBoxLayout()
        jump_layout.setAlignment(Qt.AlignTop)
        
        jump_layout.addWidget(QLabel("跳转到图片:"))
        
        self.ui_components['jump_input'] = QLineEdit()
        self.ui_components['jump_input'].setPlaceholderText("输入图片编号")
        self.ui_components['jump_input'].setMaximumWidth(100)
        self.ui_components['jump_input'].setStyleSheet("""
            QLineEdit {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 5px;
                color: #e5e7eb;
                padding: 5px;
                font-size: 12px;
            }
        """)
        jump_layout.addWidget(self.ui_components['jump_input'])
        
        self.ui_components['jump_btn'] = QPushButton("查看")
        self.ui_components['jump_btn'].setMaximumWidth(60)
        self.ui_components['jump_btn'].setStyleSheet("""
            QPushButton {
                background-color: #374151;
                color: #e5e7eb;
                border: 1px solid #4b5563;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
            QPushButton:pressed {
                background-color: #6b7280;
            }
        """)
        jump_layout.addWidget(self.ui_components['jump_btn'])
        
        parent_layout.addLayout(jump_layout)
    
    def _create_timeline(self) -> QHBoxLayout:
        """创建时间轴控件"""
        timeline_layout = QHBoxLayout()
        timeline_layout.setAlignment(Qt.AlignTop)
        
        self.ui_components['extract_slider'] = QSlider(Qt.Horizontal)
        self.ui_components['extract_slider'].setRange(0, 100)
        
        self.ui_components['extract_time_label'] = QLabel("t = 0 ms")
        self.ui_components['extract_time_label'].setStyleSheet("color: #9ca3af; font-size: 12px;")
        
        timeline_layout.addWidget(self.ui_components['extract_slider'])
        timeline_layout.addWidget(self.ui_components['extract_time_label'])
        
        return timeline_layout
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧图表面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(10)
        
        # 温度图表组
        temp_group = self._create_temperature_chart_group()
        right_layout.addWidget(temp_group)
        
        # 直径图表组
        diam_group = self._create_diameter_chart_group()
        right_layout.addWidget(diam_group)
        
        # 控制按钮
        button_layout = self._create_control_buttons()
        right_layout.addLayout(button_layout)
        
        right_widget.setLayout(right_layout)
        return right_widget
    
    def _create_temperature_chart_group(self) -> QGroupBox:
        """创建温度图表组"""
        temp_group = QGroupBox("火球温度随时间变化")
        temp_group.setStyleSheet(self._get_chart_group_style())
        
        temp_layout = QVBoxLayout()
        temp_layout.setAlignment(Qt.AlignTop)
        
        self.ui_components['temp_chart'] = MatplotlibWidget(width=4, height=2.5)
        temp_layout.addWidget(self.ui_components['temp_chart'])
        
        temp_group.setLayout(temp_layout)
        return temp_group
    
    def _create_diameter_chart_group(self) -> QGroupBox:
        """创建直径图表组"""
        diam_group = QGroupBox("火球直径随时间变化")
        diam_group.setStyleSheet(self._get_chart_group_style())
        
        diam_layout = QVBoxLayout()
        diam_layout.setAlignment(Qt.AlignTop)
        
        self.ui_components['diam_chart'] = MatplotlibWidget(width=4, height=2.5)
        diam_layout.addWidget(self.ui_components['diam_chart'])
        
        diam_group.setLayout(diam_layout)
        return diam_group
    
    def _create_control_buttons(self) -> QHBoxLayout:
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignTop)
        
        self.ui_components['save_button'] = QPushButton("保存提取序列")
        self.ui_components['save_button'].setEnabled(False)
        button_layout.addWidget(self.ui_components['save_button'])
        
        button_layout.addStretch()
        button_layout.addWidget(QLabel("有参考点后可保存"))
        
        return button_layout
    
    def create_sidebar_widget(self) -> QGroupBox:
        """创建侧边栏组件"""
        # 确保侧边栏组件已创建
        if 'sequence_btn' not in self.ui_components:
            self._create_sidebar_components()
        
        sidebar_widget = QGroupBox("特征提取")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)
        
        # 序列文件选择
        layout.addWidget(QLabel("火球爆炸序列文件（JSON格式）"))
        layout.addWidget(self.ui_components['sequence_btn'])
        
        # 参考点选择组
        prompt_group = self._create_prompt_selection_group()
        layout.addWidget(prompt_group)
        
        # 特征提取按钮
        extract_layout = QVBoxLayout()
        extract_layout.setAlignment(Qt.AlignTop)
        extract_layout.addWidget(self.ui_components['extract_btn'])
        layout.addLayout(extract_layout)
        
        # 添加弹性空间
        layout.addStretch()
        
        sidebar_widget.setLayout(layout)
        return sidebar_widget
    
    def _create_prompt_selection_group(self) -> QGroupBox:
        """创建参考点选择组"""
        prompt_group = QGroupBox("参考点选择")
        prompt_group.setStyleSheet("""
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
        
        prompt_layout = QVBoxLayout()
        prompt_layout.setAlignment(Qt.AlignTop)
        prompt_layout.setSpacing(8)
        
        # 开始选择参考点按钮
        prompt_layout.addWidget(self.ui_components['prompt_btn'])
        
        # 正负点和起爆点选择单选按钮组（纵向排列）
        point_type_layout = QVBoxLayout()
        point_type_layout.setAlignment(Qt.AlignTop)
        point_type_layout.setSpacing(5)  # 设置按钮间距
        
        point_type_layout.addWidget(self.ui_components['positive_radio'])
        point_type_layout.addWidget(self.ui_components['negative_radio'])
        point_type_layout.addWidget(self.ui_components['ignition_radio'])
        prompt_layout.addLayout(point_type_layout)
        
        # 清除按钮
        prompt_layout.addWidget(self.ui_components['cancel_extract_btn'])
        
        # 参考点信息显示区域
        prompt_layout.addWidget(QLabel("已选择的参考点信息"))
        prompt_layout.addWidget(self.ui_components['prompt_info_text'])
        
        prompt_group.setLayout(prompt_layout)
        return prompt_group
    
    def _get_chart_group_style(self) -> str:
        """获取图表组样式"""
        return """
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
        """
    
    def init_temperature_chart(self):
        """初始化温度图表"""
        try:
            temp_chart = self.ui_components['temp_chart']
            temp_chart.clear()
            
            ax = temp_chart.figure.add_subplot(111)
            
            # 设置图表样式
            temp_chart.figure.patch.set_facecolor('#111827')
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
            temp_chart.figure.tight_layout(pad=1.0)
            temp_chart.canvas.draw()
            
        except Exception as e:
            print(f"初始化温度图表失败: {e}")
    
    def init_diameter_chart(self):
        """初始化直径图表"""
        try:
            diam_chart = self.ui_components['diam_chart']
            diam_chart.clear()
            
            ax = diam_chart.figure.add_subplot(111)
            
            # 设置图表样式
            diam_chart.figure.patch.set_facecolor('#111827')
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
            diam_chart.figure.tight_layout(pad=1.0)
            diam_chart.canvas.draw()
            
        except Exception as e:
            print(f"初始化直径图表失败: {e}")
    
    def get_ui_components(self) -> dict:
        """
        获取所有UI组件的引用
        
        Returns:
            dict: UI组件字典
        """
        return self.ui_components.copy()
