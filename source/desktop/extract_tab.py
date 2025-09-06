#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块标签页
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget


class ExtractTab(QWidget):
    """特征提取模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # 左侧图像预览
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("提取结果预览与质量检查"))
        toolbar.addStretch()
        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        toolbar.addWidget(self.progress_label)
        left_layout.addLayout(toolbar)
        
        self.extract_preview = ImagePreviewWidget()
        left_layout.addWidget(self.extract_preview)
        
        # 时间轴
        timeline_layout = QHBoxLayout()
        self.extract_slider = QSlider(Qt.Horizontal)
        self.extract_slider.setRange(0, 100)
        self.extract_time_label = QLabel("t = 0 ms")
        self.extract_time_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        timeline_layout.addWidget(self.extract_slider)
        timeline_layout.addWidget(self.extract_time_label)
        left_layout.addLayout(timeline_layout)
        
        left_widget.setLayout(left_layout)
        
        # 右侧图表和控制
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # 直径图表
        self.diam_chart = MatplotlibWidget(width=4, height=3)
        right_layout.addWidget(self.diam_chart)
        
        # 控制按钮
        button_layout = QHBoxLayout()
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
    
    def get_sidebar_widget(self):
        """获取特征提取模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            from PySide6.QtWidgets import QGroupBox, QComboBox, QLineEdit
            
            self._sidebar_widget = QGroupBox("特征提取")
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)  # 只向上对齐
            
            layout.addWidget(QLabel("点位/掩膜辅助（可选，JSON/GeoJSON/自定义）"))
            self.hints_btn = QPushButton("选择辅助文件")
            layout.addWidget(self.hints_btn)
            
            layout.addWidget(QLabel("模型选择"))
            self.extract_model = QComboBox()
            self.extract_model.addItems(["U-Net 分割", "DeepLabV3", "SAM"])
            layout.addWidget(self.extract_model)
            
            layout.addWidget(QLabel("批处理大小"))
            self.batch_size = QLineEdit("8")
            layout.addWidget(self.batch_size)
            
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
            
            self._sidebar_widget.setLayout(layout)
        
        return self._sidebar_widget
