#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块标签页
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider)
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
