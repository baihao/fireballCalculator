#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出模块标签页
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QTextEdit, QPushButton, QComboBox, QCheckBox, QGroupBox)
from PySide6.QtCore import QTimer


class ExportTab(QWidget):
    """导出模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("导出预览与日志"))
        layout.addLayout(toolbar)
        
        # 日志显示区域
        self.log_text = QTextEdit()
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0b1220;
                color: #cbd5e1;
                border: 1px solid #1f2937;
                border-radius: 10px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.log_text.setPlainText("[日志] 准备导出…")
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
    
    def get_sidebar_widget(self):
        """获取导出模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            from PySide6.QtWidgets import QGroupBox, QComboBox, QCheckBox
            from PySide6.QtCore import Qt
            
            self._sidebar_widget = QGroupBox("结果导出")
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)  # 只向上对齐
            
            layout.addWidget(QLabel("选择导出内容"))
            self.exp_features = QCheckBox("特征时间序列")
            self.exp_features.setChecked(True)
            self.exp_predictions = QCheckBox("预测结果")
            self.exp_predictions.setChecked(True)
            self.exp_errors = QCheckBox("误差分析")
            layout.addWidget(self.exp_features)
            layout.addWidget(self.exp_predictions)
            layout.addWidget(self.exp_errors)
            
            layout.addWidget(QLabel("导出格式"))
            self.exp_format = QComboBox()
            self.exp_format.addItems(["CSV", "JSON", "Excel (xlsx)"])
            layout.addWidget(self.exp_format)
            
            self.export_btn = QPushButton("导出")
            self.export_btn.setStyleSheet("QPushButton { background-color: #0ea5e9; color: white; }")
            layout.addWidget(self.export_btn)
            
            layout.addStretch()
            self.export_status = QLabel("等待导出")
            self.export_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
            layout.addWidget(self.export_status)
            
            self._sidebar_widget.setLayout(layout)
        
        return self._sidebar_widget
