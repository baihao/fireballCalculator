#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出模块标签页
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QTextEdit)
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
