#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆炸火球分析系统 - 桌面应用框架
主窗口和通用组件定义
"""

import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                               QLineEdit, QComboBox, QSlider, QProgressBar,
                               QTextEdit, QFileDialog, QCheckBox, QGroupBox,
                               QGridLayout, QSplitter, QFrame, QScrollArea)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QPen
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

# 导入计算模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fireball_radius_calculator import FireballCalculator
from fireball_temperature_calculator import FireballTemperatureCalculator
from transmissivity_calculator import TransmissivityParams, transmissivity
from fireball_heat_radiation_calculator import (compute_temperature_profile, 
                                               compute_diameter_profile,
                                               compute_heat_flux_over_time,
                                               integrate_heat_radiation)


class MatplotlibWidget(QWidget):
    """自定义matplotlib图表组件"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def clear(self):
        self.figure.clear()
        self.canvas.draw()
        
    def plot_line(self, x_data, y_data, title="", xlabel="", ylabel="", color='#38bdf8'):
        """绘制单条线图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x_data, y_data, color=color, linewidth=2)
        ax.set_title(title, fontsize=12, color='#38bdf8')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#0b1220')
        self.figure.patch.set_facecolor('#0b1220')
        ax.tick_params(colors='#e5e7eb')
        self.canvas.draw()
        
    def plot_multiple_lines(self, data_dict, title="", xlabel="", ylabel=""):
        """绘制多条线图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        colors = ['#5f0f40', '#2a4d69', '#4f9d9d', '#8acb88', '#f4d35e']
        
        for i, (label, (x_data, y_data)) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            ax.plot(x_data, y_data, color=color, linewidth=2, label=label)
            
        ax.set_title(title, fontsize=12, color='#38bdf8')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_facecolor('#0b1220')
        self.figure.patch.set_facecolor('#0b1220')
        ax.tick_params(colors='#e5e7eb')
        self.canvas.draw()


class ImagePreviewWidget(QWidget):
    """图像预览组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMaximumSize(600, 450)  # 设置最大尺寸
        self.setFixedSize(500, 375)    # 设置固定尺寸
        self.setStyleSheet("""
            QWidget {
                background-color: #050a16;
                border: 1px solid #1f2937;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # 设置边距
        self.image_label = QLabel("预览图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)  # 禁用自动缩放内容
        self.image_label.setStyleSheet("""
            QLabel {
                color: #9ca3af;
                font-size: 14px;
                background-color: transparent;
            }
        """)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
    def set_image(self, image_path):
        """设置预览图像"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # 使用固定尺寸缩放，不改变组件大小
            # 获取标签的可用尺寸（减去边距）
            label_size = self.image_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                # 缩放图像以适应标签尺寸，保持宽高比
                scaled_pixmap = pixmap.scaled(
                    label_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                # 如果标签还没有尺寸，使用组件尺寸
                widget_size = self.size()
                if widget_size.width() > 0 and widget_size.height() > 0:
                    scaled_pixmap = pixmap.scaled(
                        widget_size, 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)
                else:
                    self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("预览图像")
    
    def clear(self):
        """清空图像预览"""
        self.image_label.clear()
        self.image_label.setText("预览图像")


class SidebarWidget(QWidget):
    """侧边栏组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 侧边栏容器，由各个标签页填充
        self.sidebar_container = QWidget()
        layout.addWidget(self.sidebar_container)
        
        self.setLayout(layout)
        
    def set_sidebar_content(self, content_widget):
        """设置侧边栏内容"""
        # 如果还没有布局，创建一个
        if not self.sidebar_container.layout():
            layout = QVBoxLayout()
            self.sidebar_container.setLayout(layout)
        
        # 隐藏所有现有的侧边栏内容
        for i in range(self.sidebar_container.layout().count()):
            child = self.sidebar_container.layout().itemAt(i)
            if child and child.widget():
                child.widget().hide()
        
        # 如果控件不在布局中，添加到布局
        if content_widget.parent() != self.sidebar_container:
            self.sidebar_container.layout().addWidget(content_widget)
        
        # 显示新的侧边栏内容
        content_widget.show()
        


class FireballAnalysisApp(QMainWindow):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_connections()
        self.apply_dark_theme()
        
    def init_ui(self):
        self.setWindowTitle("爆炸火球分析系统 - 桌面UI")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧侧边栏
        self.sidebar = SidebarWidget()
        self.sidebar.setFixedWidth(300)
        main_layout.addWidget(self.sidebar)
        
        # 右侧标签页区域
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1f2937;
                background-color: #111827;
            }
            QTabBar::tab {
                background-color: #0b1220;
                color: #e5e7eb;
                padding: 8px 12px;
                margin-right: 2px;
                border: 1px solid #1f2937;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #111827;
                border-color: #38bdf8;
            }
            QTabBar::tab:hover {
                background-color: #1f2937;
            }
        """)
        
        # 导入各个标签页
        from input_tab import InputTab
        from extract_tab import ExtractTab
        from model_tab import ModelTab
        from export_tab import ExportTab
        
        # 创建各个标签页
        self.input_tab = InputTab()
        self.extract_tab = ExtractTab()
        self.model_tab = ModelTab()
        self.export_tab = ExportTab()
        
        self.tab_widget.addTab(self.input_tab, "输入")
        self.tab_widget.addTab(self.extract_tab, "特征提取")
        self.tab_widget.addTab(self.model_tab, "建模与预测")
        self.tab_widget.addTab(self.export_tab, "导出")
        
        main_layout.addWidget(self.tab_widget)
        
        central_widget.setLayout(main_layout)
        
        # 状态栏
        self.statusBar().showMessage("状态: 空闲")
        
    def setup_connections(self):
        """设置信号连接"""
        # 标签页切换
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # 延迟初始化侧边栏，确保标签页完全创建
        QTimer.singleShot(100, lambda: self.on_tab_changed(0))
        
    def apply_dark_theme(self):
        """应用深色主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #e5e7eb;
            }
            QWidget {
                background-color: #0f172a;
                color: #e5e7eb;
            }
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
            QPushButton {
                background-color: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 8px 12px;
                color: #e5e7eb;
            }
            QPushButton:hover {
                background-color: #1f2937;
            }
            QPushButton:pressed {
                background-color: #374151;
            }
            QLineEdit, QComboBox {
                background-color: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 8px 10px;
                color: #e5e7eb;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #38bdf8;
            }
            QLabel {
                color: #9ca3af;
                font-size: 12px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #1f2937;
                height: 8px;
                background: #0b1220;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #38bdf8;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
    def on_tab_changed(self, index):
        """标签页切换事件"""
        # 预加载所有侧边栏内容（如果还没有加载）
        if not hasattr(self, '_sidebars_loaded'):
            self._load_all_sidebars()
            self._sidebars_loaded = True
        
        if index == 0:  # 输入
            self.sidebar.set_sidebar_content(self.input_tab.get_sidebar_widget())
        elif index == 1:  # 特征提取
            self.sidebar.set_sidebar_content(self.extract_tab.get_sidebar_widget())
        elif index == 2:  # 建模与预测
            self.sidebar.set_sidebar_content(self.model_tab.get_sidebar_widget())
        elif index == 3:  # 导出
            self.sidebar.set_sidebar_content(self.export_tab.get_sidebar_widget())
    
    def _load_all_sidebars(self):
        """预加载所有侧边栏内容"""
        # 获取所有侧边栏组件
        input_sidebar = self.input_tab.get_sidebar_widget()
        extract_sidebar = self.extract_tab.get_sidebar_widget()
        model_sidebar = self.model_tab.get_sidebar_widget()
        export_sidebar = self.export_tab.get_sidebar_widget()
        
        # 将所有侧边栏添加到布局中（初始隐藏）
        layout = QVBoxLayout()
        layout.addWidget(input_sidebar)
        layout.addWidget(extract_sidebar)
        layout.addWidget(model_sidebar)
        layout.addWidget(export_sidebar)
        
        # 隐藏除输入外的所有侧边栏
        extract_sidebar.hide()
        model_sidebar.hide()
        export_sidebar.hide()
        
        # 显示输入侧边栏
        input_sidebar.show()
        
        self.sidebar.sidebar_container.setLayout(layout)
            
