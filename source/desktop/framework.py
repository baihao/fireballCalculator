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
        self.setStyleSheet("""
            QWidget {
                background-color: #050a16;
                border: 1px solid #1f2937;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        self.image_label = QLabel("预览图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                color: #9ca3af;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
    def set_image(self, image_path):
        """设置预览图像"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("预览图像")


class SidebarWidget(QWidget):
    """侧边栏组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 输入模块侧栏
        self.input_sidebar = self.create_input_sidebar()
        layout.addWidget(self.input_sidebar)
        
        # 其他模块侧栏（初始隐藏）
        self.extract_sidebar = self.create_extract_sidebar()
        self.extract_sidebar.hide()
        layout.addWidget(self.extract_sidebar)
        
        self.model_sidebar = self.create_model_sidebar()
        self.model_sidebar.hide()
        layout.addWidget(self.model_sidebar)
        
        self.export_sidebar = self.create_export_sidebar()
        self.export_sidebar.hide()
        layout.addWidget(self.export_sidebar)
        
        self.setLayout(layout)
        
    def create_input_sidebar(self):
        """创建输入模块侧栏"""
        group = QGroupBox("图像与参数输入")
        layout = QVBoxLayout()
        
        # 文件选择
        self.image_files_btn = QPushButton("导入火球图像序列或视频文件")
        self.temp_files_btn = QPushButton("导入火球温度时间序列")
        
        layout.addWidget(QLabel("导入火球图像序列（多选）或视频文件"))
        layout.addWidget(self.image_files_btn)
        layout.addWidget(QLabel("导入火球温度时间序列（CSV/JSON）"))
        layout.addWidget(self.temp_files_btn)
        
        # 炸药参数
        layout.addWidget(QLabel("炸药类别"))
        self.explosive_type = QComboBox()
        self.explosive_type.addItems(["TNT", "Polyurethane", "30% Al / Rubber", 
                                    "40% Al / Rubber", "50% Al / Rubber", "60% Al / Rubber"])
        self.explosive_type.setCurrentText("40% Al / Rubber")
        layout.addWidget(self.explosive_type)
        
        # 当量和含铝量
        layout.addWidget(QLabel("当量（kg TNT）"))
        self.equivalent = QLineEdit("10")
        layout.addWidget(self.equivalent)
        
        layout.addWidget(QLabel("含铝量（%）"))
        self.al_percent = QLineEdit("30")
        layout.addWidget(self.al_percent)
        
        # 环境参数
        layout.addWidget(QLabel("环境温度 Ta（°C）"))
        self.env_temp = QLineEdit("24")
        layout.addWidget(self.env_temp)
        
        layout.addWidget(QLabel("相对湿度 RH（%）"))
        self.env_humidity = QLineEdit("48")
        layout.addWidget(self.env_humidity)
        
        layout.addWidget(QLabel("水饱和气压 PwSat(Ta)（Pa）"))
        self.env_pressure = QLineEdit("2987.87")
        layout.addWidget(self.env_pressure)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空输入")
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        self.input_status = QLabel("等待文件导入…")
        self.input_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
        button_layout.addWidget(self.input_status)
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        return group
        
    def create_extract_sidebar(self):
        """创建特征提取侧栏"""
        group = QGroupBox("特征提取")
        layout = QVBoxLayout()
        
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
        
        group.setLayout(layout)
        return group
        
    def create_model_sidebar(self):
        """创建建模与预测侧栏"""
        group = QGroupBox("建模与预测")
        layout = QVBoxLayout()
        
        # 训练部分
        layout.addWidget(QLabel("建模 - 训练"))
        layout.addWidget(QLabel("选择训练时间序列（可多选）"))
        self.train_series_btn = QPushButton("选择训练文件")
        layout.addWidget(self.train_series_btn)
        
        layout.addWidget(QLabel("算法"))
        self.algo = QComboBox()
        self.algo.addItems(["工程进算法", "T-Transformer"])
        layout.addWidget(self.algo)
        
        # 学习率和轮次
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学习率:"))
        self.lr = QLineEdit("0.0005")
        lr_layout.addWidget(self.lr)
        lr_layout.addWidget(QLabel("轮次:"))
        self.epochs = QLineEdit("50")
        lr_layout.addWidget(self.epochs)
        layout.addLayout(lr_layout)
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setStyleSheet("QPushButton { background-color: #0ea5e9; color: white; }")
        layout.addWidget(self.train_btn)
        
        # 预测部分
        layout.addWidget(QLabel("预测 - 运行"))
        layout.addWidget(QLabel("选择已训练模型"))
        self.model_list = QComboBox()
        self.model_list.addItems(["示例模型 v1"])
        layout.addWidget(self.model_list)
        
        # 预测参数
        pred_layout = QHBoxLayout()
        pred_layout.addWidget(QLabel("当量:"))
        self.p_eq = QLineEdit("10")
        pred_layout.addWidget(self.p_eq)
        pred_layout.addWidget(QLabel("含铝量:"))
        self.p_al = QLineEdit("30")
        pred_layout.addWidget(self.p_al)
        layout.addLayout(pred_layout)
        
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(QLabel("仿真步长:"))
        self.p_step = QLineEdit("1")
        sim_layout.addWidget(self.p_step)
        sim_layout.addWidget(QLabel("仿真时长:"))
        self.p_duration = QLineEdit("140")
        sim_layout.addWidget(self.p_duration)
        layout.addLayout(sim_layout)
        
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setStyleSheet("QPushButton { background-color: #10b981; color: white; }")
        layout.addWidget(self.predict_btn)
        
        group.setLayout(layout)
        return group
        
    def create_export_sidebar(self):
        """创建导出侧栏"""
        group = QGroupBox("结果导出")
        layout = QVBoxLayout()
        
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
        
        group.setLayout(layout)
        return group


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
        
        # 侧边栏按钮连接
        self.sidebar.image_files_btn.clicked.connect(self.select_image_files)
        self.sidebar.temp_files_btn.clicked.connect(self.select_temp_files)
        self.sidebar.clear_btn.clicked.connect(self.clear_inputs)
        
        # 参数变化连接
        self.sidebar.explosive_type.currentTextChanged.connect(self.update_input_parameters)
        self.sidebar.equivalent.textChanged.connect(self.update_input_parameters)
        self.sidebar.al_percent.textChanged.connect(self.update_input_parameters)
        self.sidebar.env_temp.textChanged.connect(self.update_input_parameters)
        self.sidebar.env_humidity.textChanged.connect(self.update_input_parameters)
        self.sidebar.env_pressure.textChanged.connect(self.update_input_parameters)
        
        self.sidebar.extract_btn.clicked.connect(self.start_extraction)
        self.sidebar.cancel_extract_btn.clicked.connect(self.cancel_extraction)
        
        self.sidebar.train_btn.clicked.connect(self.start_training)
        self.sidebar.predict_btn.clicked.connect(self.start_prediction)
        
        self.sidebar.export_btn.clicked.connect(self.start_export)
        
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
        # 显示对应的侧边栏
        self.sidebar.input_sidebar.hide()
        self.sidebar.extract_sidebar.hide()
        self.sidebar.model_sidebar.hide()
        self.sidebar.export_sidebar.hide()
        
        if index == 0:  # 输入
            self.sidebar.input_sidebar.show()
        elif index == 1:  # 特征提取
            self.sidebar.extract_sidebar.show()
        elif index == 2:  # 建模与预测
            self.sidebar.model_sidebar.show()
        elif index == 3:  # 导出
            self.sidebar.export_sidebar.show()
            
    def select_image_files(self):
        """选择图像文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择火球图像序列或视频文件",
            "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if files:
            self.sidebar.input_status.setText(f"已选择 {len(files)} 个文件")
            self.statusBar().showMessage(f"已导入 {len(files)} 个文件")
            
    def select_temp_files(self):
        """选择温度时间序列文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择火球温度时间序列文件",
            "", "CSV文件 (*.csv);;JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*)"
        )
        if files:
            self.statusBar().showMessage(f"已导入温度序列文件: {files[0]}")
            
    def clear_inputs(self):
        """清空输入"""
        self.sidebar.input_status.setText("已清空")
        self.statusBar().showMessage("状态: 空闲")
        
    def update_input_parameters(self):
        """更新输入标签页的参数显示"""
        material_type = self.sidebar.explosive_type.currentText()
        equivalent = self.sidebar.equivalent.text()
        al_percent = self.sidebar.al_percent.text()
        env_temp = self.sidebar.env_temp.text()
        env_humidity = self.sidebar.env_humidity.text()
        env_pressure = self.sidebar.env_pressure.text()
        
        self.input_tab.update_parameters_display(
            material_type, equivalent, al_percent, 
            env_temp, env_humidity, env_pressure
        )
        
    def start_extraction(self):
        """开始特征提取"""
        self.sidebar.extract_status.setText("正在提取…")
        self.statusBar().showMessage("状态: 特征提取中")
        
        # 模拟提取过程
        QTimer.singleShot(2000, self.extraction_completed)
        
    def extraction_completed(self):
        """特征提取完成"""
        self.sidebar.extract_status.setText("已完成")
        self.statusBar().showMessage("状态: 空闲")
        
    def cancel_extraction(self):
        """取消特征提取"""
        self.sidebar.extract_status.setText("已取消")
        self.statusBar().showMessage("状态: 空闲")
        
    def start_training(self):
        """开始训练"""
        self.statusBar().showMessage("状态: 训练中")
        QTimer.singleShot(3000, self.training_completed)
        
    def training_completed(self):
        """训练完成"""
        self.statusBar().showMessage("状态: 空闲")
        
    def start_prediction(self):
        """开始预测"""
        self.statusBar().showMessage("状态: 预测中")
        self.model_tab.update_prediction_charts()
        QTimer.singleShot(1500, self.prediction_completed)
        
    def prediction_completed(self):
        """预测完成"""
        self.statusBar().showMessage("状态: 空闲")
        
    def start_export(self):
        """开始导出"""
        format_type = self.sidebar.exp_format.currentText()
        items = []
        if self.sidebar.exp_features.isChecked():
            items.append("特征时间序列")
        if self.sidebar.exp_predictions.isChecked():
            items.append("预测结果")
        if self.sidebar.exp_errors.isChecked():
            items.append("误差分析")
            
        log_msg = f"[导出] 内容: {', '.join(items)} | 格式: {format_type} | 时间: {QTimer().currentTime().toString()}"
        self.export_tab.log_text.append(log_msg)
        self.sidebar.export_status.setText("导出完成(示例)")
        self.statusBar().showMessage("导出完成")
