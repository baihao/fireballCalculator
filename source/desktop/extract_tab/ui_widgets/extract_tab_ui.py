#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块UI构建器
负责创建和配置所有UI组件
"""

import os
from typing import Any, Dict
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox,
                               QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QTextEdit, QPlainTextEdit,
                               QScrollArea, QCheckBox)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget
from .checkbar import create_checkbar
from chart_widgets.diameter_chart import DiameterChart
from chart_widgets.diameter_velocity_chart import DiameterVelocityChart
from .interactive_image_widget import create_interactive_image_widget


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
        # 数据源按钮
        self.ui_components['sequence_btn'] = QPushButton("导入爆炸序列文件")
        self.ui_components['image_folder_btn'] = QPushButton("导入火球图像序列")
        
        # Prompt相关组件
        self.ui_components['prompt_btn'] = QPushButton("开始选择参考点")
        self.ui_components['finish_prompt_btn'] = QPushButton("参考点选择完成")
        self.ui_components['finish_prompt_btn'].setEnabled(False)
        self.ui_components['point_type_group'] = QButtonGroup()
        self.ui_components['positive_radio'] = QRadioButton("选择正点")
        self.ui_components['negative_radio'] = QRadioButton("选择负点")
        self.ui_components['ignition_radio'] = QRadioButton("选择起爆点")
        self.ui_components['positive_radio'].setChecked(True)
        
        self.ui_components['point_type_group'].addButton(self.ui_components['positive_radio'])
        self.ui_components['point_type_group'].addButton(self.ui_components['negative_radio'])
        self.ui_components['point_type_group'].addButton(self.ui_components['ignition_radio'])
        
        self.ui_components['cancel_prompt_btn'] = QPushButton("清除当前图片上参考点")
        
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
        
        # 重新提取按钮
        self.ui_components['reextract_btn'] = QPushButton("重新提取")
        self.ui_components['reextract_btn'].setStyleSheet("QPushButton { background-color: #f59e0b; color: white; }")
        self.ui_components['reextract_btn'].setVisible(False)  # 初始隐藏
        
        # 保存相关组件
        self.ui_components['export_segmentation_checkbox'] = QCheckBox("同时导出分割图片")
        self.ui_components['export_segmentation_checkbox'].setChecked(False)
        self.ui_components['save_button'] = QPushButton("保存结果序列")
        self.ui_components['save_button'].setEnabled(False)
    
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
        
        # 爆炸信息 / 炸药参数（与原型一致：两分组并排，组内纵向；仅占预览列宽度）
        params_row = QHBoxLayout()
        params_row.setSpacing(10)

        explosion_group = QGroupBox("图片参数")
        explosion_group.setStyleSheet(self._param_group_style())
        eg_layout = QVBoxLayout()
        eg_layout.setSpacing(6)
        eg_layout.addWidget(QLabel("帧率（fps）"))
        self.ui_components['mv_frame_rate_fps'] = QLineEdit("1000")
        self._style_param_lineedit(self.ui_components['mv_frame_rate_fps'])
        eg_layout.addWidget(self.ui_components['mv_frame_rate_fps'])
        eg_layout.addWidget(QLabel("视场范围(m)"))
        self.ui_components['mv_field_of_view_m'] = QLineEdit("60")
        self._style_param_lineedit(self.ui_components['mv_field_of_view_m'])
        eg_layout.addWidget(self.ui_components['mv_field_of_view_m'])
        explosion_group.setLayout(eg_layout)

        explosive_group = QGroupBox("炸药参数")
        explosive_group.setStyleSheet(self._param_group_style())
        ex_layout = QVBoxLayout()
        ex_layout.setSpacing(6)
        ex_layout.addWidget(QLabel("当量（kg TNT）"))
        self.ui_components['mv_equivalent'] = QLineEdit("1")
        self._style_param_lineedit(self.ui_components['mv_equivalent'])
        ex_layout.addWidget(self.ui_components['mv_equivalent'])
        ex_layout.addWidget(QLabel("含铝量（%）"))
        self.ui_components['mv_al_percent'] = QLineEdit("30")
        self._style_param_lineedit(self.ui_components['mv_al_percent'])
        ex_layout.addWidget(self.ui_components['mv_al_percent'])
        explosive_group.setLayout(ex_layout)

        params_row.addWidget(explosion_group, 1)
        params_row.addWidget(explosive_group, 1)
        left_layout.addLayout(params_row)

        # 运行日志（多行，不占右侧图表区）
        log_label = QLabel("运行日志")
        log_label.setStyleSheet("color: #9ca3af; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(log_label)
        self.ui_components['run_log'] = QPlainTextEdit()
        self.ui_components['run_log'].setReadOnly(True)
        self.ui_components['run_log'].setPlaceholderText("[日志] 导入、分割等输出将显示在此处…")
        self.ui_components['run_log'].setMinimumHeight(100)
        self.ui_components['run_log'].setMaximumHeight(200)
        self.ui_components['run_log'].setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 8px;
                color: #cbd5e1;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        left_layout.addWidget(self.ui_components['run_log'])
        # 兼容旧代码：extract_status 指向运行日志
        self.ui_components['extract_status'] = self.ui_components['run_log']

        left_layout.addStretch(1)
        
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
        # 分组检查条（替换原图片索引标签）
        self.ui_components['check_bar'] = create_checkbar()
        parent_layout.addWidget(self.ui_components['check_bar'])
        
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
        # 收紧右侧整体留白
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(6, 6, 6, 6)
        
        metrics_group = self._create_key_metrics_group()
        right_layout.addWidget(metrics_group)
        
        # 直径图表组
        diam_group = self._create_diameter_chart_group()
        right_layout.addWidget(diam_group)
        
        # 直径变化速率图表组（显示在直径图表下方）
        diam_vel_group = self._create_diameter_velocity_chart_group()
        right_layout.addWidget(diam_vel_group)
        
        right_layout.setStretchFactor(metrics_group, 1)
        right_layout.setStretchFactor(diam_group, 1)
        right_layout.setStretchFactor(diam_vel_group, 1)
        
        right_widget.setLayout(right_layout)
        return right_widget
    
    def _create_key_metrics_group(self) -> QGroupBox:
        """创建关键参数数值面板（替代原温度图）。"""
        metrics_group = QGroupBox("关键参数")
        metrics_group.setStyleSheet(self._get_chart_group_style())

        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(6, 4, 6, 6)
        metrics_layout.setSpacing(4)

        self.ui_components["key_metrics_panel"] = QPlainTextEdit()
        self.ui_components["key_metrics_panel"].setReadOnly(True)
        self.ui_components["key_metrics_panel"].setPlaceholderText(
            "分割与拖曳拟合完成后，将在此显示标定、分割质量、直径统计与 K/B/C 等关键数值…"
        )
        self.ui_components["key_metrics_panel"].setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 8px;
                color: #cbd5e1;
                font-family: ui-monospace, 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        metrics_layout.addWidget(self.ui_components["key_metrics_panel"])
        metrics_group.setLayout(metrics_layout)
        return metrics_group

    def _create_diameter_chart_group(self) -> QGroupBox:
        """创建直径图表组"""
        diam_group = QGroupBox("火球直径随时间变化")
        diam_group.setStyleSheet(self._get_chart_group_style())
        
        diam_layout = QVBoxLayout()
        diam_layout.setAlignment(Qt.AlignTop)
        # 收紧组内边距
        diam_layout.setContentsMargins(6, 4, 6, 6)
        diam_layout.setSpacing(4)
        
        self.ui_components['diam_chart'] = DiameterChart(width=4, height=3.0)
        diam_layout.addWidget(self.ui_components['diam_chart'])
        
        diam_group.setLayout(diam_layout)
        return diam_group
    
    def _create_diameter_velocity_chart_group(self) -> QGroupBox:
        """创建直径变化速率图表组"""
        diam_vel_group = QGroupBox("火球直径变化速率随时间变化")
        diam_vel_group.setStyleSheet(self._get_chart_group_style())
        
        diam_vel_layout = QVBoxLayout()
        diam_vel_layout.setAlignment(Qt.AlignTop)
        # 收紧组内边距
        diam_vel_layout.setContentsMargins(6, 4, 6, 6)
        diam_vel_layout.setSpacing(4)
        
        self.ui_components['diam_vel_chart'] = DiameterVelocityChart(width=4, height=3.0)
        diam_vel_layout.addWidget(self.ui_components['diam_vel_chart'])
        
        diam_vel_group.setLayout(diam_vel_layout)
        return diam_vel_group
    
    
    def _param_group_style(self) -> str:
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #374151;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #111827;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #38bdf8;
            }
        """

    def _style_param_lineedit(self, w: QLineEdit) -> None:
        w.setStyleSheet("""
            QLineEdit {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 8px;
                color: #e5e7eb;
            }
        """)

    def _style_param_combo(self, w: QComboBox) -> None:
        w.setStyleSheet("""
            QComboBox {
                background-color: #0b1220;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 8px;
                color: #e5e7eb;
            }
        """)

    def create_sidebar_widget(self) -> QGroupBox:
        """创建侧边栏组件"""
        # 确保侧边栏组件已创建
        if 'sequence_btn' not in self.ui_components:
            self._create_sidebar_components()
        
        sidebar_widget = QGroupBox("机器视觉")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        data_group = QGroupBox("数据源")
        data_group.setStyleSheet(self._param_group_style())
        data_layout = QVBoxLayout()
        data_layout.setSpacing(8)
        data_layout.addWidget(QLabel("导入爆炸序列文件（JSON）"))
        data_layout.addWidget(self.ui_components['sequence_btn'])
        data_layout.addWidget(QLabel("导入火球图像序列（选择文件夹）"))
        data_layout.addWidget(self.ui_components['image_folder_btn'])
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 参考点选择组
        prompt_group = self._create_prompt_selection_group()
        layout.addWidget(prompt_group)
        
        # 特征提取按钮
        extract_layout = QVBoxLayout()
        extract_layout.setAlignment(Qt.AlignTop)
        extract_layout.addWidget(self.ui_components['extract_btn'])
        extract_layout.addWidget(self.ui_components['reextract_btn'])
        layout.addLayout(extract_layout)
        
        # 导出分割结果组
        export_group = self._create_export_group()
        layout.addWidget(export_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        sidebar_widget.setLayout(layout)
        return sidebar_widget
    
    def _create_prompt_selection_group(self) -> QGroupBox:
        """创建参考点选择组"""
        prompt_group = QGroupBox("参考点与分割")
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
        
        # 开始选择参考点 / 完成选择（独立按钮）
        prompt_layout.addWidget(self.ui_components['prompt_btn'])
        prompt_layout.addWidget(self.ui_components['finish_prompt_btn'])
        
        # 正负点和起爆点选择单选按钮组（纵向排列）
        point_type_layout = QVBoxLayout()
        point_type_layout.setAlignment(Qt.AlignTop)
        point_type_layout.setSpacing(5)  # 设置按钮间距
        
        point_type_layout.addWidget(self.ui_components['positive_radio'])
        point_type_layout.addWidget(self.ui_components['negative_radio'])
        point_type_layout.addWidget(self.ui_components['ignition_radio'])
        prompt_layout.addLayout(point_type_layout)
        
        # 清除按钮
        prompt_layout.addWidget(self.ui_components['cancel_prompt_btn'])
        
        # 参考点信息显示区域
        prompt_layout.addWidget(QLabel("已选择的参考点信息"))
        prompt_layout.addWidget(self.ui_components['prompt_info_text'])
        
        prompt_group.setLayout(prompt_layout)
        return prompt_group
    
    def _create_export_group(self) -> QGroupBox:
        """创建导出分割结果组"""
        export_group = QGroupBox("输出")
        export_group.setStyleSheet("""
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
        
        export_layout = QVBoxLayout()
        export_layout.setAlignment(Qt.AlignTop)
        export_layout.setSpacing(8)
        
        # 同时导出分割图片复选框
        export_layout.addWidget(self.ui_components['export_segmentation_checkbox'])
        
        # 保存结果序列按钮
        export_layout.addWidget(self.ui_components['save_button'])
        
        export_group.setLayout(export_layout)
        return export_group
    
    def _get_chart_group_style(self) -> str:
        """获取图表组样式"""
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #1f2937;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 6px;
                background-color: #111827;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px 0 5px;
                color: #38bdf8;
            }
        """
    
    def init_diameter_chart(self):
        """初始化直径图表"""
        try:
            diam_chart: DiameterChart = self.ui_components['diam_chart']
            diam_chart.reset()
            
        except Exception as e:
            print(f"初始化直径图表失败: {e}")
    
    def init_diameter_velocity_chart(self):
        """初始化直径变化速率图表"""
        try:
            diam_vel_chart: DiameterVelocityChart = self.ui_components['diam_vel_chart']
            diam_vel_chart.reset()
        except Exception as e:
            print(f"初始化直径变化速率图表失败: {e}")
    
    def get_ui_components(self) -> dict:
        """
        获取所有UI组件的引用
        
        Returns:
            dict: UI组件字典
        """
        return self.ui_components.copy()
