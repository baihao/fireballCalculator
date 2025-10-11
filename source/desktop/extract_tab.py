#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块标签页
"""

import numpy as np
import json
import os
import sys
import subprocess
import threading
from typing import Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox,
                               QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QTextEdit, QScrollArea)
from PySide6.QtCore import Qt, Signal
from framework import MatplotlibWidget, ImagePreviewWidget
from segment_utils import build_time_diameter_series, run_segmentation_script
from sequence_manager import SequenceManager
from extract_tab_ui import ExtractTabUI
from info_builder import build_prompt_info_text, build_segmentation_info_text
from controllers.prompt_controller import PromptController

# 添加路径以导入火球计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fireball_radius_calculator import FireballCalculator
from diameter_process.diameter_drag_fitting import DiameterDragFitter


class ExtractTab(QWidget):
    """特征提取模块标签页"""
    # 异步分割：日志与完成信号
    log_received = Signal(str)
    seg_finished = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化图像序列相关属性
        self.image_paths = []  # 图像路径列表（统一使用）
        self.current_image_index = 0  # 当前显示的图像索引
        self.sequence_data = None  # 序列数据
        self.explosion_duration = 140  # 爆炸时长（毫秒）
        # 参数与分析缓存
        self.parameters = {}
        self.last_diameter_series = []  # List[Tuple[time_ms, diameter_m]]
        self.last_drag_fit_result = None
        
        # 分割结果相关属性
        self.segmentation_results = []  # 分割结果列表
        self.has_segmentation_data = False  # 是否有分割结果数据
        
        # 状态日志缓冲区（用于多行显示）
        self.status_log_buffer = []
        self.max_status_lines = 5  # 最多显示5行
        
        # 初始化火球计算器和序列管理器
        self.fireball_calculator = FireballCalculator()
        self.sequence_manager = SequenceManager()
        
        # 初始化控制器
        self.prompt_controller = PromptController(self)
        
        # 初始化UI构建器并创建界面
        self.ui_builder = ExtractTabUI()
        self.ui_builder.create_main_layout(self)
        self.ui_components = self.ui_builder.get_ui_components()
        
        # 设置控制器的 UI 组件引用
        self.prompt_controller.setup_ui_components(self.ui_components)
        
        # 获取UI组件引用（为了向后兼容）
        self._setup_ui_component_references()
        
        self.setup_connections()
        self.init_charts()
        # 连接异步信号
        self.log_received.connect(self._on_segmentation_log)
        self.seg_finished.connect(self._on_segmentation_finished)

    def _has_analysis_results(self) -> bool:
        """是否具备可保存的分析结果（直径曲线 + 拟合参数）。"""
        try:
            has_curve = bool(self.last_diameter_series)
            has_fit = isinstance(self.last_drag_fit_result, dict) and self.last_drag_fit_result.get('K') is not None
            return has_curve and has_fit
        except Exception:
            return False

    def _update_save_button_state(self):
        """统一刷新保存按钮可用状态。"""
        try:
            if hasattr(self, 'save_button'):
                self.save_button.setEnabled(self._has_analysis_results())
        except Exception:
            pass

    
    def _setup_ui_component_references(self):
        """设置UI组件引用（向后兼容）"""
        # 主要控件引用
        self.extract_preview = self.ui_components['extract_preview']
        self.extract_slider = self.ui_components['extract_slider']
        self.extract_time_label = self.ui_components['extract_time_label']
        self.extract_status = self.ui_components['extract_status']
        self.progress_label = self.ui_components['progress_label']
        
        # 图表控件引用（继承自 BaseChart）
        self.temp_chart = self.ui_components['temp_chart']
        self.diam_chart = self.ui_components['diam_chart']
        
        # 按钮控件引用
        self.sequence_btn = self.ui_components['sequence_btn']
        self.prompt_btn = self.ui_components['prompt_btn']
        self.extract_btn = self.ui_components['extract_btn']
        self.reextract_btn = self.ui_components['reextract_btn']
        self.cancel_extract_btn = self.ui_components['cancel_extract_btn']
        self.save_button = self.ui_components['save_button']
        
        # 单选按钮和组引用
        self.point_type_group = self.ui_components['point_type_group']
        self.positive_radio = self.ui_components['positive_radio']
        self.negative_radio = self.ui_components['negative_radio']
        self.ignition_radio = self.ui_components['ignition_radio']
        
        # 信息显示与导航控件引用
        self.prompt_info_text = self.ui_components['prompt_info_text']
        # 图片索引标签已被 CheckBar 替换
        self.check_bar = self.ui_components.get('check_bar', None)
        self.jump_input = self.ui_components['jump_input']
        self.jump_btn = self.ui_components['jump_btn']
    
    def setup_connections(self):
        """设置信号连接"""
        # 时间轴控件
        self.extract_slider.valueChanged.connect(self.on_time_changed)
        
        # 侧边栏按钮（PromptController 相关的信号已在控制器内部连接）
        self.sequence_btn.clicked.connect(self.select_sequence_folder)
        self.extract_btn.clicked.connect(self.start_feature_extraction)
        self.reextract_btn.clicked.connect(self.start_reextraction)
        self.cancel_extract_btn.clicked.connect(self._cancel_current_image_points)
        
        # 图像导航
        self.jump_btn.clicked.connect(self.jump_to_image)
        self.jump_input.returnPressed.connect(self.jump_to_image)
        
        # 保存按钮
        self.save_button.clicked.connect(self.save_extraction_sequence)
    
    def get_sidebar_widget(self):
        """获取特征提取模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            # 使用UI构建器创建侧边栏
            self._sidebar_widget = self.ui_builder.create_sidebar_widget()
        
        return self._sidebar_widget
    
    def _reset_state_before_import(self):
        """在导入新序列前清空之前的内存状态与UI显示"""
        try:
            print("🧹 重置状态：清空旧的序列、特征点与分割结果…")
            # 1) 清空内存数据
            self.image_paths = []
            self.current_image_index = 0
            self.sequence_data = None
            self.segmentation_results = []
            self.has_segmentation_data = False
            # 重置控制器状态
            self.prompt_controller.reset_state()
            # 避免误用上一份文件路径
            if hasattr(self, '_current_sequence_file_path'):
                delattr(self, '_current_sequence_file_path')
            
            # 2) 重置图像显示
            try:
                if hasattr(self.extract_preview, 'clear_segmentation_result'):
                    self.extract_preview.clear_segmentation_result()
                if hasattr(self.extract_preview, 'clear_points'):
                    self.extract_preview.clear_points()
                if hasattr(self.extract_preview, 'set_show_segmentation'):
                    self.extract_preview.set_show_segmentation(False)
            except Exception as e:
                print(f"⚠️ 重置图像显示时出错: {e}")
            
            # 3) 重置时间轴与索引
            try:
                self.extract_slider.setRange(0, 0)
                self.extract_slider.setValue(0)
                self.extract_time_label.setText("t = 0 ms")
                # 保存按钮初始禁用（等待生成直径/拟合结果）
                self._update_save_button_state()
            except Exception as e:
                print(f"⚠️ 重置时间轴时出错: {e}")
            
            # 4) 重置图表
            try:
                self.ui_builder.init_temperature_chart()
                self.ui_builder.init_diameter_chart()
            except Exception as e:
                print(f"⚠️ 重置图表时出错: {e}")
            
            # 5) 重置状态文本与按钮
            try:
                self.extract_status.setText("待开始")
                self.extract_btn.setVisible(True)
                if hasattr(self, 'reextract_btn'):
                    self.reextract_btn.setVisible(False)
            except Exception as e:
                print(f"⚠️ 重置状态控件时出错: {e}")
            
            print("✅ 状态重置完成")
        except Exception as e:
            print(f"❌ 重置状态失败: {e}")

    def _apply_sequence_data(self, sequence_data: dict, sequence_file_path: Optional[str] = None) -> bool:
        """将已加载的序列数据应用到界面与状态中，返回是否存在分割结果。"""
        try:
            # 保存序列数据和文件路径
            self.sequence_data = sequence_data
            if sequence_file_path:
                self._current_sequence_file_path = sequence_file_path
            
            # 提取图像路径
            image_paths = self.sequence_manager.get_image_paths_from_sequence(sequence_data)
            if not image_paths:
                QMessageBox.warning(self, "警告", "序列文件中没有图像路径！")
                self.extract_status.setText("无图像数据")
                return False
            
            # 提取参数
            parameters = self.sequence_manager.get_parameters_from_sequence(sequence_data)
            self.explosion_duration = int(parameters.get('explosion_duration', 140))
            self.pixel_length = float(parameters.get('pixel_length', 1.0))
            self.parameters = parameters or {}
            
            # 设置图像路径和索引
            self.image_paths = image_paths
            self.current_image_index = 0
            
            # 计算分组（平均分组，至少2组；每组最多250张）
            group_count = 1
            try:
                total = len(self.image_paths)
                if total > 0:
                    max_groups_by_size = (total + 249) // 250
                    # 至少分2组；若不足，按2组；否则取不超过max_groups_by_size且不超过total的合理值
                    group_count = max(2, max_groups_by_size)
                    group_count = min(group_count, total)
                else:
                    group_count = 1
            except Exception:
                group_count = 1
            
            # 设置控制器的分组数量
            self.prompt_controller.set_group_count(group_count)

            # 设置时间轴范围
            self.extract_slider.setRange(0, len(self.image_paths) - 1)
            self.extract_slider.setValue(0)

            # 设置图像控件为最大尺寸并显示第一张
            self.extract_preview.resize(self.extract_preview.maximumSize())
            self.display_image_at_index(0)

            # 加载温度数据
            time_data, temp_data = self.sequence_manager.get_temperature_data_from_sequence(sequence_data)
            if time_data and temp_data:
                self.update_temperature_chart(time_data, temp_data)

            # 分割结果优先
            has_segmentation_results = self.sequence_manager.has_segmentation_results(sequence_data)
            if has_segmentation_results:
                segmentation_results = self.sequence_manager.get_segmentation_results_from_sequence(sequence_data)
                self.segmentation_results = segmentation_results
                self.has_segmentation_data = True
                # 清除prompt数据避免冲突
                self.prompt_controller.reset_state()
                # 信息面板与直径图
                successful_count = sum(1 for r in segmentation_results if r.get('success', False))
                self.update_segmentation_info_display(successful_count, len(segmentation_results))
                self.update_diameter_chart_from_segmentation_results(segmentation_results)
                # 加载到现有分割结果后，也应允许保存分析结果（若具备）
                self._update_save_button_state()
                # 切换按钮
                self.extract_btn.setVisible(False)
                self.reextract_btn.setVisible(True)
            else:
                # 加载prompt数据与起爆点
                self.has_segmentation_data = False
                self.extract_btn.setVisible(True)
                self.reextract_btn.setVisible(False)
                prompt_data = self.sequence_manager.get_prompt_data_from_sequence(sequence_data)
                if prompt_data:
                    self.prompt_controller.load_prompt_data(prompt_data)
                ignition_point = self.sequence_manager.get_ignition_point_from_sequence(sequence_data)
                if ignition_point:
                    self.prompt_controller.set_ignition_point(ignition_point[0], ignition_point[1])
                self.prompt_controller.update_prompt_info_display()
                self.prompt_controller.load_points_for_current_image(0)

            # 序列摘要与状态
            summary = self.sequence_manager.get_sequence_summary(sequence_data)
            status_msg = f"已加载序列: {summary['image_count']} 个文件，时长: {self.explosion_duration}ms"
            if summary['has_temperature_data']:
                status_msg += f"，温度数据: {summary['temperature_points']} 点"
            if summary['has_prompt_data']:
                status_msg += f"，参考点数据: {summary['total_prompt_points']} 点"
            if summary['has_ignition_point']:
                status_msg += f"，起爆点: {summary['ignition_point']}"
            self.extract_status.setText(status_msg)
            return has_segmentation_results
            
        except Exception as e:
            print(f"❌ 应用序列数据失败: {e}")
            QMessageBox.critical(self, "错误", f"应用序列数据失败:\n{str(e)}")
            return False
    
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

            # 高亮 CheckBar 当前分组
            try:
                if getattr(self, 'check_bar', None) is not None:
                    self.check_bar.set_focus(value)
            except Exception:
                pass
        else:
            self.extract_time_label.setText(f"t = {value} ms")
    
    def display_image_at_index(self, index):
        """显示指定索引的图像"""
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return
        
        try:
            image_path = self.image_paths[index]
            # 使用交互式图像控件的 set_image 方法
            success = self.extract_preview.set_image(image_path)
            if success:
                self.current_image_index = index
                # 同步到 PromptController
                self.prompt_controller.set_current_image_index(index)
                
                # 优先显示分割结果，如果有的话
                if self.has_segmentation_data and index < len(self.segmentation_results):
                    # 有分割结果：显示分割结果，不显示特征点
                    segmentation_data = self.segmentation_results[index]
                    self.extract_preview.set_segmentation_result(segmentation_data)
                    self.extract_preview.set_show_segmentation(True)
                    self.extract_preview.clear_points()  # 清除特征点
                    print(f"显示图像: {image_path} (索引: {index}) - 分割结果模式")
                else:
                    # 没有分割结果：正常加载特征点
                    self.extract_preview.set_show_segmentation(False)
                    self.prompt_controller.load_points_for_current_image(index)
                    print(f"显示图像: {image_path} (索引: {index}) - 特征点模式")
                
                # 更新图片索引显示
                self.update_image_index_display()
            else:
                print(f"图像加载失败: {image_path}")
                
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
                # 导入前清空之前的状态
                self._reset_state_before_import()
                
                # 使用序列管理器加载文件
                success, sequence_data, message = self.sequence_manager.load_sequence_file(file_path)
                
                if not success:
                    QMessageBox.critical(self, "错误", f"加载序列文件失败:\n{message}")
                    self.extract_status.setText("文件加载失败")
                    return
                
                # 统一应用逻辑
                self._apply_sequence_data(sequence_data, file_path)
                # 序列加载完成
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"处理序列文件失败:\n{str(e)}")
                self.extract_status.setText("处理失败")
                print(f"处理序列文件失败: {e}")
    
    def init_charts(self):
        """初始化图表"""
        # 使用UI构建器初始化图表
        self.ui_builder.init_temperature_chart()
        self.ui_builder.init_diameter_chart()
    
    def update_temperature_chart(self, time_data, temp_data):
        """更新温度图表"""
        try:
            print(f"开始更新温度图表: {len(time_data)} 个数据点")
            self.temp_chart.update_data(time_data, temp_data)
            print("✅ 温度图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新温度图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_diameter_chart(self, time_data, diameter_data):
        """更新直径图表（提取完成后调用）"""
        try:
            print(f"📊 update_diameter_chart 被调用")
            self.diam_chart.update_data(time_data, diameter_data)
            print("✅ 直径图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新直径图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def start_feature_extraction(self):
        """开始特征提取（调用分割脚本）"""
        try:
            print("🔥 开始特征提取...")
            self.extract_status.setText("正在检查序列文件...")
            self.extract_btn.setEnabled(False)
            
            # 检查是否有序列数据
            if not self.sequence_data:
                QMessageBox.warning(self, "警告", "请先加载火球爆炸序列文件！")
                self.extract_status.setText("请先加载序列文件")
                self.extract_btn.setEnabled(True)
                return
            
            # 检查序列文件路径
            if not hasattr(self, '_current_sequence_file_path') or not self._current_sequence_file_path:
                QMessageBox.warning(self, "警告", "无法找到序列文件路径！")
                self.extract_status.setText("序列文件路径丢失")
                self.extract_btn.setEnabled(True)
                return
            
            # 检查分割状态
            segmentation_status = self.check_segmentation_status()
            
            if segmentation_status == 'no_prompt_data':
                # 情况1：没有prompt数据
                QMessageBox.warning(self, "警告", 
                    "序列文件中没有特征点数据！\n\n请先：\n1. 点击'开始选择参考点'\n2. 在图像上选择正负点\n3. 完成特征点选择后再进行提取")
                self.extract_status.setText("请先选择特征点")
                self.extract_btn.setEnabled(True)
                return
                
            elif segmentation_status == 'already_segmented':
                # 情况2：已有分割结果
                reply = QMessageBox.question(self, "提示", 
                    "检测到序列文件中已包含分割结果！\n\n特征提取已经完成。\n\n如需重新提取，请：\n1. 点击'重新提取'\n2. 重新选择特征点\n3. 再次执行提取",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # 用户选择重新提取
                    self.prepare_for_reextraction()
                else:
                    # 用户取消，恢复按钮状态
                    self.extract_btn.setEnabled(True)
                    self.extract_status.setText("已有分割结果")
                    return
            
            # 情况3：有prompt数据但没有分割结果，执行分割
            print("开始执行分割脚本...")
            self.extract_status.setText("正在执行分割脚本...")
            
            # 调用异步分割脚本（完成后的处理在 _on_segmentation_finished 中）
            self.run_segmentation_script(self._current_sequence_file_path)
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            self.extract_status.setText("特征提取失败")
            self.extract_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"特征提取失败:\n{str(e)}")
    
    # 已删除未使用的 calculate_fireball_diameter_curve 函数
    
    def check_segmentation_status(self):
        """
        检查序列文件的分割状态
        
        Returns:
            str: 'no_prompt_data' | 'already_segmented' | 'ready_for_segmentation'
        """
        try:
            # 检查是否有分割结果
            # 注意：使用当前内存中的 self.sequence_data 判断
            # 在重新提取后已重新加载 self.sequence_data，若仍有结果则认为已分割
            if self.sequence_manager.has_segmentation_results(self.sequence_data):
                return 'already_segmented'
            
            # 检查是否有prompt数据
            prompt_data = self.prompt_controller.get_prompt_data()
            if not prompt_data or (isinstance(prompt_data, dict) and len(prompt_data.keys()) == 0):
                return 'no_prompt_data'
            
            return 'ready_for_segmentation'
            
        except Exception as e:
            print(f"❌ 检查分割状态失败: {e}")
            return 'no_prompt_data'
    
    def run_segmentation_script(self, sequence_file_path: str) -> bool:
        """异步运行分割脚本：后台线程读日志，通过信号更新UI，不阻塞主线程。"""
        # 1) 禁用交互控件
        try:
            for w in [
                self.sequence_btn, self.prompt_btn, self.extract_btn, self.reextract_btn,
                self.cancel_extract_btn, self.save_button
            ]:
                if hasattr(w, 'setEnabled'):
                    w.setEnabled(False)
        except Exception:
            pass

        self.extract_status.setText("正在执行分割脚本…")
        # 清空日志缓冲区
        self.status_log_buffer = ["正在执行分割脚本…"]

        def worker():
            def on_line(line: str):
                self.log_received.emit(line)
            ok = run_segmentation_script(sequence_file_path, on_output_line=on_line)
            self.seg_finished.emit(ok)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _on_segmentation_log(self, line: str):
        text = line.rstrip('\n')
        if text:
            # 添加到日志缓冲区
            self.status_log_buffer.append(text)
            
            # 保持缓冲区大小不超过最大行数
            if len(self.status_log_buffer) > self.max_status_lines:
                self.status_log_buffer.pop(0)
            
            # 更新状态显示
            status_text = '\n'.join(self.status_log_buffer)
            self.extract_status.setText(status_text)

    def _on_segmentation_finished(self, ok: bool):
        # 恢复控件
        try:
            for w in [
                self.sequence_btn, self.prompt_btn, self.extract_btn, self.reextract_btn,
                self.cancel_extract_btn, self.save_button
            ]:
                if hasattr(w, 'setEnabled'):
                    w.setEnabled(True)
        except Exception:
            pass

        if ok:
            self.extract_status.setText("分割完成，正在加载分割结果…")
            self.reload_sequence_with_segmentation_results()
            self.extract_status.setText("特征提取完成")
            # 依据分析结果是否就绪来控制保存按钮
            self._update_save_button_state()
        else:
            self.extract_status.setText("分割脚本执行失败")
            QMessageBox.critical(self, "错误", "分割脚本执行失败！\n请检查控制台输出获取详细信息。")
    
    def prepare_for_reextraction(self):
        """准备重新提取：清除分割结果，重置为特征点选择模式"""
        try:
            # 清除分割结果数据
            self.segmentation_results = []
            self.has_segmentation_data = False
            
            # 清除JSON文件中的分割结果
            if hasattr(self, '_current_sequence_file_path') and self._current_sequence_file_path:
                success, message = self.sequence_manager.clear_segmentation_results_from_sequence(self._current_sequence_file_path)
                if success:
                    print(f"✅ {message}")
                    # 重要：清除文件后，重新加载到内存，避免旧的 self.sequence_data 仍然包含分割结果
                    reload_ok, new_data, _ = self.sequence_manager.load_sequence_file(self._current_sequence_file_path)
                    if reload_ok:
                        self.sequence_data = new_data
                else:
                    print(f"❌ {message}")
                
            # 重置显示模式
            self.extract_preview.set_show_segmentation(False)
            # 清空直径图表（用户反馈：未被清空）
            try:
                self.ui_builder.init_diameter_chart()
            except Exception as e:
                print(f"⚠️ 清空直径图表失败: {e}")
            
            # 重新加载特征点数据
            prompt_data = self.sequence_manager.get_prompt_data_from_sequence(self.sequence_data)
            if prompt_data:
                self.prompt_controller.load_prompt_data(prompt_data)
                self.prompt_controller.load_points_for_current_image(self.current_image_index)
                self.prompt_controller.update_prompt_info_display()
            
            # 更新状态
            self.extract_status.setText("已清除分割结果，请重新选择特征点")
            
            print("✅ 已准备重新提取")
                
        except Exception as e:
            print(f"❌ 准备重新提取失败: {e}")
    
    def reload_sequence_with_segmentation_results(self):
        """重新加载序列文件并显示分割结果"""
        try:
            # 重新加载分割后的序列文件（原文件名 + "_segmented"）
            from pathlib import Path
            original_path = Path(self._current_sequence_file_path)
            segmented_path = original_path.with_name(f"{original_path.stem}_segmented{original_path.suffix}")

            if not segmented_path.exists():
                print(f"⚠️ 分割后的序列文件不存在: {segmented_path}")
                return False

            success, sequence_data, message = self.sequence_manager.load_sequence_file(str(segmented_path))
            
            if not success:
                print(f"❌ 重新加载序列文件失败: {message}")
                return False
            
            # 统一应用逻辑
            return self._apply_sequence_data(sequence_data, str(segmented_path))
            
        except Exception as e:
            print(f"❌ 重新加载序列文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_diameter_chart_from_segmentation_results(self, segmentation_results):
        """根据分割结果更新直径图表（仅绘制成功点，不插值）"""
        try:
            series = build_time_diameter_series(
                segmentation_results, 
                float(self.explosion_duration),
                float(self.pixel_length)
            )
            if not series:
                # 清空图表
                self.ui_builder.init_diameter_chart()
                return
            time_data = [t for t, _ in series]
            diameter_data = [d for _, d in series]
            # 缓存曲线
            try:
                self.last_diameter_series = list(zip(time_data, diameter_data))
            except Exception:
                self.last_diameter_series = []

            # 调用拖曳曲线拟合，获取 K、B、C 与截断点
            K = B = C = None
            cutoff_ms = None
            try:
                fitter = DiameterDragFitter()
                fit_result = fitter.fit_drag_curve(
                    time_data,
                    diameter_data,
                    use_robust_fitting=True,
                    time_unit='ms',
                    enable_data_filtering=True,  # 强制启用数据过滤
                    drop_threshold=0.02,
                    window_size=10,
                )
                if fit_result.get('success', False):
                    K = fit_result.get('K')
                    B = fit_result.get('B')
                    C = fit_result.get('C')
                    df = fit_result.get('data_filtering', {}) or {}
                    cutoff_ms = df.get('cutoff_time')
                # 缓存拟合结果（成功或失败都存）
                try:
                    self.last_drag_fit_result = {
                        'success': bool(fit_result.get('success', False)),
                        'K': K, 'B': B, 'C': C,
                        'expression': 'D(t) = K * (1 - B * exp(-C * t^2))',
                        'data_filtering': fit_result.get('data_filtering', {}),
                    }
                except Exception:
                    self.last_drag_fit_result = None
            except Exception as e:
                print(f"⚠️ 直径拖曳拟合失败，退回仅绘制数据点: {e}")

            # 使用 DiameterChart 新接口绘制（带拟合与截断线，若有）
            try:
                self.diam_chart.update_data(time_data, diameter_data, K, B, C, cutoff_ms=cutoff_ms)
            except Exception as e:
                print(f"⚠️ 调用直径图更新接口失败，退回简单绘制: {e}")
                self.update_diameter_chart(time_data, diameter_data)
        except Exception as e:
            print(f"❌ 更新直径图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    
    
    
    def start_reextraction(self):
        """开始重新提取"""
        try:
            print("🔄 开始重新提取...")
            
            # 检查是否有序列数据
            if not self.sequence_data:
                QMessageBox.warning(self, "警告", "请先加载火球爆炸序列文件！")
                return
            
            # 确认操作
            reply = QMessageBox.question(self, "确认重新提取", 
                "重新提取将会：\n\n"
                "1. 清除当前的分割结果\n"
                "2. 恢复特征点选择模式\n"
                "3. 您需要重新选择特征点\n\n"
                "是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # 准备重新提取
                self.prepare_for_reextraction()
                
                # 隐藏重新提取按钮，显示正常提取按钮
                self.reextract_btn.setVisible(False)
                self.extract_btn.setVisible(True)
                
                print("✅ 已准备重新提取，请重新选择特征点")
            
        except Exception as e:
            print(f"❌ 重新提取失败: {e}")
            QMessageBox.critical(self, "错误", f"重新提取失败:\n{str(e)}")
    
    def update_segmentation_info_display(self, successful_count: int, total_count: int):
        """更新分割结果信息显示"""
        try:
            text = build_segmentation_info_text(successful_count, total_count)
            self.prompt_info_text.setPlainText(text)
        except Exception as e:
            print(f"❌ 更新分割结果信息显示失败: {e}")
            self.prompt_info_text.setPlainText(f"显示错误: {str(e)}")
    
    def _cancel_current_image_points(self):
        """取消当前图像上的所有点"""
        try:
            self.prompt_controller.cancel_current_image_points(self.current_image_index)
            # 清空图像控件中的点显示
            self.extract_preview.clear_points()
            self.extract_status.setText("已取消当前图像的参考点选择")
        except Exception as e:
            print(f"❌ 取消当前图像点失败: {e}")
    
    def update_ignition_point_display(self):
        """更新起爆点显示"""
        try:
            # 更新当前图像的起爆点显示
            self.prompt_controller.load_points_for_current_image(self.current_image_index)
                
        except Exception as e:
            print(f"❌ 更新起爆点显示失败: {e}")
    
    
    def save_extraction_sequence(self):
        """保存提取序列（按照example_data.json格式）"""
        try:
            # 改为保存直径与拟合参数结果
            if not self.last_diameter_series:
                QMessageBox.warning(self, "警告", "没有直径数据可保存！")
                return
            
            if not self.last_drag_fit_result or not (
                isinstance(self.last_drag_fit_result, dict) and self.last_drag_fit_result.get('K') is not None
            ):
                QMessageBox.warning(self, "警告", "没有拖曳曲线拟合参数可保存！")
                return
            
            # 选择保存文件（文件名反映保存内容）
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存直径与拟合结果",
                "fireball_diameter_fit.json",
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if file_path:
                # 导出分析结果
                success = self.export_analysis_results_to_json(file_path)
                
                if success:
                    QMessageBox.information(self, "成功", 
                                          f"分析结果已保存到:\n{file_path}\n\n"
                                          f"包含 {len(self.last_diameter_series)} 个直径数据点与拟合参数")
                    self.extract_status.setText("分析结果保存成功")
                else:
                    QMessageBox.critical(self, "错误", "保存失败，请检查文件路径和权限！")
                    self.extract_status.setText("分析结果保存失败")
                    
        except Exception as e:
            print(f"❌ 保存提取序列失败: {e}")
            QMessageBox.critical(self, "错误", f"保存提取序列失败:\n{str(e)}")
            self.extract_status.setText("保存失败")
    
    def export_analysis_results_to_json(self, file_path: str) -> bool:
        """导出直径曲线、爆炸参数、拖曳拟合结果到 JSON 文件。"""
        try:
            from pathlib import Path
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                # 各时刻直径
                "diameter_over_time": [
                    {"time_ms": float(t), "diameter_m": float(d)}
                    for (t, d) in (self.last_diameter_series or [])
                ],
                # 爆炸基本参数
                "parameters": {
                    "material_type": self.parameters.get('material_type'),
                    "equivalent": self.parameters.get('equivalent'),
                    "al_percent": self.parameters.get('al_percent'),
                    "explosion_duration": self.parameters.get('explosion_duration'),
                    "pixel_length": self.parameters.get('pixel_length'),
                },
                # 拖曳曲线拟合参数与表达式
                "drag_fit": (self.last_drag_fit_result or {
                    "success": False,
                    "K": None, "B": None, "C": None,
                    "expression": "D(t) = K * (1 - B * exp(-C * t^2))",
                })
            }

            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 导出分析结果失败: {e}")
            return False
    
    def update_image_index_display(self):
        """更新图片索引相关的可视化（当前仅更新 CheckBar）。"""
        try:
            # 同步更新 CheckBar（长度不会变，仅在切换图片时无需变更标注集合）
            if getattr(self, 'check_bar', None) is not None:
                self.check_bar.update(
                    length=len(self.image_paths),
                    group_count=self.prompt_controller.group_count,
                    annotated_indices=sorted(self.prompt_controller.get_annotated_indices())
                )
        except Exception as e:
            print(f"❌ 更新图片索引显示失败: {e}")
    
    def jump_to_image(self):
        """跳转到指定图片"""
        try:
            if not self.image_paths:
                QMessageBox.warning(self, "警告", "请先加载图像序列！")
                return
            
            # 获取输入的图片编号
            input_text = self.jump_input.text().strip()
            if not input_text:
                return
            
            try:
                # 转换为索引（用户输入从1开始，内部索引从0开始）
                image_number = int(input_text)
                image_index = image_number - 1
                
                # 检查索引范围
                if image_index < 0 or image_index >= len(self.image_paths):
                    QMessageBox.warning(self, "警告", 
                                      f"图片编号超出范围！\n有效范围: 1-{len(self.image_paths)}")
                    return
                
                # 跳转到指定图片
                self.display_image_at_index(image_index)
                
                # 同步更新时间轴
                self.extract_slider.setValue(image_index)

                # 高亮 CheckBar 当前分组
                try:
                    if getattr(self, 'check_bar', None) is not None:
                        self.check_bar.set_focus(image_index)
                except Exception:
                    pass
                
                # 清空输入框
                self.jump_input.clear()
                
                print(f"跳转到图片 {image_number} (索引: {image_index})")
                
            except ValueError:
                QMessageBox.warning(self, "警告", "请输入有效的数字！")
                
        except Exception as e:
            print(f"❌ 跳转到图片失败: {e}")
            QMessageBox.critical(self, "错误", f"跳转失败:\n{str(e)}")
    
    
