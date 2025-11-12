#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列显示控制器
负责管理图像序列的显示、时间轴同步、CheckBar 高亮等所有显示相关的逻辑
"""

from PySide6.QtWidgets import QSlider, QLabel, QPushButton, QLineEdit
from ..ui_widgets.extract_tab_ui import ExtractTabUI
from ..sequence_model import SequenceModel
from .prompt_controller import PromptController
from ..ui_widgets.interactive_image_widget import InteractiveImageWidget


class SequencyDisplayController:
    """序列显示控制器"""
    
    def __init__(
        self,
        ui_builder: ExtractTabUI,
        sequence_model: SequenceModel,
        prompt_controller: PromptController,
    ):
        """
        初始化控制器
        
        Args:
            ui_builder: UI 构建器，用于获取控件引用
            sequence_model: 序列数据模型
            prompt_controller: Prompt 控制器，用于加载 prompt 点
        """
        self.sequence_model = sequence_model
        self.prompt_controller = prompt_controller
        
        # 从 UI 构建器获取控件引用
        ui_components = ui_builder.get_ui_components()
        self.preview: InteractiveImageWidget = ui_components.get('extract_preview')
        self.slider: QSlider = ui_components.get('extract_slider')
        self.time_label: QLabel = ui_components.get('extract_time_label')
        self.check_bar = ui_components.get('check_bar')
        self.jump_input: QLineEdit = ui_components.get('jump_input')
        self.jump_btn: QPushButton = ui_components.get('jump_btn')
        
        # 当前显示的图像索引
        self._current_index: int = 0
        
        # 初始禁用跳转控件，待序列加载后启用
        self._set_jump_controls_enabled(False)

        # 连接信号
        self._connect_signals()
    
    def _connect_signals(self):
        """连接 UI 组件信号到控制器方法"""
        try:
            if self.slider:
                self.slider.valueChanged.connect(self._on_slider_changed)
            # 图像导航控件
            if self.jump_btn:
                self.jump_btn.clicked.connect(self.jump_to_image)
            if self.jump_input:
                self.jump_input.returnPressed.connect(self.jump_to_image)
            print("✅ SequencyDisplayController 信号连接完成")
        except Exception as e:
            print(f"⚠️ SequencyDisplayController 信号连接失败: {e}")
    
    def _on_slider_changed(self, value: int):
        """时间轴变化时自动调用 display_image（避免循环调用）"""
        # 如果当前索引已经是 value，则不需要更新
        if self._current_index == value:
            return
        self.display_image(value, update_slider=False)
    
    def reset(self):
        """清空显示状态（图像、时间轴、分割层等）"""
        try:
            # 清空图像显示
            if self.preview:
                if hasattr(self.preview, 'clear_segmentation_result'):
                    self.preview.clear_segmentation_result()
                if hasattr(self.preview, 'clear_points'):
                    self.preview.clear_points()
                if hasattr(self.preview, 'set_show_segmentation'):
                    self.preview.set_show_segmentation(False)
            
            # 重置时间轴与索引
            if self.slider:
                self.slider.setRange(0, 0)
                self.slider.setValue(0)
            if self.time_label:
                self.time_label.setText("t = 0 ms")
            
            self._current_index = 0
            self._set_jump_controls_enabled(False)
            
        except Exception as e:
            print(f"⚠️ 重置显示状态失败: {e}")
    
    def apply_sequence(self):
        """序列加载完成后初始化显示（设置 slider 范围、显示第一张图）"""
        try:
            image_paths = self.sequence_model.image_paths
            if not image_paths:
                self._set_jump_controls_enabled(False)
                return
            
            # 设置时间轴范围
            if self.slider:
                self.slider.setRange(0, len(image_paths) - 1)
                self.slider.setValue(0)
            
            # 设置图像控件为最大尺寸
            if self.preview:
                self.preview.resize(self.preview.maximumSize())
            
            self._set_jump_controls_enabled(True)

            # 显示第一张图像
            self.display_image(0)
            
        except Exception as e:
            print(f"⚠️ 应用序列显示失败: {e}")
    
    def display_image(self, index: int, update_slider: bool = True):
        """
        显示指定索引的图像（内部自动处理所有 UI 更新）
        
        Args:
            index: 图像索引
            update_slider: 是否更新 slider 值（默认 True，从外部调用时更新；从 slider 信号调用时为 False）
        """
        image_paths = self.sequence_model.image_paths
        if not image_paths or index < 0 or index >= len(image_paths):
            return
        
        try:
            image_path = image_paths[index]
            
            # 加载图像
            if not self.preview:
                return
            
            success = self.preview.set_image(image_path)
            if not success:
                print(f"图像加载失败: {image_path}")
                return
            
            # 更新当前索引
            self._current_index = index
            
            # 同步更新 slider（如果需要，且避免循环调用）
            if update_slider and self.slider:
                # 临时断开信号，避免循环调用
                self.slider.blockSignals(True)
                self.slider.setValue(index)
                self.slider.blockSignals(False)
            
            # 同步到 PromptController
            self.prompt_controller.set_current_image_index(index)
            
            # 根据模型状态决定显示模式
            self._load_image_with_mode(index)
            
            # 自动更新时间标签
            self._update_time_label(index)
            
            # 自动高亮 CheckBar 当前分组
            self._highlight_current_group(index)
            
        except Exception as e:
            print(f"❌ 显示图像失败: {e}")
    
    def sync_to_model(self):
        """当模型状态更新后，刷新显示"""
        try:
            # 重新显示当前图像以刷新状态
            self.display_image(self._current_index)
        except Exception as e:
            print(f"⚠️ 同步模型状态失败: {e}")
    
    def handle_segmentation_update(self):
        """分割完成或清除后调用，刷新显示模式"""
        try:
            # 重新显示当前图像以刷新显示模式
            self.display_image(self._current_index)
        except Exception as e:
            print(f"⚠️ 处理分割更新失败: {e}")
    
    def _load_image_with_mode(self, index: int):
        """根据模型状态决定显示 Prompt 还是 Segmentation"""
        try:
            segmentation_results = self.sequence_model.get_segmentation_results()
            
            if (self.sequence_model.has_segmentation_data()
                    and index < len(segmentation_results)):
                # 有分割结果：显示分割结果，不显示特征点
                segmentation_data = segmentation_results[index]
                self.preview.set_segmentation_result(segmentation_data)
                self.preview.set_show_segmentation(True)
                self.preview.clear_points()  # 清除特征点
                print(f"显示图像: 索引{index} - 分割结果模式")
            else:
                # 没有分割结果：正常加载特征点
                self.preview.set_show_segmentation(False)
                self.prompt_controller.load_points_for_current_image(index)
                print(f"显示图像: 索引{index} - 特征点模式")
                
        except Exception as e:
            print(f"⚠️ 加载图像模式失败: {e}")
    
    def _update_time_label(self, index: int):
        """自动计算并更新时间标签"""
        try:
            if not self.time_label:
                return
            
            image_paths = self.sequence_model.image_paths
            if not image_paths:
                self.time_label.setText(f"t = {index} ms")
                return
            
            total_frames = len(image_paths)
            if total_frames > 1:
                time_ms = (index / (total_frames - 1)) * self.sequence_model.explosion_duration_ms
            else:
                time_ms = 0
            
            self.time_label.setText(f"t = {time_ms:.1f} ms (帧 {index + 1}/{total_frames})")
            
        except Exception as e:
            print(f"⚠️ 更新时间标签失败: {e}")
    
    def _highlight_current_group(self, index: int):
        """自动高亮 CheckBar 当前分组"""
        try:
            if self.check_bar:
                self.check_bar.set_focus(index)
        except Exception as e:
            print(f"⚠️ 高亮当前分组失败: {e}")

    def _set_jump_controls_enabled(self, enabled: bool):
        """启用或禁用跳转相关控件"""
        try:
            if self.jump_btn:
                self.jump_btn.setEnabled(enabled)
            if self.jump_input:
                self.jump_input.setEnabled(enabled)
        except Exception as e:
            print(f"⚠️ 设置跳转控件状态失败: {e}")
    
    def get_current_index(self) -> int:
        """获取当前显示的图像索引"""
        return self._current_index
    
    def jump_to_image(self):
        """跳转到指定图片"""
        try:
            image_paths = self.sequence_model.image_paths
            if not image_paths:
                self._set_jump_controls_enabled(False)
                return
            
            # 获取输入的图片编号
            if not self.jump_input:
                return
            input_text = self.jump_input.text().strip()
            if not input_text:
                return
            
            try:
                # 转换为索引（用户输入从1开始，内部索引从0开始）
                image_number = int(input_text)
                image_index = image_number - 1
                
                # 检查索引范围
                if image_index < 0 or image_index >= len(image_paths):
                    print(f"⚠️ 图片编号超出范围！有效范围: 1-{len(image_paths)}")
                    return
                
                # 跳转到指定图片（内部会自动更新 slider 和 time label）
                self.display_image(image_index)
                
                # 清空输入框
                self.jump_input.clear()
                
                print(f"跳转到图片 {image_number} (索引: {image_index})")
                
            except ValueError:
                print("⚠️ 跳转输入无效：请输入数字")
                
        except Exception as e:
            print(f"❌ 跳转到图片失败: {e}")

