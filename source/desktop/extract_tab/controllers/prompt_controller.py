#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考点选择控制器
负责管理参考点选择相关的所有逻辑
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from PySide6.QtWidgets import QMessageBox
from ..utils.info_builder import build_prompt_info_text
from ..ui_widgets.extract_tab_ui import ExtractTabUI
from ..sequence_model import SequenceModel


class PromptController:
    """参考点选择控制器"""
    
    def __init__(self, parent_tab, ui_builder: ExtractTabUI, sequence_model: SequenceModel):
        """
        初始化控制器
        
        Args:
            parent_tab: 父标签页实例，用于访问序列管理器等共享资源
            ui_builder: UI 构建器，用于获取控件引用
        """
        self.parent = parent_tab
        self.sequence_model = sequence_model
        
        # UI 组件引用（在 setup_ui_components 中设置）
        self.prompt_btn = None
        self.cancel_prompt_btn = None
        self.extract_status = None
        self.extract_preview = None
        self.positive_radio = None
        self.negative_radio = None
        self.ignition_radio = None
        self.prompt_info_text = None
        self.check_bar = None
        
        # 参考点选择相关属性
        self.is_prompt_selection_mode = False
        self.current_image_index: int = 0

        # 初始化 UI 组件引用
        self._setup_ui_components(ui_builder.get_ui_components())

    def _set_status_line(self, text: str) -> None:
        """状态写入运行日志（QPlainTextEdit / QTextEdit / QLabel）。"""
        w = self.extract_status
        if w is None:
            return
        try:
            if hasattr(w, "appendPlainText"):
                w.appendPlainText(text if text.endswith("\n") else text + "\n")
            elif hasattr(w, "append"):
                w.append(text)
            elif hasattr(w, "setPlainText"):
                w.setPlainText(text)
            else:
                w.setText(text)
        except Exception:
            pass
    
    def _setup_ui_components(self, ui_components: Dict[str, Any]):
        """
        设置 UI 组件引用并连接信号
        
        Args:
            ui_components: UI 组件字典
        """
        self.prompt_btn = ui_components.get('prompt_btn')
        self.cancel_prompt_btn = ui_components.get('cancel_prompt_btn')
        self.extract_status = ui_components.get('extract_status')
        self.extract_preview = ui_components.get('extract_preview')
        self.positive_radio = ui_components.get('positive_radio')
        self.negative_radio = ui_components.get('negative_radio')
        self.ignition_radio = ui_components.get('ignition_radio')
        self.prompt_info_text = ui_components.get('prompt_info_text')
        self.check_bar = ui_components.get('check_bar')
        
        # 连接信号到控制器的方法
        self._connect_signals()
    
    def _connect_signals(self):
        """连接 UI 组件信号到控制器方法"""
        try:
            # 连接按钮点击事件
            if self.prompt_btn:
                self.prompt_btn.clicked.connect(self.toggle_prompt_selection)
            if self.cancel_prompt_btn:
                self.cancel_prompt_btn.clicked.connect(self._on_cancel_prompt_clicked)
            
            # 连接单选按钮状态变化
            if self.positive_radio:
                self.positive_radio.toggled.connect(self.on_radio_button_changed)
            if self.negative_radio:
                self.negative_radio.toggled.connect(self.on_radio_button_changed)
            if self.ignition_radio:
                self.ignition_radio.toggled.connect(self.on_radio_button_changed)
            
            # 连接图像点击事件（需要通过 lambda 传递当前图像索引）
            if self.extract_preview:
                self.extract_preview.point_clicked.connect(self._on_preview_point_clicked)
            
            print("✅ PromptController 信号连接完成")
        except Exception as e:
            print(f"⚠️ PromptController 信号连接失败: {e}")
    
    def _on_preview_point_clicked(self, x: int, y: int, point_type: str):
        """
        处理图像预览控件的点击事件（内部信号处理）
        
        Args:
            x, y: 点击坐标
            point_type: 点类型
        """
        try:
            self.on_image_point_clicked(x, y, point_type, self.current_image_index)
        except Exception as e:
            print(f"❌ 处理预览点击失败: {e}")
    
    def set_current_image_index(self, index: int):
        """
        设置当前图像索引
        
        Args:
            index: 图像索引
        """
        self.current_image_index = index
    
    def reset(self):
        """重置控制器数据与交互状态。"""
        self.sequence_model.clear_prompt_data()
        self.sequence_model.set_ignition_point(None)
        self.current_image_index = 0
        self.reset_interaction_state()
        self._refresh_checkbar()
        self.update_prompt_info_display()
        try:
            if self.extract_preview:
                self.extract_preview.clear_points()
        except Exception:
            pass

    def reset_interaction_state(self):
        """仅重置交互模式和相关 UI。"""
        self.is_prompt_selection_mode = False
        if self.prompt_btn:
            self.prompt_btn.setText("开始选择参考点")
        if self.extract_preview:
            self.extract_preview.set_interaction_mode('none')
            self.extract_preview.set_interactive_enabled(False)

    def load_prompt_data(self, prompt_data: Dict[int, Dict[str, List]]):
        """
        加载已存在的参考点数据
        
        Args:
            prompt_data: 参考点数据字典
        """
        self.sequence_model.set_prompt_data(prompt_data or {})
        self._refresh_checkbar()
        self.update_prompt_info_display()
        self.load_points_for_current_image(self.current_image_index)

    def sync_from_model(self):
        """当模型数据更新后，刷新 UI 展示。"""
        self.reset_interaction_state()
        self._refresh_checkbar()
        self.update_prompt_info_display()
        self.load_points_for_current_image(self.current_image_index)
    
    def get_prompt_data(self) -> Dict[int, Dict[str, List]]:
        """获取参考点数据"""
        return self.sequence_model.get_prompt_data()
    
    def get_ignition_point(self) -> Optional[Tuple[int, int]]:
        """获取起爆点"""
        return self.sequence_model.get_ignition_point()
    
    def get_annotated_indices(self) -> Set[int]:
        """获取已标注的图片索引集合"""
        return self.sequence_model.get_annotated_indices()
    
    def toggle_prompt_selection(self):
        """切换参考点选择模式"""
        try:
            if not self.is_prompt_selection_mode:
                # 进入选点前：将爆炸/炸药参数写回序列 JSON
                try:
                    if hasattr(self.parent, "flush_parameters_before_action"):
                        self.parent.flush_parameters_before_action()
                except Exception:
                    pass
                # 开始选择prompt点
                self.is_prompt_selection_mode = True
                self.prompt_btn.setText("选择参考点完成")
                self._set_status_line("正在选择参考点...")
                
                # 根据当前单选按钮状态设置交互模式
                current_type = self.get_current_point_type()
                self.extract_preview.set_interaction_mode(current_type)
                self.extract_preview.set_interactive_enabled(True)
                
                print(f"🎯 开始参考点选择模式: {current_type}")
            else:
                # 完成选点：校验与写盘成功后再 reset_interaction_state（避免保存失败时已误退出选点模式）
                if not self._finalize_prompt_selection():
                    return
                self.reset_interaction_state()
                
        except Exception as e:
            print(f"❌ 切换参考点选择模式失败: {e}")
            QMessageBox.critical(self.parent, "错误", f"切换参考点选择模式失败:\n{str(e)}")
    
    def _finalize_prompt_selection(self) -> bool:
        """
        在用户点击"选择参考点完成"后执行最终校验与后续动作：
        1) 校验每一分组都至少有一张已标注图片；
        2) 未达成则保持选择模式并提示；
        3) 通过则退出交互并自动保存。

        Returns:
            bool: 是否完成（通过校验并已保存）。未通过返回 False。
        """
        # 校验分组是否全部覆盖
        all_marked = False
        try:
            if self.check_bar is not None:
                all_marked = self.check_bar.is_all_groups_marked()
        except Exception:
            all_marked = False

        if not all_marked:
            self._set_status_line("参考点选择未完成：存在未标注的分组")
            QMessageBox.warning(self.parent, "未完成标注", "请确保每个分组至少标注一张图片！")
            # 保持在选择模式，便于继续补充标注
            self.is_prompt_selection_mode = True
            self.prompt_btn.setText("选择参考点完成")
            current_type = self.get_current_point_type()
            self.extract_preview.set_interaction_mode(current_type)
            self.extract_preview.set_interactive_enabled(True)
            return False

        # 通过校验：先写盘，成功后再由 toggle 末尾 reset_interaction_state 关闭交互
        self._set_status_line("正在将参考点写入序列文件…")
        try:
            if hasattr(self.parent, "flush_parameters_before_action"):
                self.parent.flush_parameters_before_action()
        except Exception:
            pass
        ok, msg = self._auto_save_prompt_data()
        if not ok:
            QMessageBox.warning(
                self.parent,
                "保存失败",
                f"参考点未能写入序列文件，下次打开将无法直接分割：\n{msg}",
            )
            self._set_status_line("参考点保存失败，请检查文件权限或磁盘空间后重试")
            # 保持选点模式，便于用户再次点击「完成」重试保存
            self.is_prompt_selection_mode = True
            if self.prompt_btn:
                self.prompt_btn.setText("选择参考点完成")
            current_type = self.get_current_point_type()
            if self.extract_preview:
                self.extract_preview.set_interaction_mode(current_type)
                self.extract_preview.set_interactive_enabled(True)
            return False
        self._set_status_line("参考点选择完成，已写入序列文件")
        try:
            if hasattr(self.parent, "append_run_log"):
                self.parent.append_run_log(f"✓ {msg}")
        except Exception:
            pass
        print("✅ 参考点选择完成并已写入序列文件")
        return True
    
    def get_current_point_type(self) -> str:
        """获取当前选择的点类型"""
        if self.positive_radio and self.positive_radio.isChecked():
            return 'positive'
        elif self.negative_radio and self.negative_radio.isChecked():
            return 'negative'
        elif self.ignition_radio and self.ignition_radio.isChecked():
            return 'ignition'
        else:
            return 'positive'  # 默认返回正点
    
    def add_prompt_point(self, image_index: int, x: int, y: int, is_positive: bool = True):
        """
        添加prompt点到数据结构
        
        Args:
            image_index: 图像索引
            x, y: 点坐标
            is_positive: 是否为正点
        """
        try:
            self.sequence_model.add_prompt_point(image_index, (x, y), is_positive)
            print(f"添加参考点: 图像{image_index}, 坐标({x}, {y}), 类型: {'正点' if is_positive else '负点'}")
            
            # 更新显示
            self.update_prompt_info_display()
            self._refresh_checkbar()
            
        except Exception as e:
            print(f"❌ 添加参考点失败: {e}")
    
    def set_ignition_point(self, x: int, y: int):
        """
        设置起爆点
        
        Args:
            x, y: 起爆点坐标
        """
        self.sequence_model.set_ignition_point((x, y))
        print(f"设置起爆点: 坐标({x}, {y})")
        self.update_prompt_info_display()
    
    def clear_prompt_data(self):
        """清空所有prompt数据"""
        self.sequence_model.clear_prompt_data()
        self.sequence_model.set_ignition_point(None)
        self.update_prompt_info_display()
        print("🗑️ 已清空所有参考点数据")
        self._refresh_checkbar()
    
    def _on_cancel_prompt_clicked(self):
        """处理取消按钮点击事件"""
        try:
            try:
                if hasattr(self.parent, "flush_parameters_before_action"):
                    self.parent.flush_parameters_before_action()
            except Exception:
                pass
            self.cancel_current_image_points(self.current_image_index)
            self._set_status_line("已取消当前图像的参考点选择")
        except Exception as e:
            print(f"❌ 处理取消按钮点击失败: {e}")
    
    def cancel_current_image_points(self, image_index: int):
        """
        取消指定图像上的所有点
        
        Args:
            image_index: 图像索引
        """
        try:
            self.sequence_model.remove_prompt_points(image_index)
            self.update_prompt_info_display()
            print(f"🗑️ 已取消图像{image_index}上的所有参考点")
            self._refresh_checkbar()
            # 刷新当前图像的显示（清除图像控件上的点标记）
            self.load_points_for_current_image(image_index)
        except Exception as e:
            print(f"❌ 取消当前图像点失败: {e}")
    
    def load_points_for_current_image(self, image_index: int):
        """
        为指定图像加载已有的点标记
        
        Args:
            image_index: 图像索引
        """
        try:
            positive_points = []
            negative_points = []
            
            prompt_entry = self.sequence_model.get_prompt_points(image_index)
                
            points = prompt_entry.get("points", [])
            labels = prompt_entry.get("labels", [])
            for point, label in zip(points, labels):
                if label == 1:
                    positive_points.append(tuple(point))
                else:
                    negative_points.append(tuple(point))
            
            # 设置到图像控件（包括起爆点）
            self.extract_preview.set_points(
                positive_points,
                negative_points,
                self.sequence_model.get_ignition_point(),
            )
            
            print(
                f"为图像{image_index}加载了{len(positive_points)}个正点, "
                f"{len(negative_points)}个负点, 起爆点: {self.sequence_model.get_ignition_point()}"
            )
                
        except Exception as e:
            print(f"❌ 加载当前图像点失败: {e}")
    
    def on_image_point_clicked(self, x: int, y: int, point_type: str, image_index: int):
        """
        处理图像点击事件
        
        Args:
            x, y: 点击的图像坐标
            point_type: 点类型 ('positive', 'negative', 'ignition')
            image_index: 当前图像索引
        """
        try:
            if not self.is_prompt_selection_mode:
                return
            
            if point_type == 'ignition':
                # 设置起爆点（全局唯一）
                self.set_ignition_point(x, y)
                # 更新所有图像的起爆点显示
                self.update_ignition_point_display()
            else:
                # 添加正负点到数据结构
                is_positive = (point_type == 'positive')
                self.add_prompt_point(image_index, x, y, is_positive)
            
            print(f"图像点击: 索引{image_index}, 坐标({x}, {y}), 类型: {point_type}")
            
        except Exception as e:
            print(f"❌ 处理图像点击失败: {e}")
    
    def on_radio_button_changed(self):
        """处理单选按钮状态变化"""
        try:
            if self.is_prompt_selection_mode:
                # 更新交互模式
                current_type = self.get_current_point_type()
                self.extract_preview.set_interaction_mode(current_type)
                print(f"切换点选择类型为: {current_type}")
                
        except Exception as e:
            print(f"❌ 处理单选按钮变化失败: {e}")
    
    def update_prompt_info_display(self):
        """更新参考点信息显示"""
        try:
            text = build_prompt_info_text(
                self.sequence_model.get_prompt_data(),
                self.sequence_model.get_ignition_point(),
            )
            self.prompt_info_text.setPlainText(text)
        except Exception as e:
            print(f"❌ 更新参考点信息显示失败: {e}")
            self.prompt_info_text.setPlainText(f"显示错误: {str(e)}")
    
    def update_ignition_point_display(self):
        """更新起爆点显示"""
        try:
            # 更新当前图像的起爆点显示
            self.load_points_for_current_image(self.current_image_index)
            
        except Exception as e:
            print(f"❌ 更新起爆点显示失败: {e}")
    
    def _refresh_checkbar(self):
        """根据当前分组与已标注索引，刷新 CheckBar 显示"""
        try:
            if self.check_bar is not None:
                self.check_bar.update(
                    length=len(self.sequence_model.image_paths),
                    group_count=self.sequence_model.group_count,
                    annotated_indices=sorted(self.sequence_model.get_annotated_indices())
                )
        except Exception:
            pass
    
    def _auto_save_prompt_data(self) -> Tuple[bool, str]:
        """
        将 prompt 与起爆点同步到当前序列 JSON（整文件写入）。
        返回 (是否成功, 说明信息)。
        """
        try:
            if not self.sequence_model.sequence_data:
                return False, "没有序列数据，无法保存参考点"
            success, message = self.sequence_model.save_prompt_artifacts()
            if success:
                print(f"✅ {message}")
                self._set_status_line("参考点已写入序列文件，可重新打开该 JSON 直接做火球分割")
            else:
                print(f"❌ 保存参考点失败: {message}")
                self._set_status_line(f"参考点保存失败: {message}")
            return success, message
        except Exception as e:
            print(f"❌ 自动保存参考点异常: {e}")
            return False, str(e)
    
    def set_prompt_controls_enabled(self, enabled: bool):
        """
        启用或禁用所有 prompt 相关的控件
        
        Args:
            enabled: True 表示启用，False 表示禁用
        """
        try:
            # 按钮控件
            if self.prompt_btn:
                self.prompt_btn.setEnabled(enabled)
            if self.cancel_prompt_btn:
                self.cancel_prompt_btn.setEnabled(enabled)
            
            # 单选按钮控件
            if self.positive_radio:
                self.positive_radio.setEnabled(enabled)
            if self.negative_radio:
                self.negative_radio.setEnabled(enabled)
            if self.ignition_radio:
                self.ignition_radio.setEnabled(enabled)
            
            # 图像预览控件的交互（点击事件）
            if self.extract_preview:
                # 只有在启用时才允许交互，禁用时关闭交互
                # 注意：这里不改变交互模式，只控制是否允许交互
                if enabled:
                    # 如果当前在选择模式，恢复交互
                    if self.is_prompt_selection_mode:
                        current_type = self.get_current_point_type()
                        self.extract_preview.set_interaction_mode(current_type)
                        self.extract_preview.set_interactive_enabled(True)
                    else:
                        # 不在选择模式，保持禁用状态
                        self.extract_preview.set_interactive_enabled(False)
                else:
                    # 禁用交互
                    self.extract_preview.set_interactive_enabled(False)
                    
        except Exception as e:
            print(f"❌ 设置 prompt 控件启用状态失败: {e}")
