#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考点选择控制器
负责管理参考点选择相关的所有逻辑
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from PySide6.QtWidgets import QMessageBox
from info_builder import build_prompt_info_text
from extract_tab_ui import ExtractTabUI


class PromptController:
    """参考点选择控制器"""
    
    def __init__(self, parent_tab, ui_builder: ExtractTabUI):
        """
        初始化控制器
        
        Args:
            parent_tab: 父标签页实例，用于访问序列管理器等共享资源
            ui_builder: UI 构建器，用于获取控件引用
        """
        self.parent = parent_tab
        
        # UI 组件引用（在 setup_ui_components 中设置）
        self.prompt_btn = None
        self.extract_status = None
        self.extract_preview = None
        self.positive_radio = None
        self.negative_radio = None
        self.ignition_radio = None
        self.prompt_info_text = None
        self.check_bar = None
        
        # 参考点选择相关属性
        self.is_prompt_selection_mode = False
        self.prompt_data: Dict[int, Dict[str, List]] = {}
        self.current_prompt_points: List = []
        self.ignition_point: Optional[Tuple[int, int]] = None
        
        # 当前图像索引（由控制器自己维护）
        self.current_image_index: int = 0
        
        # CheckBar 分组与标注跟踪
        self.group_count = 1
        self.annotated_indices: Set[int] = set()
        
        # 初始化 UI 组件引用
        self._setup_ui_components(ui_builder.get_ui_components())
    
    def _setup_ui_components(self, ui_components: Dict[str, Any]):
        """
        设置 UI 组件引用并连接信号
        
        Args:
            ui_components: UI 组件字典
        """
        self.prompt_btn = ui_components.get('prompt_btn')
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
    
    def set_group_count(self, group_count: int):
        """设置分组数量"""
        self.group_count = group_count
    
    def reset_state(self):
        """重置控制器状态"""
        self.is_prompt_selection_mode = False
        self.prompt_data = {}
        self.current_prompt_points = []
        self.ignition_point = None
        self.current_image_index = 0
        self.annotated_indices = set()
        self.group_count = 1
    
    def load_prompt_data(self, prompt_data: Dict[int, Dict[str, List]]):
        """
        加载已存在的参考点数据
        
        Args:
            prompt_data: 参考点数据字典
        """
        self.prompt_data = prompt_data or {}
        
        # 从已存在的 prompt_data 还原已标注索引集合
        try:
            self.annotated_indices = {
                idx for idx, d in self.prompt_data.items()
                if isinstance(d, dict) and len(d.get("points", [])) > 0
            }
        except Exception:
            self.annotated_indices = set()
        
        self._refresh_checkbar()
        self.update_prompt_info_display()
    
    def get_prompt_data(self) -> Dict[int, Dict[str, List]]:
        """获取参考点数据"""
        return self.prompt_data
    
    def get_ignition_point(self) -> Optional[Tuple[int, int]]:
        """获取起爆点"""
        return self.ignition_point
    
    def get_annotated_indices(self) -> Set[int]:
        """获取已标注的图片索引集合"""
        return self.annotated_indices.copy()
    
    def toggle_prompt_selection(self):
        """切换参考点选择模式"""
        try:
            if not self.is_prompt_selection_mode:
                # 开始选择prompt点
                self.is_prompt_selection_mode = True
                self.prompt_btn.setText("选择参考点完成")
                self.extract_status.setText("正在选择参考点...")
                
                # 根据当前单选按钮状态设置交互模式
                current_type = self.get_current_point_type()
                self.extract_preview.set_interaction_mode(current_type)
                self.extract_preview.set_interactive_enabled(True)
                
                print(f"🎯 开始参考点选择模式: {current_type}")
            else:
                # 完成选择参考点
                self.is_prompt_selection_mode = False
                self.prompt_btn.setText("开始选择参考点")
                # 交由独立函数处理校验与后续动作
                if not self._finalize_prompt_selection():
                    return
                
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
            self.extract_status.setText("参考点选择未完成：存在未标注的分组")
            QMessageBox.warning(self.parent, "未完成标注", "请确保每个分组至少标注一张图片！")
            # 保持在选择模式，便于继续补充标注
            self.is_prompt_selection_mode = True
            self.prompt_btn.setText("选择参考点完成")
            current_type = self.get_current_point_type()
            self.extract_preview.set_interaction_mode(current_type)
            self.extract_preview.set_interactive_enabled(True)
            return False

        # 通过校验：退出交互并保存
        self.extract_status.setText("参考点选择完成")
        self.extract_preview.set_interaction_mode('none')
        self.extract_preview.set_interactive_enabled(False)
        self._auto_save_prompt_data()
        print("✅ 参考点选择完成")
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
            # 确保数据结构存在
            if image_index not in self.prompt_data:
                self.prompt_data[image_index] = {
                    "points": [],
                    "labels": []
                }
            
            # 添加点坐标和标签
            self.prompt_data[image_index]["points"].append([x, y])
            self.prompt_data[image_index]["labels"].append(1 if is_positive else 0)
            
            print(f"添加参考点: 图像{image_index}, 坐标({x}, {y}), 类型: {'正点' if is_positive else '负点'}")
            
            # 更新显示
            self.update_prompt_info_display()

            # 标记该图片索引为已标注，并刷新 CheckBar
            try:
                self.annotated_indices.add(image_index)
            except Exception:
                pass
            self._refresh_checkbar()
            
        except Exception as e:
            print(f"❌ 添加参考点失败: {e}")
    
    def set_ignition_point(self, x: int, y: int):
        """
        设置起爆点
        
        Args:
            x, y: 起爆点坐标
        """
        self.ignition_point = (x, y)
        print(f"设置起爆点: 坐标({x}, {y})")
        self.update_prompt_info_display()
    
    def clear_prompt_data(self):
        """清空所有prompt数据"""
        self.prompt_data = {}
        self.ignition_point = None
        self.update_prompt_info_display()
        print("🗑️ 已清空所有参考点数据")
        # 清空标注并刷新 CheckBar
        try:
            self.annotated_indices.clear()
        except Exception:
            pass
        self._refresh_checkbar()
    
    def cancel_current_image_points(self, image_index: int):
        """
        取消指定图像上的所有点
        
        Args:
            image_index: 图像索引
        """
        try:
            if image_index in self.prompt_data:
                # 移除数据结构中的点
                del self.prompt_data[image_index]
                
                # 更新信息显示
                self.update_prompt_info_display()
                
                print(f"🗑️ 已取消图像{image_index}上的所有参考点")

                # 若该图片无点后，从已标注集合移除，并刷新 CheckBar
                try:
                    if image_index in self.annotated_indices:
                        self.annotated_indices.remove(image_index)
                except Exception:
                    pass
                self._refresh_checkbar()
                
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
            
            if image_index in self.prompt_data:
                points = self.prompt_data[image_index]["points"]
                labels = self.prompt_data[image_index]["labels"]
                
                # 分离正负点
                for point, label in zip(points, labels):
                    if label == 1:
                        positive_points.append(tuple(point))
                    else:
                        negative_points.append(tuple(point))
            
            # 设置到图像控件（包括起爆点）
            self.extract_preview.set_points(positive_points, negative_points, self.ignition_point)
            
            print(f"为图像{image_index}加载了{len(positive_points)}个正点, {len(negative_points)}个负点, 起爆点: {self.ignition_point}")
                
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
            text = build_prompt_info_text(self.prompt_data, self.ignition_point)
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
                    length=len(self.parent.image_paths),
                    group_count=self.group_count,
                    annotated_indices=sorted(self.annotated_indices)
                )
        except Exception:
            pass
    
    def _auto_save_prompt_data(self):
        """自动保存prompt数据和起爆点到当前序列文件"""
        try:
            # 检查是否有序列数据
            if not hasattr(self.parent, 'sequence_data') or not self.parent.sequence_data:
                print("没有序列数据，无法自动保存prompt数据")
                return
            
            # 检查是否有数据需要保存
            if not self.prompt_data and not self.ignition_point:
                print("没有prompt数据或起爆点，无需保存")
                return
            
            # 检查是否有原始文件路径（从序列加载时保存）
            if not hasattr(self.parent, '_current_sequence_file_path'):
                print("没有当前序列文件路径，无法自动保存")
                return
            
            # 使用序列管理器保存prompt数据和起爆点
            success, message = self.parent.sequence_manager.save_prompt_and_ignition_data_to_sequence(
                self.parent._current_sequence_file_path, self.prompt_data, self.ignition_point
            )
            
            if success:
                print(f"✅ 自动保存prompt数据和起爆点成功: {message}")
                self.extract_status.setText("参考点数据已自动保存")
                # 同步更新内存中的 sequence_data，避免后续再次读取文件
                try:
                    if isinstance(self.parent.sequence_data, dict):
                        # 确保 image_sequence 节点存在
                        if 'image_sequence' not in self.parent.sequence_data or not isinstance(self.parent.sequence_data['image_sequence'], dict):
                            self.parent.sequence_data['image_sequence'] = {}
                        image_seq = self.parent.sequence_data['image_sequence']
                        # 写入 prompt_data（键使用字符串以保持与文件一致）
                        image_seq['prompt_data'] = {str(k): v for k, v in self.prompt_data.items()}
                        # 写入起爆点
                        if self.ignition_point is not None:
                            image_seq['target_center'] = list(self.ignition_point)
                        else:
                            # 若起爆点被清除
                            if 'target_center' in image_seq:
                                del image_seq['target_center']
                except Exception as _:
                    pass
            else:
                print(f"❌ 自动保存参考点数据失败: {message}")
                self.extract_status.setText("参考点数据保存失败")
                
        except Exception as e:
            print(f"❌ 自动保存参考点数据异常: {e}")
