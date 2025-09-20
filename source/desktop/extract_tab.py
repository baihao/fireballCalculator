#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块标签页
"""

import numpy as np
import json
import os
import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox,
                               QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QTextEdit, QScrollArea)
from PySide6.QtCore import Qt
from framework import MatplotlibWidget, ImagePreviewWidget
from sequence_manager import SequenceManager
from extract_tab_ui import ExtractTabUI

# 添加路径以导入火球计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fireball_radius_calculator import FireballCalculator


class ExtractTab(QWidget):
    """特征提取模块标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化图像序列相关属性
        self.image_paths = []  # 图像路径列表（统一使用）
        self.current_image_index = 0  # 当前显示的图像索引
        self.sequence_data = None  # 序列数据
        self.explosion_duration = 140  # 爆炸时长（毫秒）
        
        # 初始化prompt选择相关属性
        self.is_prompt_selection_mode = False  # 是否处于参考点选择模式
        self.prompt_data = {}  # prompt数据：{image_index: {"points": [[x,y], ...], "labels": [1,0,1,...]}}
        self.current_prompt_points = []  # 当前图像的参考点临时存储
        self.ignition_point = None  # 起爆点坐标 (x, y) 或 None
        
        # 初始化火球计算器和序列管理器
        self.fireball_calculator = FireballCalculator()
        self.sequence_manager = SequenceManager()
        
        # 初始化UI构建器并创建界面
        self.ui_builder = ExtractTabUI()
        self.ui_builder.create_main_layout(self)
        self.ui_components = self.ui_builder.get_ui_components()
        
        # 获取UI组件引用（为了向后兼容）
        self._setup_ui_component_references()
        
        self.setup_connections()
        self.init_charts()
    
    def _setup_ui_component_references(self):
        """设置UI组件引用（向后兼容）"""
        # 主要控件引用
        self.extract_preview = self.ui_components['extract_preview']
        self.extract_slider = self.ui_components['extract_slider']
        self.extract_time_label = self.ui_components['extract_time_label']
        self.extract_status = self.ui_components['extract_status']
        self.progress_label = self.ui_components['progress_label']
        
        # 图表控件引用
        self.temp_chart = self.ui_components['temp_chart']
        self.diam_chart = self.ui_components['diam_chart']
        
        # 按钮控件引用
        self.sequence_btn = self.ui_components['sequence_btn']
        self.prompt_btn = self.ui_components['prompt_btn']
        self.extract_btn = self.ui_components['extract_btn']
        self.cancel_extract_btn = self.ui_components['cancel_extract_btn']
        self.save_button = self.ui_components['save_button']
        
        # 单选按钮和组引用
        self.point_type_group = self.ui_components['point_type_group']
        self.positive_radio = self.ui_components['positive_radio']
        self.negative_radio = self.ui_components['negative_radio']
        self.ignition_radio = self.ui_components['ignition_radio']
        
        # 信息显示控件引用
        self.prompt_info_text = self.ui_components['prompt_info_text']
        self.image_index_label = self.ui_components['image_index_label']
        self.jump_input = self.ui_components['jump_input']
        self.jump_btn = self.ui_components['jump_btn']
    
    def setup_connections(self):
        """设置信号连接"""
        # 时间轴和图像控件
        self.extract_slider.valueChanged.connect(self.on_time_changed)
        self.extract_preview.point_clicked.connect(self.on_image_point_clicked)
        
        # 侧边栏按钮
        self.sequence_btn.clicked.connect(self.select_sequence_folder)
        self.prompt_btn.clicked.connect(self.toggle_prompt_selection)
        self.extract_btn.clicked.connect(self.start_feature_extraction)
        self.cancel_extract_btn.clicked.connect(self.cancel_current_image_points)
        
        # 单选按钮状态变化
        self.positive_radio.toggled.connect(self.on_radio_button_changed)
        self.negative_radio.toggled.connect(self.on_radio_button_changed)
        self.ignition_radio.toggled.connect(self.on_radio_button_changed)
        
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
                # 加载该图像已有的点标记
                self.load_points_for_current_image()
                # 更新图片索引显示
                self.update_image_index_display()
                print(f"显示图像: {image_path} (索引: {index})")
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
                # 使用序列管理器加载文件
                success, sequence_data, message = self.sequence_manager.load_sequence_file(file_path)
                
                if not success:
                    QMessageBox.critical(self, "错误", f"加载序列文件失败:\n{message}")
                    self.extract_status.setText("文件加载失败")
                    return
                
                # 保存序列数据和文件路径
                self.sequence_data = sequence_data
                self._current_sequence_file_path = file_path
                
                # 提取图像路径
                image_paths = self.sequence_manager.get_image_paths_from_sequence(sequence_data)
                if not image_paths:
                    QMessageBox.warning(self, "警告", "序列文件中没有图像路径！")
                    self.extract_status.setText("无图像数据")
                    return
                
                # 提取参数
                parameters = self.sequence_manager.get_parameters_from_sequence(sequence_data)
                self.explosion_duration = int(parameters.get('explosion_duration', 140))
                
                # 设置图像路径和索引
                self.image_paths = image_paths
                self.current_image_index = 0
                
                # 设置时间轴范围
                self.extract_slider.setRange(0, len(self.image_paths) - 1)
                self.extract_slider.setValue(0)
                
                # 设置图像控件为最大尺寸
                self.extract_preview.resize(self.extract_preview.maximumSize())
                
                # 显示第一张图像
                self.display_image_at_index(0)
                
                # 加载温度数据
                time_data, temp_data = self.sequence_manager.get_temperature_data_from_sequence(sequence_data)
                if time_data and temp_data:
                    self.update_temperature_chart(time_data, temp_data)
                    print(f"加载温度数据: {len(temp_data)} 个数据点")
                
                # 加载prompt数据
                prompt_data = self.sequence_manager.get_prompt_data_from_sequence(sequence_data)
                if prompt_data:
                    self.prompt_data = prompt_data
                    print(f"加载prompt数据: {len(prompt_data)} 张图像的prompt点")
                
                # 加载起爆点
                ignition_point = self.sequence_manager.get_ignition_point_from_sequence(sequence_data)
                if ignition_point:
                    self.ignition_point = ignition_point
                    print(f"加载起爆点: {ignition_point}")
                
                # 更新信息面板显示（包含prompt数据和起爆点）
                self.update_prompt_info_display()
                
                # 如果当前显示的图像有prompt点或起爆点，加载它们
                self.load_points_for_current_image()
                
                # 获取摘要信息
                summary = self.sequence_manager.get_sequence_summary(sequence_data)
                
                # 更新状态
                status_msg = f"已加载序列: {summary['image_count']} 个文件，时长: {self.explosion_duration}ms"
                if summary['has_temperature_data']:
                    status_msg += f"，温度数据: {summary['temperature_points']} 点"
                if summary['has_prompt_data']:
                    status_msg += f"，参考点数据: {summary['total_prompt_points']} 点"
                if summary['has_ignition_point']:
                    status_msg += f"，起爆点: {summary['ignition_point']}"
                
                self.extract_status.setText(status_msg)
                print(f"成功加载火球序列: {summary}")
                    
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
            
            # 使用 MatplotlibWidget 的 plot_line 方法
            print(f"绘制温度曲线: 时间范围 {min(time_data)}-{max(time_data)} ms, 温度范围 {min(temp_data)}-{max(temp_data)} K")
            
            self.temp_chart.plot_line(
                time_data, temp_data,
                title="火球温度随时间变化",
                xlabel="时间 (ms)",
                ylabel="温度 (K)",
                color='#38bdf8'
            )
            
            print("✅ 温度图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新温度图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_diameter_chart(self, time_data, diameter_data):
        """更新直径图表（提取完成后调用）"""
        try:
            print(f"开始更新直径图表: {len(time_data)} 个数据点")
            
            # 使用 MatplotlibWidget 的 plot_line 方法
            self.diam_chart.plot_line(
                time_data, diameter_data,
                title="火球直径随时间变化",
                xlabel="时间 (ms)",
                ylabel="直径 (m)",
                color='#f59e0b'
            )
            
            print("✅ 直径图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新直径图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def start_feature_extraction(self):
        """开始特征提取"""
        try:
            print("🔥 开始特征提取...")
            self.extract_status.setText("正在提取特征...")
            self.extract_btn.setEnabled(False)
            
            # 检查是否有序列数据
            if not self.sequence_data:
                QMessageBox.warning(self, "警告", "请先加载火球爆炸序列文件！")
                self.extract_status.setText("请先加载序列文件")
                self.extract_btn.setEnabled(True)
                return
            
            # 获取材料类型（从序列数据中获取）
            material_type = self.sequence_data.get('parameters', {}).get('material_type', '40%Al/Rubber')
            
            # 映射材料类型到计算器中的名称
            material_mapping = {
                '40% Al / Rubber': '40%Al/Rubber',
                '30% Al / Rubber': '30%Al/Rubber', 
                '50% Al / Rubber': '50%Al/Rubber',
                '60% Al / Rubber': '60%Al/Rubber',
                'Polyurethane': 'Polyurethane'
            }
            
            material_name = material_mapping.get(material_type, '40%Al/Rubber')
            print(f"使用材料类型: {material_name}")
            
            # 计算火球直径随时间变化
            self.calculate_fireball_diameter_curve(material_name)
            
            # 更新状态
            self.extract_status.setText("特征提取完成")
            self.save_button.setEnabled(True)
            self.extract_btn.setEnabled(True)
            
            print("✅ 特征提取完成！")
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            self.extract_status.setText("特征提取失败")
            self.extract_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"特征提取失败:\n{str(e)}")
    
    def calculate_fireball_diameter_curve(self, material_name):
        """计算火球直径随时间变化的曲线"""
        try:
            print(f"计算 {material_name} 材料的火球直径曲线...")
            
            # 生成时间序列 (0-140ms)
            time_points = 100  # 时间点数量
            t_ms = np.linspace(0, self.explosion_duration, time_points)  # 毫秒
            t_s = t_ms / 1000.0  # 转换为秒
            
            # 计算直径
            diameter_data = []
            for t in t_s:
                diameter = self.fireball_calculator.calculate_diameter(t, material_name)
                diameter_data.append(diameter)
            
            print(f"✅ 火球直径计算完成: {len(diameter_data)} 个数据点")
            print(f"   时间范围: {min(t_ms)} - {max(t_ms)} ms")
            print(f"   直径范围: {min(diameter_data):.3f} - {max(diameter_data):.3f} m")
            
            # 更新直径图表
            self.update_diameter_chart(t_ms, diameter_data)
            
            # 保存计算结果
            self.extraction_results = {
                'time_ms': t_ms.tolist(),
                'diameter_m': diameter_data,
                'material': material_name,
                'explosion_duration': self.explosion_duration
            }
            
        except Exception as e:
            print(f"❌ 计算火球直径曲线失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def toggle_prompt_selection(self):
        """切换prompt点选择模式"""
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
                self.extract_status.setText("参考点选择完成")
                
                # 禁用交互
                self.extract_preview.set_interaction_mode('none')
                self.extract_preview.set_interactive_enabled(False)
                
                # 自动保存prompt数据和起爆点到序列文件
                self._auto_save_prompt_data()
                
                print("✅ 参考点选择完成")
                
        except Exception as e:
            print(f"❌ 切换参考点选择模式失败: {e}")
            QMessageBox.critical(self, "错误", f"切换参考点选择模式失败:\n{str(e)}")
    
    def _auto_save_prompt_data(self):
        """自动保存prompt数据和起爆点到当前序列文件"""
        try:
            # 检查是否有序列数据
            if not self.sequence_data:
                print("没有序列数据，无法自动保存prompt数据")
                return
            
            # 检查是否有数据需要保存
            if not self.prompt_data and not self.ignition_point:
                print("没有prompt数据或起爆点，无需保存")
                return
            
            # 检查是否有原始文件路径（从序列加载时保存）
            if not hasattr(self, '_current_sequence_file_path'):
                print("没有当前序列文件路径，无法自动保存")
                return
            
            # 使用序列管理器保存prompt数据和起爆点
            success, message = self.sequence_manager.save_prompt_and_ignition_data_to_sequence(
                self._current_sequence_file_path, self.prompt_data, self.ignition_point
            )
            
            if success:
                print(f"✅ 自动保存prompt数据和起爆点成功: {message}")
                self.extract_status.setText("参考点数据已自动保存")
            else:
                print(f"❌ 自动保存参考点数据失败: {message}")
                self.extract_status.setText("参考点数据保存失败")
                
        except Exception as e:
            print(f"❌ 自动保存参考点数据异常: {e}")
    
    def get_current_point_type(self):
        """获取当前选择的点类型"""
        if hasattr(self, 'positive_radio') and self.positive_radio.isChecked():
            return 'positive'
        elif hasattr(self, 'negative_radio') and self.negative_radio.isChecked():
            return 'negative'
        elif hasattr(self, 'ignition_radio') and self.ignition_radio.isChecked():
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
            
        except Exception as e:
            print(f"❌ 添加参考点失败: {e}")
    
    def clear_prompt_data(self):
        """清空所有prompt数据"""
        self.prompt_data = {}
        self.ignition_point = None
        self.update_prompt_info_display()
        print("🗑️ 已清空所有参考点数据")
    
    def update_prompt_info_display(self):
        """更新参考点信息显示"""
        try:
            if not self.prompt_data and not self.ignition_point:
                self.prompt_info_text.setPlainText("暂无选择的参考点\n\n提示：\n1. 先加载图像序列\n2. 点击'开始选择参考点'\n3. 在图像上点击选择正负点")
                return
            
            # 生成用户可读的信息
            info_lines = []
            
            # 显示起爆点信息（如果存在）
            if self.ignition_point:
                info_lines.append("🎯 起爆点信息：")
                info_lines.append(f"  - 坐标：({self.ignition_point[0]}, {self.ignition_point[1]})")
                info_lines.append("")  # 空行分隔
            
            # 显示各图像的参考点信息
            if self.prompt_data:
                for image_idx in sorted(self.prompt_data.keys()):
                    points = self.prompt_data[image_idx]["points"]
                    labels = self.prompt_data[image_idx]["labels"]
                
                    # 分离正点和负点
                    positive_points = []
                    negative_points = []
                    
                    for point, label in zip(points, labels):
                        if label == 1:
                            positive_points.append(f"({point[0]}, {point[1]})")
                        else:
                            negative_points.append(f"({point[0]}, {point[1]})")
                    
                    # 格式化显示
                    info_lines.append(f"第 {image_idx + 1} 张图片参考点：")
                    
                    if positive_points:
                        info_lines.append(f"  - 正点坐标：{', '.join(positive_points)}")
                    
                    if negative_points:
                        info_lines.append(f"  - 负点坐标：{', '.join(negative_points)}")
                    
                    info_lines.append("")  # 空行分隔
            
            # 添加统计信息
            total_images_with_prompts = len(self.prompt_data) if self.prompt_data else 0
            total_points = sum(len(data["points"]) for data in self.prompt_data.values()) if self.prompt_data else 0
            total_positive = sum(sum(1 for label in data["labels"] if label == 1) for data in self.prompt_data.values()) if self.prompt_data else 0
            total_negative = total_points - total_positive
            
            info_lines.append("=" * 30)
            info_lines.append("统计信息：")
            if total_images_with_prompts > 0:
                info_lines.append(f"  - 有参考点的图像：{total_images_with_prompts} 张")
                info_lines.append(f"  - 总点数：{total_points} 个")
                info_lines.append(f"  - 正点：{total_positive} 个")
                info_lines.append(f"  - 负点：{total_negative} 个")
            if self.ignition_point:
                info_lines.append(f"  - 起爆点：1 个")
            
            # 更新文本显示
            self.prompt_info_text.setPlainText("\n".join(info_lines))
            
            # 更新保存按钮状态（有参考点或起爆点时可保存）
            self.save_button.setEnabled(len(self.prompt_data) > 0 or self.ignition_point is not None)
            
        except Exception as e:
            print(f"❌ 更新参考点信息显示失败: {e}")
            self.prompt_info_text.setPlainText(f"显示错误: {str(e)}")
    
    def export_prompt_data_to_json(self, file_path: str):
        """
        导出prompt数据到JSON文件
        
        Args:
            file_path: 导出文件路径
        """
        try:
            export_data = {
                "image_paths": self.image_paths,
                "prompt_data": {str(k): v for k, v in self.prompt_data.items()}  # 键转换为字符串
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            
            print(f"✅ 参考点数据已导出到: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出参考点数据失败: {e}")
            return False
    
    def on_image_point_clicked(self, x: int, y: int, point_type: str):
        """
        处理图像点击事件
        
        Args:
            x, y: 点击的图像坐标
            point_type: 点类型 ('positive', 'negative', 'ignition')
        """
        try:
            if not self.is_prompt_selection_mode:
                return
            
            if point_type == 'ignition':
                # 设置起爆点（全局唯一）
                self.ignition_point = (x, y)
                print(f"设置起爆点: 坐标({x}, {y})")
                # 更新所有图像的起爆点显示
                self.update_ignition_point_display()
                # 更新信息面板显示
                self.update_prompt_info_display()
            else:
                # 添加正负点到数据结构
                is_positive = (point_type == 'positive')
                self.add_prompt_point(self.current_image_index, x, y, is_positive)
            
            print(f"图像点击: 索引{self.current_image_index}, 坐标({x}, {y}), 类型: {point_type}")
            
        except Exception as e:
            print(f"❌ 处理图像点击失败: {e}")
    
    def update_ignition_point_display(self):
        """更新起爆点显示"""
        try:
            # 更新当前图像的起爆点显示
            self.load_points_for_current_image()
            
        except Exception as e:
            print(f"❌ 更新起爆点显示失败: {e}")
    
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
    
    def cancel_current_image_points(self):
        """取消当前图像上的所有点"""
        try:
            if self.current_image_index in self.prompt_data:
                # 移除数据结构中的点
                del self.prompt_data[self.current_image_index]
                
                # 清空图像控件中的点显示
                self.extract_preview.clear_points()
                
                # 更新信息显示
                self.update_prompt_info_display()
                
                print(f"🗑️ 已取消图像{self.current_image_index}上的所有参考点")
                self.extract_status.setText("已取消当前图像的参考点选择")
                
        except Exception as e:
            print(f"❌ 取消当前图像点失败: {e}")
    
    def load_points_for_current_image(self):
        """为当前图像加载已有的点标记"""
        try:
            positive_points = []
            negative_points = []
            
            if self.current_image_index in self.prompt_data:
                points = self.prompt_data[self.current_image_index]["points"]
                labels = self.prompt_data[self.current_image_index]["labels"]
                
                # 分离正负点
                for point, label in zip(points, labels):
                    if label == 1:
                        positive_points.append(tuple(point))
                    else:
                        negative_points.append(tuple(point))
            
            # 设置到图像控件（包括起爆点）
            self.extract_preview.set_points(positive_points, negative_points, self.ignition_point)
            
            print(f"为图像{self.current_image_index}加载了{len(positive_points)}个正点, {len(negative_points)}个负点, 起爆点: {self.ignition_point}")
                
        except Exception as e:
            print(f"❌ 加载当前图像点失败: {e}")
    
    def save_extraction_sequence(self):
        """保存提取序列（按照example_data.json格式）"""
        try:
            if not self.prompt_data:
                QMessageBox.warning(self, "警告", "没有选择任何参考点，无法保存！")
                return
            
            if not self.image_paths:
                QMessageBox.warning(self, "警告", "没有加载图像序列，无法保存！")
                return
            
            # 选择保存文件
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存提取序列",
                "fireball_prompt_data.json",
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if file_path:
                # 导出数据
                success = self.export_prompt_data_to_json(file_path)
                
                if success:
                    QMessageBox.information(self, "成功", 
                                          f"提取序列已保存到:\n{file_path}\n\n"
                                          f"包含 {len(self.prompt_data)} 张图像的参考点数据")
                    self.extract_status.setText("序列保存成功")
                else:
                    QMessageBox.critical(self, "错误", "保存失败，请检查文件路径和权限！")
                    self.extract_status.setText("序列保存失败")
                    
        except Exception as e:
            print(f"❌ 保存提取序列失败: {e}")
            QMessageBox.critical(self, "错误", f"保存提取序列失败:\n{str(e)}")
            self.extract_status.setText("保存失败")
    
    def update_image_index_display(self):
        """更新图片索引显示"""
        try:
            if self.image_paths:
                current = self.current_image_index + 1  # 显示从1开始
                total = len(self.image_paths)
                self.image_index_label.setText(f"{current}/{total}")
            else:
                self.image_index_label.setText("0/0")
                
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
                
                # 清空输入框
                self.jump_input.clear()
                
                print(f"跳转到图片 {image_number} (索引: {image_index})")
                
            except ValueError:
                QMessageBox.warning(self, "警告", "请输入有效的数字！")
                
        except Exception as e:
            print(f"❌ 跳转到图片失败: {e}")
            QMessageBox.critical(self, "错误", f"跳转失败:\n{str(e)}")
