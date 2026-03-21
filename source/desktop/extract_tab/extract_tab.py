#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器视觉模块标签页（原特征提取 + 原输入侧能力：序列/图像文件夹/温度、参数与日志）
"""

import json
import os
import sys
import threading
from collections import deque
from typing import Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSplitter, QSlider, QComboBox, QLineEdit, QGroupBox,
                               QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QTextEdit, QScrollArea)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from .utils.segment_utils import build_time_diameter_series, run_segmentation_direct
from .utils.sequence_manager import SequenceManager
from .sequence_model import SequenceModel
from .ui_widgets.extract_tab_ui import ExtractTabUI
from .utils.info_builder import build_segmentation_info_text
from .controllers.prompt_controller import PromptController
from .controllers.chart_controller import ChartController
from .controllers.sequence_display_controller import SequencyDisplayController

# 添加路径以导入火球计算器
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fireball_radius_calculator import FireballCalculator
from diameter_process.diameter_drag_fitting import DiameterDragFitter


class ExtractTab(QWidget):
    """机器视觉模块标签页"""
    # 异步分割：日志与完成信号
    log_received = Signal(str)
    seg_finished = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 分割脚本流式输出：deque 控行数，与 QPlainTextEdit 同步裁剪
        self.max_status_lines = 500
        self.status_log_buffer: deque[str] = deque(maxlen=self.max_status_lines)
        
        # 初始化火球计算器和序列管理器
        self.fireball_calculator = FireballCalculator()
        self.sequence_manager = SequenceManager()
        self.sequence_model = SequenceModel(self.sequence_manager)
        
        # 初始化UI构建器并创建界面
        self.ui_builder = ExtractTabUI()
        self.ui_builder.create_main_layout(self)
        self.ui_components = self.ui_builder.get_ui_components()
        
        # 初始化控制器
        self.prompt_controller = PromptController(self, self.ui_builder, self.sequence_model)
        self.chart_controller = ChartController(self.ui_builder)
        self.display_controller = SequencyDisplayController(
            self.ui_builder,
            self.sequence_model,
            self.prompt_controller,
        )
        
        # 获取UI组件引用（为了向后兼容）
        self._setup_ui_component_references()
        
        self.setup_connections()
        self.init_charts()
        # 连接异步信号
        self.log_received.connect(self._on_segmentation_log)
        self.seg_finished.connect(self._on_segmentation_finished)
        self.append_run_log("[日志] 待开始 — 请导入爆炸序列文件或火球图像序列文件夹。")

    def _has_analysis_results(self) -> bool:
        """是否具备可保存的分析结果（直径曲线 + 拟合参数）。"""
        try:
            return self.chart_controller.has_analysis_results()
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
        # 运行日志（QPlainTextEdit，extract_status 与之同一引用）
        self.extract_status = self.ui_components['extract_status']
        self.run_log = self.ui_components.get('run_log', self.extract_status)
        
        # 按钮控件引用（仍在使用）
        self.sequence_btn = self.ui_components['sequence_btn']
        self.image_folder_btn = self.ui_components['image_folder_btn']
        self.temp_btn = self.ui_components['temp_btn']
        self.extract_btn = self.ui_components['extract_btn']
        self.reextract_btn = self.ui_components['reextract_btn']
        self.save_button = self.ui_components['save_button']
        self.export_segmentation_checkbox = self.ui_components['export_segmentation_checkbox']
        
        # 参数控件（爆炸信息 / 炸药参数）
        self.mv_explosion_duration = self.ui_components['mv_explosion_duration']
        self.mv_pixel_length = self.ui_components['mv_pixel_length']
        self.mv_explosive_type = self.ui_components['mv_explosive_type']
        self.mv_equivalent = self.ui_components['mv_equivalent']
        self.mv_al_percent = self.ui_components['mv_al_percent']

        self.prompt_btn = self.ui_components['prompt_btn']
        self.cancel_prompt_btn = self.ui_components['cancel_prompt_btn']
        self.positive_radio = self.ui_components['positive_radio']
        self.negative_radio = self.ui_components['negative_radio']
        self.ignition_radio = self.ui_components['ignition_radio']
        self.jump_btn = self.ui_components.get('jump_btn')
        
        # 信息显示控件（仍在使用）
        self.prompt_info_text = self.ui_components['prompt_info_text']
        
        # 以下控件已由各自的控制器管理，不再需要直接引用：
        # - extract_preview, extract_slider, extract_time_label: 由 SequencyDisplayController 管理
        # - temp_chart, diam_chart: 由 ChartController 管理
        # - prompt_btn, point_type_group, positive_radio, negative_radio, ignition_radio: 由 PromptController 管理
        # - check_bar: 由 SequencyDisplayController 管理
    
    def setup_connections(self):
        """设置信号连接"""
        # 时间轴控件（由 SequencyDisplayController 内部处理）
        
        # 侧边栏按钮（PromptController 相关的信号已在控制器内部连接）
        self.sequence_btn.clicked.connect(self.select_sequence_folder)
        self.image_folder_btn.clicked.connect(self.select_image_sequence_folder)
        self.temp_btn.clicked.connect(self.select_temperature_sequence)
        self.extract_btn.clicked.connect(self.start_feature_extraction)
        self.reextract_btn.clicked.connect(self.start_reextraction)
        # cancel_prompt_btn 的信号连接已由 PromptController 处理
        
        # 图像导航（由 SequencyDisplayController 内部处理）
        
        # 保存按钮
        self.save_button.clicked.connect(self.save_extraction_sequence)

        # 参数变更：仅更新内存中的模型，写盘在「其他操作前」flush
        for w in (self.mv_explosion_duration, self.mv_pixel_length, self.mv_equivalent, self.mv_al_percent):
            w.textChanged.connect(self._on_mv_parameter_text_changed)
        self.mv_explosive_type.currentTextChanged.connect(self._on_mv_parameter_text_changed)
    
    def append_run_log(self, line: str) -> None:
        """追加一行运行日志。"""
        try:
            w = self.ui_components.get('run_log')
            if w is None:
                return
            if hasattr(w, 'appendPlainText'):
                w.appendPlainText((line if line.endswith('\n') else line + '\n'))
            elif hasattr(w, 'append'):
                w.append(line)
        except Exception as e:
            print(f"append_run_log: {e}")

    def _scroll_run_log_to_bottom(self) -> None:
        w = self.run_log
        if w is None:
            return
        sb = w.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _trim_plain_text_top(self, w, n_blocks: int) -> None:
        """从文档顶部一次删除 n_blocks 个段落（批量删顶，避免逐行 setPlainText）。"""
        if n_blocks <= 0 or w is None:
            return
        doc = w.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        for _ in range(n_blocks):
            if not cursor.movePosition(
                QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.MoveAnchor
            ):
                break
        start = QTextCursor(doc)
        start.movePosition(QTextCursor.MoveOperation.Start)
        start.setPosition(cursor.position(), QTextCursor.MoveMode.KeepAnchor)
        start.removeSelectedText()

    def _status_set_plain(self, text: str) -> None:
        """整段替换日志区（用于重置等少量全量替换）。"""
        try:
            w = self.run_log
            if w is not None and hasattr(w, 'setPlainText'):
                w.setPlainText(text)
                self._scroll_run_log_to_bottom()
            elif w is not None and hasattr(w, 'setText'):
                w.setText(text)
        except Exception as e:
            print(f"_status_set_plain: {e}")

    def flush_parameters_before_action(self) -> None:
        """在执行其他操作前将当前参数写入当前序列 JSON（若有路径）。"""
        try:
            if not self.sequence_model.current_path:
                return
            self._sync_model_parameters_from_ui()
            ok, err = self.sequence_model.flush_sequence_json_to_disk()
            if not ok:
                self.append_run_log(f"⚠️ 自动保存序列失败: {err}")
        except Exception as e:
            print(f"flush_parameters_before_action: {e}")

    def flush_sequence_silent(self) -> None:
        """供主窗口关闭时调用：尽量把当前参数写回磁盘。"""
        try:
            self.flush_parameters_before_action()
        except Exception:
            pass

    def _sync_model_parameters_from_ui(self) -> None:
        if not self.sequence_model.current_path:
            return
        self.sequence_model.apply_parameters_from_ui(
            self.mv_explosive_type.currentText(),
            self.mv_equivalent.text().strip(),
            self.mv_al_percent.text().strip(),
            self.mv_explosion_duration.text().strip(),
            self.mv_pixel_length.text().strip(),
        )

    def _sync_ui_parameters_from_model(self) -> None:
        """从已加载序列回填参数控件。"""
        try:
            p = self.sequence_model.parameters
            if not p:
                return
            for w in (
                self.mv_explosion_duration,
                self.mv_pixel_length,
                self.mv_equivalent,
                self.mv_al_percent,
            ):
                w.blockSignals(True)
            self.mv_explosive_type.blockSignals(True)
            self.mv_explosion_duration.setText(str(p.get("explosion_duration", "140")))
            self.mv_pixel_length.setText(str(p.get("pixel_length", "0.01")))
            mt = str(p.get("material_type", "温压弹"))
            idx = self.mv_explosive_type.findText(mt)
            if idx >= 0:
                self.mv_explosive_type.setCurrentIndex(idx)
            else:
                self.mv_explosive_type.setCurrentText(mt)
            self.mv_equivalent.setText(str(p.get("equivalent", "1")))
            self.mv_al_percent.setText(str(p.get("al_percent", "30")))
            for w in (
                self.mv_explosion_duration,
                self.mv_pixel_length,
                self.mv_equivalent,
                self.mv_al_percent,
            ):
                w.blockSignals(False)
            self.mv_explosive_type.blockSignals(False)
        except Exception as e:
            print(f"_sync_ui_parameters_from_model: {e}")

    def _on_mv_parameter_text_changed(self, *args) -> None:
        """参数框编辑时仅同步内存（需已加载序列且有路径）。"""
        try:
            if not self.sequence_model.current_path:
                return
            self._sync_model_parameters_from_ui()
        except Exception:
            pass

    def select_image_sequence_folder(self):
        """选择图像文件夹，在同级生成 {文件夹名}_fireball_sequence.json 并加载。"""
        self.flush_parameters_before_action()
        folder_path = QFileDialog.getExistingDirectory(self, "选择火球图像序列文件夹", "")
        if not folder_path:
            return
        ok, msg, work_path = self.sequence_manager.create_work_sequence_from_image_folder(
            folder_path,
            self.mv_explosive_type.currentText(),
            self.mv_equivalent.text().strip(),
            self.mv_al_percent.text().strip(),
            self.mv_explosion_duration.text().strip(),
            self.mv_pixel_length.text().strip(),
        )
        if not ok or not work_path:
            QMessageBox.warning(self, "警告", msg or "无法创建工作序列文件")
            return
        try:
            self._reset_state_before_import()
            success, sequence_data, message = self.sequence_manager.load_sequence_file(work_path)
            if not success:
                QMessageBox.critical(self, "错误", f"加载工作序列失败:\n{message}")
                return
            self._apply_sequence_data(sequence_data, work_path)
            self.append_run_log(f"✓ 图像序列目录: {folder_path}")
            self.append_run_log(f"✓ 工作文件: {work_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入图像序列失败:\n{str(e)}")
            print(e)

    def select_temperature_sequence(self):
        """导入温度时间序列并立即写入当前序列 JSON。"""
        self.flush_parameters_before_action()
        if not self.sequence_model.current_path:
            QMessageBox.warning(self, "警告", "请先导入爆炸序列或图像序列。")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择火球温度时间序列文件",
            "",
            "CSV文件 (*.csv);;JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return
        t_data, T_data = self.sequence_manager.load_temperature_data_file(file_path)
        if not t_data or not T_data:
            QMessageBox.warning(self, "警告", "无法读取温度数据文件。")
            return
        pairs = [[float(t), float(T)] for t, T in zip(t_data, T_data)]
        self.sequence_model.set_temperature_pairs(pairs)
        ok, err = self.sequence_model.flush_sequence_json_to_disk()
        if not ok:
            QMessageBox.critical(self, "错误", f"写入温度数据失败:\n{err}")
            return
        self.update_temperature_chart(t_data, T_data)
        self.append_run_log(f"✓ 已导入温度序列: {os.path.basename(file_path)} ({len(pairs)} 点)")

    def _segmentation_lock_widgets(self):
        """分割进行中需禁用的控件（预览与时间轴除外）。"""
        widgets = [
            self.sequence_btn,
            self.image_folder_btn,
            self.temp_btn,
            self.extract_btn,
            self.reextract_btn,
            self.save_button,
            self.prompt_btn,
            self.cancel_prompt_btn,
            self.positive_radio,
            self.negative_radio,
            self.ignition_radio,
            self.mv_explosion_duration,
            self.mv_pixel_length,
            self.mv_explosive_type,
            self.mv_equivalent,
            self.mv_al_percent,
        ]
        if self.jump_btn is not None:
            widgets.append(self.jump_btn)
        ji = self.ui_components.get('jump_input')
        if ji is not None:
            widgets.append(ji)
        if self.export_segmentation_checkbox is not None:
            widgets.append(self.export_segmentation_checkbox)
        cb = self.ui_components.get('check_bar')
        if cb is not None:
            widgets.append(cb)
        return [w for w in widgets if w is not None]

    def _set_segmentation_ui_locked(self, locked: bool) -> None:
        for w in self._segmentation_lock_widgets():
            try:
                w.setEnabled(not locked)
            except Exception:
                pass

    def get_sidebar_widget(self):
        """获取机器视觉模块的侧边栏组件"""
        if not hasattr(self, '_sidebar_widget'):
            # 使用UI构建器创建侧边栏
            self._sidebar_widget = self.ui_builder.create_sidebar_widget()
        
        return self._sidebar_widget
    
    def _reset_state_before_import(self):
        """在导入新序列前清空之前的内存状态与UI显示"""
        try:
            print("🧹 重置状态：清空旧的序列、特征点与分割结果…")
            # 1) 清空内存数据
            self.sequence_model.reset()
            # 重置控制器状态
            self.prompt_controller.reset()
            self.display_controller.reset()
            
            # 2) 保存按钮初始禁用（等待生成直径/拟合结果）
            self._update_save_button_state()
            
            # 3) 重置图表
            try:
                self.chart_controller.reset()
            except Exception as e:
                print(f"⚠️ 重置图表时出错: {e}")
            
            # 4) 重置状态文本与按钮
            try:
                self._status_set_plain("待开始")
                self.extract_btn.setVisible(True)
                if hasattr(self, 'reextract_btn'):
                    self.reextract_btn.setVisible(False)
                # 重置时启用 prompt 控件（新导入序列，没有分割结果）
                self.prompt_controller.set_prompt_controls_enabled(True)
            except Exception as e:
                print(f"⚠️ 重置状态控件时出错: {e}")
            
            print("✅ 状态重置完成")
        except Exception as e:
            print(f"❌ 重置状态失败: {e}")

    def _apply_sequence_data(self, sequence_data: dict, sequence_file_path: Optional[str] = None) -> bool:
        """将已加载的序列数据应用到界面与状态中，返回是否存在分割结果。"""
        try:
            # 刷新模型缓存
            self.sequence_model.apply_sequence_dict(sequence_data, sequence_file_path)

            image_paths = self.sequence_model.image_paths
            if not image_paths:
                QMessageBox.warning(self, "警告", "序列文件中没有图像路径！")
                self.append_run_log("无图像数据")
                return False

            # 应用序列显示（设置时间轴、显示第一张图像等）
            self.display_controller.apply_sequence()

            # 加载温度数据
            time_data, temp_data = self.sequence_model.get_temperature_series()
            if time_data and temp_data:
                self.update_temperature_chart(time_data, temp_data)

            # 分割结果优先
            if self.sequence_model.has_segmentation_data():
                segmentation_results = self.sequence_model.get_segmentation_results()
                summary = self.sequence_model.get_segmentation_summary()
                self.update_segmentation_info_display(summary["success"], summary["total"])
                self.update_diameter_chart_from_segmentation_results(segmentation_results)
                self._update_save_button_state()
                self.extract_btn.setVisible(False)
                self.reextract_btn.setVisible(True)
                # 禁用 prompt 相关控件（已有分割结果，不应再修改参考点）
                self.prompt_controller.set_prompt_controls_enabled(False)
                # 同步显示控制器以刷新显示模式
                self.display_controller.sync_to_model()
            else:
                self.extract_btn.setVisible(True)
                self.reextract_btn.setVisible(False)
                # 启用 prompt 相关控件（没有分割结果，可以修改参考点）
                self.prompt_controller.set_prompt_controls_enabled(True)
                self.prompt_controller.sync_from_model()
                # 同步显示控制器以刷新显示模式
                self.display_controller.sync_to_model()

            # 序列摘要与状态
            summary = self.sequence_model.get_sequence_summary()
            status_msg = f"已加载序列: {summary['image_count']} 个文件，时长: {summary['explosion_duration']}ms"
            if summary['has_temperature_data']:
                status_msg += f"，温度数据: {summary['temperature_points']} 点"
            if summary['has_prompt_data']:
                status_msg += f"，参考点数据: {summary['total_prompt_points']} 点"
            if summary['has_ignition_point']:
                status_msg += f"，起爆点: {summary['ignition_point']}"
            self._sync_ui_parameters_from_model()
            self.append_run_log(status_msg)
            return self.sequence_model.has_segmentation_data()

        except Exception as e:
            print(f"❌ 应用序列数据失败: {e}")
            QMessageBox.critical(self, "错误", f"应用序列数据失败:\n{str(e)}")
            return False
    
    
    def select_sequence_folder(self):
        """选择火球爆炸序列JSON文件"""
        self.flush_parameters_before_action()
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
                    self.append_run_log("文件加载失败")
                    return
                
                # 统一应用逻辑
                self._apply_sequence_data(sequence_data, file_path)
                # 序列加载完成
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"处理序列文件失败:\n{str(e)}")
                self.append_run_log("处理失败")
                print(f"处理序列文件失败: {e}")
    
    def init_charts(self):
        """初始化图表"""
        self.chart_controller.reset()
    
    def update_temperature_chart(self, time_data, temp_data):
        """更新温度图表"""
        try:
            print(f"开始更新温度图表: {len(time_data)} 个数据点")
            self.chart_controller.update_temperature(time_data, temp_data)
            print("✅ 温度图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新温度图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_diameter_chart(self, time_data, diameter_data):
        """更新直径图表（提取完成后调用）"""
        try:
            print(f"📊 update_diameter_chart 被调用")
            self.chart_controller.update_diameter_raw(time_data, diameter_data)
            print("✅ 直径图表更新完成")
            
        except Exception as e:
            print(f"❌ 更新直径图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def start_feature_extraction(self):
        """开始特征提取（调用分割脚本）"""
        try:
            self.flush_parameters_before_action()
            print("🔥 开始特征提取...")
            self.append_run_log("正在检查序列文件...")
            self.extract_btn.setEnabled(False)
            
            # 检查是否有序列数据
            if not self.sequence_model.sequence_data:
                QMessageBox.warning(self, "警告", "请先加载火球爆炸序列文件！")
                self.append_run_log("请先加载序列文件")
                self.extract_btn.setEnabled(True)
                return
            
            # 检查序列文件路径
            if not self.sequence_model.current_path:
                QMessageBox.warning(self, "警告", "无法找到序列文件路径！")
                self.append_run_log("序列文件路径丢失")
                self.extract_btn.setEnabled(True)
                return
            
            # 检查分割状态
            segmentation_status = self.check_segmentation_status()
            
            if segmentation_status == 'no_prompt_data':
                # 情况1：没有prompt数据
                QMessageBox.warning(self, "警告", 
                    "序列文件中没有特征点数据！\n\n请先：\n1. 点击'开始选择参考点'\n2. 在图像上选择正负点\n3. 完成特征点选择后再进行提取")
                self.append_run_log("请先选择特征点")
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
                    self.append_run_log("已有分割结果")
                    return
            
            # 情况3：有prompt数据但没有分割结果，执行分割
            print("开始执行分割脚本...")
            self.append_run_log("正在执行分割脚本...")
            
            # 调用异步分割脚本（完成后的处理在 _on_segmentation_finished 中）
            self.run_segmentation_script(self.sequence_model.current_path)
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            self.append_run_log("特征提取失败")
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
            # 注意：使用当前内存中的模型缓存判断，若仍有结果则认为已分割
            if self.sequence_model.has_segmentation_data():
                return 'already_segmented'
            
            # 检查是否有prompt数据
            prompt_data = self.sequence_model.get_prompt_data()
            if not prompt_data or len(prompt_data.keys()) == 0:
                return 'no_prompt_data'
            
            return 'ready_for_segmentation'
            
        except Exception as e:
            print(f"❌ 检查分割状态失败: {e}")
            return 'no_prompt_data'
    
    def run_segmentation_script(self, sequence_file_path: str) -> bool:
        """异步运行分割脚本：后台线程读日志，通过信号更新UI，不阻塞主线程。"""
        # 1) 禁用交互控件（除预览与时间轴外）
        try:
            self.prompt_controller.set_prompt_controls_enabled(False)
            self._set_segmentation_ui_locked(True)
        except Exception:
            pass

        self.status_log_buffer.clear()
        self.status_log_buffer.append("正在执行分割脚本…")
        self._status_set_plain("正在执行分割脚本…")

        def worker():
            def on_line(line: str):
                self.log_received.emit(line)
            ok = run_segmentation_direct(sequence_file_path, on_output_line=on_line)
            self.seg_finished.emit(ok)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _on_segmentation_log(self, line: str):
        w = self.run_log
        if w is None or not hasattr(w, 'appendPlainText'):
            return
        # 一行日志内可能含多个 \n，按行写入以便 block 数与 deque 一致
        for segment in line.split('\n'):
            text = segment.rstrip('\r')
            if not text:
                continue
            self.status_log_buffer.append(text)
            w.appendPlainText(text)
            excess = w.document().blockCount() - self.max_status_lines
            if excess > 0:
                self._trim_plain_text_top(w, excess)
        self._scroll_run_log_to_bottom()

    def _on_segmentation_finished(self, ok: bool):
        # 恢复控件（参考点启用状态由 _apply_sequence_data 根据是否已分割决定）
        try:
            self._set_segmentation_ui_locked(False)
        except Exception:
            pass

        if ok:
            self.append_run_log("分割完成，正在加载分割结果…")
            self.reload_sequence_with_segmentation_results()
            self.append_run_log("特征提取完成")
            self._update_save_button_state()
        else:
            self.append_run_log("分割脚本执行失败")
            try:
                self.prompt_controller.set_prompt_controls_enabled(True)
            except Exception:
                pass
            QMessageBox.critical(self, "错误", "分割脚本执行失败！\n请检查控制台输出获取详细信息。")
    
    def prepare_for_reextraction(self):
        """准备重新提取：清除分割结果，重置为特征点选择模式"""
        try:
            self.flush_parameters_before_action()
            success, message = self.sequence_model.clear_segmentation_results()
            if message:
                print(("✅ " if success else "❌ ") + message)
                
            # 清空直径图表和直径速率图表
            try:
                self.chart_controller.clear_diameter()
            except Exception as e:
                print(f"⚠️ 清空直径图表失败: {e}")
            
            # 重新加载特征点数据
            self.prompt_controller.sync_from_model()
            # 重新启用 prompt 相关控件（清除分割结果后，可以重新选择参考点）
            self.prompt_controller.set_prompt_controls_enabled(True)
            
            # 刷新显示控制器以切换到 prompt 模式
            self.display_controller.handle_segmentation_update()

            # 更新状态
            self.append_run_log("已清除分割结果，请重新选择特征点")
            
            print("✅ 已准备重新提取")
                
        except Exception as e:
            print(f"❌ 准备重新提取失败: {e}")
    
    def reload_sequence_with_segmentation_results(self):
        """重新加载序列文件并显示分割结果"""
        try:
            # 重新加载分割后的序列文件（原文件名 + "_segmented"）
            from pathlib import Path
            current_path = self.sequence_model.current_path
            if not current_path:
                print("⚠️ 当前无序列文件路径，无法重新加载分割结果")
                return False
            original_path = Path(current_path)
            segmented_path = original_path.with_name(f"{original_path.stem}_segmented{original_path.suffix}")

            if not segmented_path.exists():
                print(f"⚠️ 分割后的序列文件不存在: {segmented_path}")
                return False

            success, sequence_data, message = self.sequence_manager.load_sequence_file(str(segmented_path))
            
            if not success:
                print(f"❌ 重新加载序列文件失败: {message}")
                return False
            
            # 统一应用逻辑（内部会调用 display_controller.apply_sequence 和 sync_to_model）
            return self._apply_sequence_data(sequence_data, str(segmented_path))
            
        except Exception as e:
            print(f"❌ 重新加载序列文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_diameter_chart_from_segmentation_results(self, segmentation_results):
        """根据分割结果更新直径图表（仅绘制成功点，不插值）"""
        try:
            self.sequence_model.update_segmentation_results(segmentation_results)
            series = build_time_diameter_series(
                segmentation_results, 
                float(self.sequence_model.explosion_duration_ms),
                float(self.sequence_model.pixel_length)
            )
            if not series:
                # 清空图表
                self.chart_controller.clear_diameter()
                return
            time_data = [t for t, _ in series]
            diameter_data = [d for _, d in series]

            # 调用拖曳曲线拟合，获取 K、B、C 与截断点
            fit_result = None
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
            except Exception as e:
                print(f"⚠️ 直径拖曳拟合失败，退回仅绘制数据点: {e}")
                fit_result = None

            # 使用 ChartController 绘制（带拟合与截断线，若有）
            try:
                if fit_result and fit_result.get('success', False):
                    self.chart_controller.update_diameter_with_fit(time_data, diameter_data, fit_result)
                else:
                    self.chart_controller.update_diameter_raw(time_data, diameter_data)
            except Exception as e:
                print(f"⚠️ 调用直径图更新接口失败，退回简单绘制: {e}")
                self.chart_controller.update_diameter_raw(time_data, diameter_data)
        except Exception as e:
            print(f"❌ 更新直径图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    
    
    
    def start_reextraction(self):
        """开始重新提取"""
        try:
            self.flush_parameters_before_action()
            print("🔄 开始重新提取...")
            
            # 检查是否有序列数据
            if not self.sequence_model.sequence_data:
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
    
    def save_extraction_sequence(self):
        """保存提取序列（按照example_data.json格式）"""
        try:
            self.flush_parameters_before_action()
            # 改为保存直径与拟合参数结果
            diameter_series = self.chart_controller.get_cached_diameter()
            if not diameter_series:
                QMessageBox.warning(self, "警告", "没有直径数据可保存！")
                return
            
            drag_fit_result = self.chart_controller.get_cached_drag_fit()
            if not drag_fit_result or not (
                isinstance(drag_fit_result, dict) and drag_fit_result.get('K') is not None
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
                
                # 如果选中了"同时导出分割图片"，则导出分割图片
                export_images_success = False
                if self.export_segmentation_checkbox.isChecked():
                    export_images_success = self.export_segmentation_images(file_path)
                
                if success:
                    message = f"分析结果已保存到:\n{file_path}\n\n包含 {len(diameter_series)} 个直径数据点与拟合参数"
                    if export_images_success:
                        message += "\n\n分割图片已成功导出"
                    elif self.export_segmentation_checkbox.isChecked():
                        message += "\n\n⚠️ 分割图片导出失败，请检查控制台输出"
                    QMessageBox.information(self, "成功", message)
                    self.append_run_log("分析结果保存成功" + ("，分割图片已导出" if export_images_success else ""))
                else:
                    QMessageBox.critical(self, "错误", "保存失败，请检查文件路径和权限！")
                    self.append_run_log("分析结果保存失败")
                    
        except Exception as e:
            print(f"❌ 保存提取序列失败: {e}")
            QMessageBox.critical(self, "错误", f"保存提取序列失败:\n{str(e)}")
            self.append_run_log("保存失败")
    
    def export_analysis_results_to_json(self, file_path: str) -> bool:
        """导出直径曲线、爆炸参数、拖曳拟合结果到 JSON 文件。"""
        try:
            from pathlib import Path
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            export_data = self.sequence_model.build_export_payload(
                diameter_series=self.chart_controller.get_cached_diameter() or [],
                drag_fit_result=self.chart_controller.get_cached_drag_fit(),
            )

            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 导出分析结果失败: {e}")
            return False
    
    def export_segmentation_images(self, json_file_path: str) -> bool:
        """导出分割图片到指定目录"""
        try:
            from pathlib import Path
            from .utils.sequence_image_composer import compose_and_save
            
            # 检查是否有分割结果
            if not self.sequence_model.has_segmentation_data():
                print("⚠️ 没有分割结果，无法导出分割图片")
                return False
            
            # 获取图像路径和分割结果
            image_paths = self.sequence_model.image_paths
            segmentation_results = self.sequence_model.get_segmentation_results()
            
            if not image_paths or not segmentation_results:
                print("⚠️ 图像路径或分割结果为空，无法导出分割图片")
                return False
            
            # 确定输出目录（与JSON文件同目录，使用JSON文件名作为子目录名）
            json_path = Path(json_file_path)
            output_dir = json_path.parent / f"{json_path.stem}_segmented_images"
            
            # 组合并保存图像
            success, saved_paths = compose_and_save(
                image_paths,
                segmentation_results,
                str(output_dir),
                base_name="fireball_segmented",
                image_format="jpg",
                start_index=0
            )
            
            if success:
                print(f"✓ 成功导出 {len(saved_paths)} 张分割图片到 {output_dir}")
                return True
            else:
                print(f"❌ 导出分割图片失败")
                return False
                
        except Exception as e:
            print(f"❌ 导出分割图片失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
