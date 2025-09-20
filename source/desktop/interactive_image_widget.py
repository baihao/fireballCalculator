#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式图像预览控件
支持在图像上点击选择点并显示标记
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QLabel, QWidget
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QMouseEvent
from typing import List, Tuple, Optional


class InteractiveImageWidget(QLabel):
    """支持交互的图像预览控件"""
    
    # 信号：点击时发出 (x, y, point_type)
    point_clicked = Signal(int, int, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 图像相关属性
        self.original_image = None  # 原始图像
        self.display_pixmap = None  # 显示的pixmap
        self.image_path = None      # 当前图像路径
        
        # 点标记相关属性
        self.positive_points = []   # 正点列表 [(x, y), ...]
        self.negative_points = []   # 负点列表 [(x, y), ...]
        
        # 交互状态
        self.interaction_mode = 'none'  # 'none', 'positive', 'negative'
        self.is_interactive = False     # 是否启用交互
        
        # 设置基本属性
        self.setMinimumSize(400, 300)
        self.setMaximumSize(600, 450)  # 设置最大尺寸防止过度放大
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #374151;
                border-radius: 8px;
                background-color: #1f2937;
            }
        """)
        self.setText("请选择图像文件")
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
    
    def set_image(self, image_path: str):
        """
        设置要显示的图像
        
        Args:
            image_path: 图像文件路径
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                self.setText(f"无法加载图像: {image_path}")
                return False
            
            # 转换为RGB格式
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_path = image_path
            
            # 清空之前的点标记
            self.clear_points()
            
            # 更新显示
            self.update_display()
            
            return True
            
        except Exception as e:
            print(f"❌ 设置图像失败: {e}")
            self.setText(f"图像加载失败: {str(e)}")
            return False
    
    def set_interaction_mode(self, mode: str):
        """
        设置交互模式
        
        Args:
            mode: 交互模式 ('none', 'positive', 'negative')
        """
        self.interaction_mode = mode
        self.is_interactive = (mode != 'none')
        
        # 更新鼠标光标
        if self.is_interactive:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        print(f"交互模式切换为: {mode}")
    
    def add_point(self, x: int, y: int, is_positive: bool = True):
        """
        添加点标记
        
        Args:
            x, y: 点坐标（图像坐标系）
            is_positive: 是否为正点
        """
        try:
            if is_positive:
                self.positive_points.append((x, y))
                print(f"添加正点: ({x}, {y})")
            else:
                self.negative_points.append((x, y))
                print(f"添加负点: ({x}, {y})")
            
            # 更新显示
            self.update_display()
            
        except Exception as e:
            print(f"❌ 添加点标记失败: {e}")
    
    def remove_last_point(self, is_positive: bool = True):
        """
        移除最后一个点
        
        Args:
            is_positive: 是否移除正点
        """
        try:
            if is_positive and self.positive_points:
                removed = self.positive_points.pop()
                print(f"移除正点: {removed}")
            elif not is_positive and self.negative_points:
                removed = self.negative_points.pop()
                print(f"移除负点: {removed}")
            
            # 更新显示
            self.update_display()
            
        except Exception as e:
            print(f"❌ 移除点标记失败: {e}")
    
    def clear_points(self):
        """清空所有点标记"""
        self.positive_points = []
        self.negative_points = []
        self.update_display()
        print("🗑️ 清空所有点标记")
    
    def set_points(self, positive_points: List[Tuple[int, int]], negative_points: List[Tuple[int, int]]):
        """
        设置点标记（用于加载已有数据）
        
        Args:
            positive_points: 正点列表
            negative_points: 负点列表
        """
        self.positive_points = positive_points.copy()
        self.negative_points = negative_points.copy()
        self.update_display()
        print(f"设置点标记: {len(positive_points)}个正点, {len(negative_points)}个负点")
    
    def get_points(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        获取当前的点标记
        
        Returns:
            Tuple: (正点列表, 负点列表)
        """
        return self.positive_points.copy(), self.negative_points.copy()
    
    def update_display(self):
        """更新图像显示，包括点标记"""
        try:
            if self.original_image is None:
                return
            
            # 创建显示图像的副本
            display_image = self.original_image.copy()
            
            # 绘制点标记
            self._draw_points_on_image(display_image)
            
            # 转换为QPixmap并缩放显示
            self._convert_and_display(display_image)
            
        except Exception as e:
            print(f"❌ 更新显示失败: {e}")
    
    def _draw_points_on_image(self, image: np.ndarray):
        """在图像上绘制点标记"""
        try:
            # 绘制正点（红色十字）
            for x, y in self.positive_points:
                self._draw_cross(image, x, y, (255, 0, 0), size=12, thickness=3)
            
            # 绘制负点（蓝色十字）
            for x, y in self.negative_points:
                self._draw_cross(image, x, y, (0, 0, 255), size=12, thickness=3)
                
        except Exception as e:
            print(f"❌ 绘制点标记失败: {e}")
    
    def _draw_cross(self, image: np.ndarray, x: int, y: int, color: Tuple[int, int, int], 
                   size: int = 12, thickness: int = 3):
        """
        在图像上绘制十字标记
        
        Args:
            image: 图像数组
            x, y: 十字中心坐标
            color: RGB颜色
            size: 十字大小
            thickness: 线条粗细
        """
        try:
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                # 绘制水平线
                start_x = max(0, x - size // 2)
                end_x = min(w - 1, x + size // 2)
                cv2.line(image, (start_x, y), (end_x, y), color, thickness)
                
                # 绘制垂直线
                start_y = max(0, y - size // 2)
                end_y = min(h - 1, y + size // 2)
                cv2.line(image, (x, start_y), (x, end_y), color, thickness)
                
        except Exception as e:
            print(f"❌ 绘制十字失败: {e}")
    
    def _convert_and_display(self, image: np.ndarray):
        """将numpy图像转换为QPixmap并显示"""
        try:
            # 转换为QPixmap
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QPixmap.fromImage(
                QLabel().grab().toImage().rgbSwapped()
            )
            
            # 使用更简单的方法
            # 将numpy数组转换为字节
            image_bytes = image.tobytes()
            
            # 创建QPixmap
            from PySide6.QtGui import QImage
            qt_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # 缩放以适应固定的最大尺寸
            max_size = self.maximumSize()
            scaled_pixmap = pixmap.scaled(
                max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.display_pixmap = scaled_pixmap
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"❌ 转换显示失败: {e}")
    
    def mousePressEvent(self, event: QMouseEvent):
        """处理鼠标点击事件"""
        try:
            if not self.is_interactive or self.original_image is None:
                return
            
            # 获取点击位置
            click_pos = event.pos()
            
            # 将控件坐标转换为图像坐标
            image_x, image_y = self._widget_to_image_coords(click_pos.x(), click_pos.y())
            
            if image_x is not None and image_y is not None:
                # 根据当前模式添加点
                if self.interaction_mode == 'positive':
                    self.add_point(image_x, image_y, True)
                    self.point_clicked.emit(image_x, image_y, 'positive')
                elif self.interaction_mode == 'negative':
                    self.add_point(image_x, image_y, False)
                    self.point_clicked.emit(image_x, image_y, 'negative')
            
        except Exception as e:
            print(f"❌ 处理鼠标点击失败: {e}")
    
    def _widget_to_image_coords(self, widget_x: int, widget_y: int) -> Tuple[Optional[int], Optional[int]]:
        """
        将控件坐标转换为图像坐标
        
        Args:
            widget_x, widget_y: 控件坐标
            
        Returns:
            Tuple: (图像x坐标, 图像y坐标) 或 (None, None)
        """
        try:
            if self.display_pixmap is None or self.original_image is None:
                return None, None
            
            # 获取显示的pixmap尺寸和位置
            pixmap_rect = self.display_pixmap.rect()
            widget_rect = self.rect()
            
            # 计算pixmap在控件中的位置（居中显示）
            pixmap_x = (widget_rect.width() - pixmap_rect.width()) // 2
            pixmap_y = (widget_rect.height() - pixmap_rect.height()) // 2
            
            # 检查点击是否在pixmap区域内
            relative_x = widget_x - pixmap_x
            relative_y = widget_y - pixmap_y
            
            if (0 <= relative_x < pixmap_rect.width() and 
                0 <= relative_y < pixmap_rect.height()):
                
                # 将pixmap坐标转换为原始图像坐标
                scale_x = self.original_image.shape[1] / pixmap_rect.width()
                scale_y = self.original_image.shape[0] / pixmap_rect.height()
                
                image_x = int(relative_x * scale_x)
                image_y = int(relative_y * scale_y)
                
                # 确保坐标在图像范围内
                image_x = max(0, min(self.original_image.shape[1] - 1, image_x))
                image_y = max(0, min(self.original_image.shape[0] - 1, image_y))
                
                return image_x, image_y
            
            return None, None
            
        except Exception as e:
            print(f"❌ 坐标转换失败: {e}")
            return None, None
    
    def get_point_info(self) -> dict:
        """
        获取点信息
        
        Returns:
            dict: 包含点信息的字典
        """
        return {
            'positive_points': self.positive_points.copy(),
            'negative_points': self.negative_points.copy(),
            'total_points': len(self.positive_points) + len(self.negative_points),
            'image_path': self.image_path
        }
    
    def set_interactive_enabled(self, enabled: bool):
        """
        设置是否启用交互
        
        Args:
            enabled: 是否启用交互
        """
        self.is_interactive = enabled
        if not enabled:
            self.interaction_mode = 'none'
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor)


def create_interactive_image_widget(parent=None) -> InteractiveImageWidget:
    """
    创建交互式图像控件的便捷函数
    
    Args:
        parent: 父控件
        
    Returns:
        InteractiveImageWidget: 交互式图像控件实例
    """
    return InteractiveImageWidget(parent)


if __name__ == "__main__":
    # 测试模块功能
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton
    import sys
    
    app = QApplication(sys.argv)
    
    # 创建测试窗口
    window = QWidget()
    layout = QVBoxLayout()
    
    # 创建交互式图像控件
    image_widget = create_interactive_image_widget()
    
    # 连接信号
    def on_point_clicked(x, y, point_type):
        print(f"点击事件: ({x}, {y}), 类型: {point_type}")
    
    image_widget.point_clicked.connect(on_point_clicked)
    
    # 添加测试按钮
    def toggle_positive():
        image_widget.set_interaction_mode('positive')
        print("切换到正点选择模式")
    
    def toggle_negative():
        image_widget.set_interaction_mode('negative')
        print("切换到负点选择模式")
    
    def clear_points():
        image_widget.clear_points()
        print("清空所有点")
    
    pos_btn = QPushButton("选择正点")
    pos_btn.clicked.connect(toggle_positive)
    
    neg_btn = QPushButton("选择负点")
    neg_btn.clicked.connect(toggle_negative)
    
    clear_btn = QPushButton("清空点")
    clear_btn.clicked.connect(clear_points)
    
    layout.addWidget(image_widget)
    layout.addWidget(pos_btn)
    layout.addWidget(neg_btn)
    layout.addWidget(clear_btn)
    
    window.setLayout(layout)
    window.setWindowTitle("交互式图像控件测试")
    window.resize(600, 500)
    window.show()
    
    print("交互式图像控件测试启动")
    print("请加载图像文件并测试点击功能")
    
    sys.exit(app.exec())
