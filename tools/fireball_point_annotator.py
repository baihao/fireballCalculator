#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球图片点标注工具

功能：
1. 加载并显示火球图片
2. 在图片上点击标注点，用红色十字显示
3. 显示标注点的坐标
4. 显示该点距离参考点(574, 542)的距离
5. 多次点击时，以最后一次标注点为准
"""

import sys
import os
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QFileDialog,
                               QMessageBox, QScrollArea)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont


class ImageLabel(QLabel):
    """可点击的图片标签，用于显示图片和标注点"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.setMinimumSize(800, 600)
        
        # 存储图片和标注点
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 标注点坐标（图片坐标系）
        self.annotated_point = None
        
        # 参考点坐标
        self.reference_point = QPoint(574, 542)
        
        # 信号：标注点改变时触发
        self.point_annotated = None  # 将在主窗口中设置回调
    
    def load_image(self, image_path):
        """加载图片"""
        try:
            self.original_pixmap = QPixmap(image_path)
            if self.original_pixmap.isNull():
                return False
            self.update_scaled_pixmap()
            return True
        except Exception as e:
            print(f"加载图片失败: {e}")
            return False
    
    def update_scaled_pixmap(self):
        """更新缩放后的图片"""
        if self.original_pixmap is None:
            return
        
        # 获取标签大小
        label_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        # 计算缩放比例，保持宽高比
        scale_x = label_size.width() / pixmap_size.width()
        scale_y = label_size.height() / pixmap_size.height()
        self.scale_factor = min(scale_x, scale_y)
        
        # 缩放图片
        scaled_size = pixmap_size * self.scale_factor
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # 计算偏移量（居中显示）
        self.offset_x = (label_size.width() - scaled_size.width()) // 2
        self.offset_y = (label_size.height() - scaled_size.height()) // 2
        
        self.update()
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图片"""
        super().resizeEvent(event)
        self.update_scaled_pixmap()
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if self.scaled_pixmap is None:
            return
        
        # 获取点击位置（相对于标签）
        label_pos = event.position().toPoint()
        
        # 转换为图片坐标系
        img_x = (label_pos.x() - self.offset_x) / self.scale_factor
        img_y = (label_pos.y() - self.offset_y) / self.scale_factor
        
        # 检查是否在图片范围内
        pixmap_size = self.original_pixmap.size()
        if 0 <= img_x < pixmap_size.width() and 0 <= img_y < pixmap_size.height():
            self.annotated_point = QPoint(int(img_x), int(img_y))
            self.update()
            
            # 触发回调
            if self.point_annotated:
                self.point_annotated(self.annotated_point)
    
    def paintEvent(self, event):
        """绘制事件"""
        super().paintEvent(event)
        
        if self.scaled_pixmap is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制图片
        painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)
        
        # 绘制参考点（蓝色十字）
        ref_x = self.offset_x + self.reference_point.x() * self.scale_factor
        ref_y = self.offset_y + self.reference_point.y() * self.scale_factor
        pen = QPen(QColor(0, 150, 255), 2)  # 蓝色
        painter.setPen(pen)
        cross_size = 15
        painter.drawLine(ref_x - cross_size, ref_y, ref_x + cross_size, ref_y)
        painter.drawLine(ref_x, ref_y - cross_size, ref_x, ref_y + cross_size)
        
        # 绘制标注点（红色十字）
        if self.annotated_point:
            ann_x = self.offset_x + self.annotated_point.x() * self.scale_factor
            ann_y = self.offset_y + self.annotated_point.y() * self.scale_factor
            pen = QPen(QColor(255, 0, 0), 3)  # 红色，更粗
            painter.setPen(pen)
            cross_size = 20
            painter.drawLine(ann_x - cross_size, ann_y, ann_x + cross_size, ann_y)
            painter.drawLine(ann_x, ann_y - cross_size, ann_x, ann_y + cross_size)
            
            # 绘制标注点坐标文本
            pen = QPen(QColor(255, 255, 0), 1)  # 黄色
            painter.setPen(pen)
            font = QFont("Arial", 10, QFont.Bold)
            painter.setFont(font)
            text = f"({self.annotated_point.x()}, {self.annotated_point.y()})"
            text_rect = painter.fontMetrics().boundingRect(text)
            text_x = ann_x + cross_size + 5
            text_y = ann_y - cross_size - 5
            # 确保文本不超出边界
            if text_x + text_rect.width() > self.width():
                text_x = ann_x - cross_size - text_rect.width() - 5
            if text_y < 0:
                text_y = ann_y + cross_size + text_rect.height() + 5
            
            # 绘制文本背景（半透明黑色）
            bg_rect = text_rect.translated(text_x, text_y)
            bg_rect.adjust(-3, -2, 3, 2)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 200))
            painter.drawText(text_x, text_y + text_rect.height(), text)


class FireballPointAnnotator(QMainWindow):
    """火球图片点标注工具主窗口"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("火球图片点标注工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        # 加载图片按钮
        self.load_btn = QPushButton("加载图片")
        self.load_btn.setMinimumHeight(35)
        self.load_btn.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_btn)
        
        # 清除标注按钮
        self.clear_btn = QPushButton("清除标注")
        self.clear_btn.setMinimumHeight(35)
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self.clear_annotation)
        toolbar_layout.addWidget(self.clear_btn)
        
        toolbar_layout.addStretch()
        
        # 图片路径标签
        self.path_label = QLabel("未加载图片")
        self.path_label.setStyleSheet("color: #888; padding: 5px;")
        toolbar_layout.addWidget(self.path_label)
        
        main_layout.addLayout(toolbar_layout)
        
        # 图片显示区域（使用滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #1e1e1e;")
        
        self.image_label = ImageLabel()
        self.image_label.point_annotated = self.on_point_annotated
        scroll_area.setWidget(self.image_label)
        
        main_layout.addWidget(scroll_area, stretch=1)
        
        # 信息显示区域
        info_layout = QVBoxLayout()
        info_layout.setSpacing(10)
        
        # 标注点信息
        info_title = QLabel("标注点信息")
        info_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fff; padding: 5px;")
        info_layout.addWidget(info_title)
        
        self.coord_label = QLabel("坐标: 未标注")
        self.coord_label.setStyleSheet("font-size: 12px; color: #fff; padding: 5px; background-color: #2b2b2b; border-radius: 5px;")
        info_layout.addWidget(self.coord_label)
        
        self.distance_label = QLabel("距离参考点(574, 542): 未标注")
        self.distance_label.setStyleSheet("font-size: 12px; color: #fff; padding: 5px; background-color: #2b2b2b; border-radius: 5px;")
        info_layout.addWidget(self.distance_label)
        
        # 参考点信息
        ref_title = QLabel("参考点信息")
        ref_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fff; padding: 5px; margin-top: 10px;")
        info_layout.addWidget(ref_title)
        
        ref_info = QLabel("参考点坐标: (574, 542)\n（蓝色十字标记）")
        ref_info.setStyleSheet("font-size: 12px; color: #88ccff; padding: 5px; background-color: #2b2b2b; border-radius: 5px;")
        info_layout.addWidget(ref_info)
        
        info_layout.addStretch()
        
        # 将信息区域添加到主布局
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        info_widget.setFixedWidth(300)
        info_widget.setStyleSheet("background-color: #1e1e1e; padding: 10px;")
        
        # 创建水平布局，将图片和信息区域并排显示
        content_layout = QHBoxLayout()
        content_layout.addWidget(scroll_area, stretch=1)
        content_layout.addWidget(info_widget)
        
        # 替换原来的scroll_area布局
        main_layout.removeWidget(scroll_area)
        main_layout.addLayout(content_layout, stretch=1)
        
        # 应用暗色主题
        self.apply_dark_theme()
    
    def apply_dark_theme(self):
        """应用暗色主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QPushButton {
                background-color: #3b3b3b;
                color: #fff;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #4b4b4b;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
            }
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #666;
            }
        """)
    
    def load_image(self):
        """加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择火球图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)"
        )
        
        if file_path:
            if self.image_label.load_image(file_path):
                self.current_image_path = file_path
                self.path_label.setText(f"图片: {os.path.basename(file_path)}")
                self.path_label.setStyleSheet("color: #4ade80; padding: 5px;")
                self.clear_btn.setEnabled(True)
                # 清除之前的标注
                self.image_label.annotated_point = None
                self.update_info_labels()
            else:
                QMessageBox.warning(self, "错误", "无法加载图片文件！")
    
    def clear_annotation(self):
        """清除标注"""
        self.image_label.annotated_point = None
        self.image_label.update()
        self.update_info_labels()
    
    def on_point_annotated(self, point):
        """标注点改变时的回调"""
        self.update_info_labels()
    
    def update_info_labels(self):
        """更新信息标签"""
        if self.image_label.annotated_point:
            point = self.image_label.annotated_point
            ref_point = self.image_label.reference_point
            
            # 更新坐标信息
            point_x = point.x()
            point_y = point.y()
            ref_x = ref_point.x()
            ref_y = ref_point.y()
            
            self.coord_label.setText(f"坐标: ({point_x}, {point_y})")
            
            # 计算距离（欧几里得距离）
            dx = point_x - ref_x
            dy = point_y - ref_y
            distance = math.sqrt(dx * dx + dy * dy)
            
            # 显示距离信息，保留2位小数
            self.distance_label.setText(
                f"距离参考点({ref_x}, {ref_y}): {distance:.2f} 像素"
            )
            
            # 调试输出
            print(f"标注点: ({point_x}, {point_y}), 参考点: ({ref_x}, {ref_y}), 距离: {distance:.2f}")
        else:
            self.coord_label.setText("坐标: 未标注")
            self.distance_label.setText("距离参考点(574, 542): 未标注")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("火球图片点标注工具")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = FireballPointAnnotator()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

