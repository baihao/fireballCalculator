#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像绘制工具函数

提供在图像上绘制火球分割结果（轮廓、质心、最大半径箭头）的通用函数，
供 interactive_image_widget.py 与 sequence_image_composer.py 共享使用。
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2


def draw_segmentation_on_image(image: np.ndarray,
                               segmentation_result: Dict[str, Any],
                               inplace: bool = False) -> np.ndarray:
    """
    在图像上绘制分割结果（蓝色轮廓、绿色箭头、紫色质心点）。

    Args:
        image: 原始图像数组（RGB格式）
        segmentation_result: 分割结果字典，包含 contours、centroid、max_radius 等信息
        inplace: 是否在原图上就地绘制；False 时会复制一份再绘制

    Returns:
        np.ndarray: 绘制后的图像数组（RGB格式）
    """
    try:
        target = image if inplace else image.copy()

        if segmentation_result is None or not segmentation_result.get("success", False):
            return target

        # 1) 绘制蓝色轮廓
        contours = segmentation_result.get("contours", [])
        if contours:
            for contour_points in contours:
                if len(contour_points) > 2:
                    contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
                    # RGB: 蓝色 (0, 0, 255)
                    cv2.drawContours(target, [contour_array], -1, (0, 0, 255), 3)

        # 2) 绘制绿色质心到最大半径端点的箭头与标记
        centroid_data = segmentation_result.get("centroid", {})
        max_radius_data = segmentation_result.get("max_radius", {})

        if centroid_data and max_radius_data:
            cx = int(centroid_data.get("x", 0))
            cy = int(centroid_data.get("y", 0))

            endpoint_data = max_radius_data.get("endpoint", {})
            ex = int(endpoint_data.get("x", 0))
            ey = int(endpoint_data.get("y", 0))

            h, w = target.shape[:2]
            if 0 <= cx < w and 0 <= cy < h and 0 <= ex < w and 0 <= ey < h:
                # RGB: 绿色 (0, 255, 0)
                cv2.arrowedLine(target, (cx, cy), (ex, ey), (0, 255, 0), 3, tipLength=0.1)
                # RGB: 紫色 (128, 0, 128)
                cv2.circle(target, (cx, cy), 5, (128, 0, 128), -1)
                # 端点小圆圈（绿色）
                cv2.circle(target, (ex, ey), 3, (0, 255, 0), 2)

        return target

    except Exception as e:
        print(f"❌ 绘制分割结果失败: {e}")
        return image if inplace else image.copy()


def draw_cross(image: np.ndarray,
               x: int,
               y: int,
               color: Tuple[int, int, int],
               size: int = 12,
               thickness: int = 3) -> None:
    """
    在图像上绘制十字标记（RGB 颜色）。
    """
    try:
        h, w = image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # 水平线
            start_x = max(0, x - size // 2)
            end_x = min(w - 1, x + size // 2)
            cv2.line(image, (start_x, y), (end_x, y), color, thickness)
            # 垂直线
            start_y = max(0, y - size // 2)
            end_y = min(h - 1, y + size // 2)
            cv2.line(image, (x, start_y), (x, end_y), color, thickness)
    except Exception as e:
        print(f"❌ 绘制十字失败: {e}")


def draw_prompt_points_on_image(image: np.ndarray,
                                positive_points: List[Tuple[int, int]],
                                negative_points: List[Tuple[int, int]],
                                ignition_point: Optional[Tuple[int, int]] = None) -> None:
    """
    在图像上绘制参考点（正/负点和起爆点）。
    - 正点：红色十字 (255, 0, 0)，size=12, thickness=3
    - 负点：蓝色十字 (0, 0, 255)，size=12, thickness=3
    - 起爆点：紫色十字 (128, 0, 128)，size=15, thickness=4
    """
    try:
        # 绘制正点
        for x, y in positive_points or []:
            draw_cross(image, x, y, (255, 0, 0), size=12, thickness=3)
        # 绘制负点
        for x, y in negative_points or []:
            draw_cross(image, x, y, (0, 0, 255), size=12, thickness=3)
        # 起爆点
        if ignition_point is not None:
            x, y = ignition_point
            draw_cross(image, x, y, (128, 0, 128), size=15, thickness=4)
    except Exception as e:
        print(f"❌ 绘制参考点失败: {e}")


