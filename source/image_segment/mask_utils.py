#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码处理工具模块
包含掩码质量评估、选择和分析功能
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

try:
    from .prompt_generation import create_prompt_generator
except ImportError:
    from prompt_generation import create_prompt_generator


class MaskValidator:
    """掩码验证器（简化版）"""
    
    def __init__(self):
        """初始化验证器"""
        pass
    
    def calculate_mask_area(self, mask: np.ndarray) -> int:
        """计算掩码面积"""
        if mask is None:
            return 0
        return int(np.sum(mask))
    
    def calculate_mask_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        计算掩码的质心坐标
        
        Args:
            mask: 输入掩码 (二值图像)
            
        Returns:
            Tuple[float, float]: 质心坐标 (x, y)，如果掩码为空返回 (0, 0)
        """
        if mask is None or np.sum(mask) == 0:
            return (0.0, 0.0)
        
        # 使用cv2.moments计算质心
        moments = cv2.moments(mask.astype(np.uint8))
        
        if moments['m00'] == 0:
            return (0.0, 0.0)
        
        # 计算质心坐标
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        return (float(cx), float(cy))
    
    def calculate_max_radius_with_point(self, mask: np.ndarray, centroid: Optional[Tuple[float, float]] = None) -> Tuple[float, Tuple[float, float]]:
        """
        计算从质心到掩码边界的最大半径，并返回最大半径对应的点坐标
        
        Args:
            mask: 输入掩码 (二值图像)
            centroid: 质心坐标，如果为None则自动计算
            
        Returns:
            Tuple[float, Tuple[float, float]]: (最大半径, 最大半径对应的点坐标)
        """
        if mask is None or np.sum(mask) == 0:
            return 0.0, (0.0, 0.0)
        
        # 获取质心坐标
        if centroid is None:
            centroid = self.calculate_mask_centroid(mask)
        
        cx, cy = centroid
        
        # 找到所有掩码像素的坐标
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return 0.0, (0.0, 0.0)
        
        # 计算从质心到所有掩码像素的距离
        distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        
        # 找到最大距离的索引
        max_distance_idx = np.argmax(distances)
        
        # 返回最大距离和对应的点坐标
        max_radius = float(distances[max_distance_idx])
        max_radius_point = (float(x_coords[max_distance_idx]), float(y_coords[max_distance_idx]))
        
        return max_radius, max_radius_point
    
    def analyze_mask_geometry(self, mask: np.ndarray, target_centre: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        分析掩码的几何特征
        
        Args:
            mask: 输入掩码
            target_centre: 目标质心坐标，如果提供则直接使用，否则自动计算
            
        Returns:
            Dict: 包含几何特征的字典，包括最大半径对应的点坐标
        """
        if mask is None or np.sum(mask) == 0:
            return {
                'area': 0,
                'centroid': (0.0, 0.0),
                'max_radius': 0.0,
                'max_radius_point': (0.0, 0.0),
                'bounding_box': (0, 0, 0, 0),
                'aspect_ratio': 0.0
            }
        
        # 基本信息
        area = self.calculate_mask_area(mask)
        
        # 如果提供了目标质心，直接使用；否则计算质心
        if target_centre is not None:
            centroid = target_centre
        else:
            centroid = self.calculate_mask_centroid(mask)
            
        # 计算最大半径和对应的点坐标
        max_radius, max_radius_point = self.calculate_max_radius_with_point(mask, centroid)
        
        # 边界框
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
            min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
            bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # 长宽比
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            aspect_ratio = float(min(width, height) / max(width, height)) if max(width, height) > 0 else 0.0
        else:
            bounding_box = (0, 0, 0, 0)
            aspect_ratio = 0.0
        
        return {
            'area': area,
            'centroid': centroid,
            'max_radius': max_radius,
            'max_radius_point': max_radius_point,
            'bounding_box': bounding_box,
            'aspect_ratio': aspect_ratio
        }
    
    def validate_mask_quality(self, mask: np.ndarray, min_area_ratio: float = 0.01, max_area_ratio: float = 0.9) -> bool:
        """
        验证掩码质量是否符合基本要求
        
        Args:
            mask: 输入掩码
            min_area_ratio: 最小面积比例
            max_area_ratio: 最大面积比例
            
        Returns:
            bool: 是否通过验证
        """
        if mask is None:
            return False
        
        # 检查掩码是否为空
        area = self.calculate_mask_area(mask)
        if area == 0:
            return False
        
        # 检查掩码面积是否合理
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = area / total_area
        
        # 面积应该在合理范围内
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False
        
        return True
    
    def compare_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
        """
        比较两个掩码的相似性和差异
        
        Args:
            mask1: 第一个掩码
            mask2: 第二个掩码
            
        Returns:
            Dict: 包含各种比较指标的字典
        """
        if mask1 is None or mask2 is None:
            return {"area_ratio": 0.0, "iou": 0.0, "dice": 0.0}
        
        area1 = self.calculate_mask_area(mask1)
        area2 = self.calculate_mask_area(mask2)
        area_ratio = area2 / area1 if area1 > 0 else 0.0
        
        # 计算交并比 (IoU)
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        iou = intersection / union if union > 0 else 0.0
        
        # 计算Dice系数
        dice = 2 * intersection / (area1 + area2) if (area1 + area2) > 0 else 0.0
        
        return {
            "area_ratio": area_ratio,
            "iou": iou,
            "dice": dice
        }


# 已迁移：请从 failure_analyzer 导入


def create_mask_validator() -> MaskValidator:
    """创建掩码验证器的便捷函数"""
    return MaskValidator()




# 向后兼容的别名
def create_mask_analyzer() -> MaskValidator:
    """向后兼容：创建掩码验证器（原掩码分析器的简化版）"""
    return MaskValidator()
