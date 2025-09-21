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


class PropagationFailureAnalyzer:
    """传播失败分析器"""
    
    def __init__(self, mask_validator: Optional[MaskValidator] = None, prompt_generator=None):
        """
        初始化失败分析器
        
        Args:
            mask_validator: 掩码验证器实例
            prompt_generator: 点生成器实例
        """
        self.mask_validator = mask_validator or MaskValidator()
        self.prompt_generator = prompt_generator or create_prompt_generator()
    
    def analyze_failure(self, target_image_path: str, reference_image_path: str, 
                       reference_mask: np.ndarray, result_mask: Optional[np.ndarray]) -> str:
        """
        分析掩码传播失败的原因
        
        Args:
            target_image_path: 目标图片路径
            reference_image_path: 参考图片路径
            reference_mask: 参考掩码
            result_mask: 传播结果掩码（可能为None）
            
        Returns:
            str: 失败原因描述
        """
        try:
            reasons = []
            
            # 1. 检查结果掩码
            result_reasons = self._analyze_result_mask(result_mask, reference_mask)
            reasons.extend(result_reasons)
            
            # 2. 检查图像差异和点生成
            image_reasons = self._analyze_image_differences(
                target_image_path, reference_image_path, reference_mask
            )
            reasons.extend(image_reasons)
            
            # 3. 检查参考掩码基本有效性
            if not self.mask_validator.validate_mask_quality(reference_mask):
                reasons.append("参考掩码质量不佳（面积比例不合理）")
            
            # 返回失败原因
            return "; ".join(reasons) if reasons else "未知原因"
                
        except Exception as e:
            return f"分析过程出错: {str(e)}"
    
    def _analyze_result_mask(self, result_mask: Optional[np.ndarray], reference_mask: np.ndarray) -> List[str]:
        """分析结果掩码的问题"""
        reasons = []
        
        if result_mask is None:
            reasons.append("结果掩码为空")
            return reasons
        
        # 检查掩码质量
        if not self.mask_validator.validate_mask_quality(result_mask):
            reasons.append("掩码质量不佳（面积比例不合理）")
        
        # 检查掩码面积比例
        result_area = self.mask_validator.calculate_mask_area(result_mask)
        ref_area = self.mask_validator.calculate_mask_area(reference_mask)
        
        if ref_area > 0:
            area_ratio = result_area / ref_area
            if area_ratio < 0.1:
                reasons.append(f"掩码面积过小 (面积比: {area_ratio:.3f})")
            elif area_ratio > 5.0:
                reasons.append(f"掩码面积过大 (面积比: {area_ratio:.3f})")
        
        return reasons
    
    def _analyze_image_differences(self, target_image_path: str, reference_image_path: str, 
                                  reference_mask: np.ndarray) -> List[str]:
        """分析图像差异相关的问题"""
        reasons = []
        
        try:
            # 读取图像
            target_image = cv2.imread(target_image_path)
            reference_image = cv2.imread(reference_image_path)
            
            if target_image is None or reference_image is None:
                reasons.append("无法读取图像文件")
                return reasons
            
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            
            # 计算图像差异
            diff = cv2.absdiff(target_rgb, reference_rgb)
            mean_diff = np.mean(diff) / 255.0
            
            if mean_diff > 0.5:
                reasons.append(f"图像差异过大 (差异度: {mean_diff:.3f})")
            
            # 检查点生成情况
            points, labels = self.prompt_generator.generate_points_with_rgb_similarity(
                reference_rgb, reference_mask, target_rgb
            )
            
            positive_points = [p for p, l in zip(points, labels) if l == 1]
            negative_points = [p for p, l in zip(points, labels) if l == 0]
            
            # 分析点生成问题
            point_reasons = self._analyze_point_generation(
                positive_points, negative_points, reference_rgb, target_rgb, reference_mask
            )
            reasons.extend(point_reasons)
            
        except Exception as e:
            reasons.append(f"图像分析失败: {str(e)}")
        
        return reasons
    
    def _analyze_point_generation(self, positive_points: List[Tuple[int, int]], 
                                 negative_points: List[Tuple[int, int]],
                                 reference_rgb: np.ndarray, target_rgb: np.ndarray,
                                 reference_mask: np.ndarray) -> List[str]:
        """分析点生成相关的问题"""
        reasons = []
        
        # 检查正点数量
        if len(positive_points) == 0:
            reasons.append("未生成有效正点")
        elif len(positive_points) < 3:
            reasons.append(f"正点数量过少 ({len(positive_points)}个)")
        
        # 检查负点数量
        if len(negative_points) == 0:
            reasons.append("未生成负点")
        
        # 检查RGB相似性
        if len(positive_points) > 0:
            try:
                ref_positive_rgbs = [reference_rgb[y, x] for x, y in 
                                   self.prompt_generator.sample_points_from_mask(reference_mask, 10, True)]
                
                similar_count = 0
                for x, y in positive_points:
                    target_rgb_point = target_rgb[y, x]
                    for ref_rgb in ref_positive_rgbs:
                        if self.prompt_generator.is_rgb_very_similar(target_rgb_point, ref_rgb):
                            similar_count += 1
                            break
                
                similarity_ratio = similar_count / len(positive_points)
                if similarity_ratio < 0.3:
                    reasons.append(f"RGB相似性过低 (相似度: {similarity_ratio:.3f})")
            except Exception:
                reasons.append("RGB相似性分析失败")
        
        return reasons


def create_mask_validator() -> MaskValidator:
    """创建掩码验证器的便捷函数"""
    return MaskValidator()


def create_failure_analyzer(mask_validator: Optional[MaskValidator] = None, prompt_generator=None) -> PropagationFailureAnalyzer:
    """创建失败分析器的便捷函数"""
    return PropagationFailureAnalyzer(mask_validator, prompt_generator)


# 向后兼容的别名
def create_mask_analyzer() -> MaskValidator:
    """向后兼容：创建掩码验证器（原掩码分析器的简化版）"""
    return MaskValidator()
