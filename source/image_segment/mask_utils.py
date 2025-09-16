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


class MaskAnalyzer:
    """掩码分析器"""
    
    def __init__(self):
        """初始化掩码分析器"""
        self.prompt_generator = create_prompt_generator()
    
    def calculate_mask_area(self, mask: np.ndarray) -> int:
        """计算掩码面积"""
        if mask is None:
            return 0
        return int(np.sum(mask))
    
    def calculate_mask_quality(self, mask: np.ndarray) -> float:
        """
        计算掩码质量分数
        
        Args:
            mask: 输入掩码
            
        Returns:
            float: 质量分数 (0-1)，越高越好
        """
        if mask is None:
            return 0.0
        
        # 基于面积和形状的质量分数
        area = self.calculate_mask_area(mask)
        if area == 0:
            return 0.0
        
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = area / total_area
        
        # 计算形状的紧凑性
        compactness = self._calculate_compactness(mask)
        
        # 综合质量分数
        quality = 0.6 * min(area_ratio * 10, 1.0) + 0.4 * compactness
        
        return quality
    
    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """计算掩码的紧凑性（形状规整度）"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0 and area > 0:
                # 紧凑性公式：4π*面积/周长²，圆形为1，其他形状<1
                compactness = 4 * np.pi * area / (perimeter ** 2)
                return min(compactness, 1.0)  # 限制最大值为1
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def validate_mask_quality(self, mask: np.ndarray, min_area_ratio: float = 0.01, max_area_ratio: float = 0.9) -> bool:
        """
        验证掩码质量是否符合要求
        
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
    
    def select_best_mask(self, masks: np.ndarray, strategy: str = "largest_area") -> np.ndarray:
        """
        从多个掩码中选择最佳的
        
        Args:
            masks: 掩码数组
            strategy: 选择策略 ("largest_area", "best_quality", "most_compact")
            
        Returns:
            np.ndarray: 最佳掩码
        """
        if len(masks) == 0:
            return None
        if len(masks) == 1:
            return masks[0]
        
        if strategy == "largest_area":
            areas = [self.calculate_mask_area(mask) for mask in masks]
            best_idx = np.argmax(areas)
        elif strategy == "best_quality":
            qualities = [self.calculate_mask_quality(mask) for mask in masks]
            best_idx = np.argmax(qualities)
        elif strategy == "most_compact":
            compactness_scores = [self._calculate_compactness(mask) for mask in masks]
            best_idx = np.argmax(compactness_scores)
        else:
            # 默认使用最大面积
            areas = [self.calculate_mask_area(mask) for mask in masks]
            best_idx = np.argmax(areas)
        
        return masks[best_idx]
    
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
    
    def __init__(self, mask_analyzer: Optional[MaskAnalyzer] = None, prompt_generator=None):
        """
        初始化失败分析器
        
        Args:
            mask_analyzer: 掩码分析器实例
            prompt_generator: 点生成器实例
        """
        self.mask_analyzer = mask_analyzer or MaskAnalyzer()
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
            
            # 3. 检查参考掩码质量
            ref_quality = self.mask_analyzer.calculate_mask_quality(reference_mask)
            if ref_quality < 0.5:
                reasons.append(f"参考掩码质量不佳 (质量分数: {ref_quality:.3f})")
            
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
        if not self.mask_analyzer.validate_mask_quality(result_mask):
            quality_score = self.mask_analyzer.calculate_mask_quality(result_mask)
            reasons.append(f"掩码质量不佳 (质量分数: {quality_score:.3f})")
        
        # 检查掩码面积比例
        result_area = self.mask_analyzer.calculate_mask_area(result_mask)
        ref_area = self.mask_analyzer.calculate_mask_area(reference_mask)
        
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


def create_mask_analyzer() -> MaskAnalyzer:
    """创建掩码分析器的便捷函数"""
    return MaskAnalyzer()


def create_failure_analyzer(mask_analyzer: Optional[MaskAnalyzer] = None, prompt_generator=None) -> PropagationFailureAnalyzer:
    """创建失败分析器的便捷函数"""
    return PropagationFailureAnalyzer(mask_analyzer, prompt_generator)
