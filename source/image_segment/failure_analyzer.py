#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码传播失败分析器
从 mask_utils.py 中拆分到单独文件，便于复用与维护
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

from .mask_utils import MaskValidator
from .prompt_generation import create_prompt_generator


class PropagationFailureAnalyzer:
    """传播失败分析器"""
    
    def __init__(self, mask_validator: Optional[MaskValidator] = None, prompt_generator=None):
        self.mask_validator = mask_validator or MaskValidator()
        self.prompt_generator = prompt_generator or create_prompt_generator()
    
    def analyze_failure(self, target_image_path: str, reference_image_path: str, 
                       reference_mask: np.ndarray, result_mask: Optional[np.ndarray]) -> str:
        """分析掩码传播失败的原因"""
        try:
            reasons: List[str] = []
            
            # 1) 结果掩码质量
            reasons.extend(self._analyze_result_mask(result_mask, reference_mask))
            
            # 2) 图像差异与点生成
            reasons.extend(self._analyze_image_differences(target_image_path, reference_image_path, reference_mask))
            
            # 3) 参考掩码基本有效性
            if not self.mask_validator.validate_mask_quality(reference_mask):
                reasons.append("参考掩码质量不佳（面积比例不合理）")
            
            return "; ".join(reasons) if reasons else "未知原因"
        except Exception as e:
            return f"分析过程出错: {str(e)}"
    
    def _analyze_result_mask(self, result_mask: Optional[np.ndarray], reference_mask: np.ndarray) -> List[str]:
        reasons: List[str] = []
        if result_mask is None:
            reasons.append("结果掩码为空")
            return reasons
        if not self.mask_validator.validate_mask_quality(result_mask):
            reasons.append("掩码质量不佳（面积比例不合理）")
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
        reasons: List[str] = []
        try:
            target_image = cv2.imread(target_image_path)
            reference_image = cv2.imread(reference_image_path)
            if target_image is None or reference_image is None:
                reasons.append("无法读取图像文件")
                return reasons
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            diff = cv2.absdiff(target_rgb, reference_rgb)
            mean_diff = float(np.mean(diff) / 255.0)
            if mean_diff > 0.5:
                reasons.append(f"图像差异过大 (差异度: {mean_diff:.3f})")
            points, labels = self.prompt_generator.generate_points_with_rgb_similarity(
                reference_rgb, reference_mask, target_rgb
            )
            positive_points = [p for p, l in zip(points, labels) if l == 1]
            negative_points = [p for p, l in zip(points, labels) if l == 0]
            reasons.extend(self._analyze_point_generation(
                positive_points, negative_points, reference_rgb, target_rgb, reference_mask
            ))
        except Exception as e:
            reasons.append(f"图像分析失败: {str(e)}")
        return reasons
    
    def _analyze_point_generation(self, positive_points: List[Tuple[int, int]], 
                                 negative_points: List[Tuple[int, int]],
                                 reference_rgb: np.ndarray, target_rgb: np.ndarray,
                                 reference_mask: np.ndarray) -> List[str]:
        reasons: List[str] = []
        if len(positive_points) == 0:
            reasons.append("未生成有效正点")
        elif len(positive_points) < 3:
            reasons.append(f"正点数量过少 ({len(positive_points)}个)")
        if len(negative_points) == 0:
            reasons.append("未生成负点")
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



def create_failure_analyzer(mask_validator: Optional[MaskValidator] = None, prompt_generator=None) -> PropagationFailureAnalyzer:
    """创建失败分析器的便捷函数"""
    return PropagationFailureAnalyzer(mask_validator, prompt_generator)

