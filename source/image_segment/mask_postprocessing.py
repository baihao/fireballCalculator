#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码后处理模块
基于两个最大连通域的面积和曲率粗糙度评分进行掩码选择
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class ContourBasedMaskProcessor:
    """基于轮廓特征的掩码处理器"""
    
    def __init__(self):
        """初始化处理器"""
        pass
    
    def filter_mask_by_contour_quality(self, mask: np.ndarray, method: str = "dual_connected_components") -> np.ndarray:
        """
        基于轮廓质量过滤掩码
        
        Args:
            mask: 输入掩码
            method: 过滤方法
                - "dual_connected_components": 基于两个最大连通域评分选择
                
        Returns:
            np.ndarray: 过滤后的掩码
        """
        if mask is None:
            return None
        
        if method == "dual_connected_components":
            # 兼容老接口：仅返回mask
            result = self.filter_by_dual_connected_components_with_details(mask)
            return result["mask"] if isinstance(result, dict) else result
        else:
            raise ValueError(f"未知的过滤方法: {method}")
    
    def filter_by_dual_connected_components_with_details(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        基于两个最大连通域的面积和曲率粗糙度评分选择最佳掩码，并返回细节结果。
        
        Args:
            mask: 输入掩码
            
        Returns:
            Dict[str, Any]: { 'mask', 'area', 'centroid': (x, y), 'contour', 'scores': {area, curvature, total} }
        """
        if mask is None:
            return {"mask": None, "area": 0.0, "centroid": (0.0, 0.0), "contour": None, "scores": {}}
        
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 找到连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        if num_labels <= 1:  # 只有背景
            return {"mask": np.zeros_like(mask), "area": 0.0, "centroid": (0.0, 0.0), "contour": None, "scores": {}}
        
        print(f"    发现{num_labels-1}个连通域，分析前两个最大连通域...")
        
        # 获取连通域面积（排除背景，索引0是背景）
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]  # 按面积降序排列
        
        # 取前两个最大的连通域
        top_two_indices = sorted_indices[:2]
        # 最大面积用于归一化（最大者得分=1）
        max_area = float(areas[top_two_indices[0]]) if len(top_two_indices) > 0 else 0.0
        candidates = []
        
        for i, idx in enumerate(top_two_indices):
            # 创建单个连通域的掩码
            component_mask = (labels == idx + 1).astype(np.uint8)
            
            # 计算面积
            area = float(areas[idx])
            
            # 预先计算该连通域的主轮廓与曲率粗糙度（避免二次计算）
            try:
                comp_contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(comp_contours, key=cv2.contourArea) if comp_contours else None
            except Exception:
                largest_contour = None

            curvature_roughness = (
                self.calculate_curvature_roughness(largest_contour) if largest_contour is not None else float('inf')
            )
            
            # 计算评分
            area_score = self.calculate_area_score(area, max_area)
            curvature_score = self.calculate_curvature_score(curvature_roughness)
            
            # 综合得分：0.7 * 面积 + 0.3 * 曲率粗糙度
            total_score = 0.7 * area_score + 0.3 * curvature_score
            
            candidates.append({
                'mask': component_mask,
                'area': area,
                'curvature_roughness': curvature_roughness,
                'area_score': area_score,
                'curvature_score': curvature_score,
                'total_score': total_score,
                'index': i + 1,
                'label': int(idx + 1),
                'centroid': (float(centroids[idx + 1][0]), float(centroids[idx + 1][1])),
                'contour': largest_contour
            })
            
            print(f"      连通域{i+1}: 面积={area}, 曲率粗糙度={curvature_roughness:.3f}")
            print(f"        面积得分={area_score:.3f}, 曲率得分={curvature_score:.3f}, 总分={total_score:.3f}")
        
        # 选择得分最高的连通域
        best_candidate = max(candidates, key=lambda x: x['total_score'])
        
        print(f"    选择连通域{best_candidate['index']}: 总分={best_candidate['total_score']:.3f}")
        
        # 直接使用候选阶段已计算的轮廓，避免重复计算
        best_contour = best_candidate.get('contour', None)

        return {
            "mask": best_candidate['mask'],
            "area": best_candidate['area'],
            "centroid": best_candidate['centroid'],
            "contour": best_contour,
            "scores": {
                "area": best_candidate['area_score'],
                "curvature": best_candidate['curvature_score'],
                "total": best_candidate['total_score']
            }
        }

    # 旧名以兼容可能的调用方
    def filter_by_dual_connected_components(self, mask: np.ndarray) -> np.ndarray:
        result = self.filter_by_dual_connected_components_with_details(mask)
        return result["mask"]
    
    def calculate_area_score(self, area: float, max_area: float) -> float:
        """
        计算面积得分（满分1分）
        
        Args:
            area: 连通域面积
            max_area: 最大连通域面积（用于归一化，最大者得分=1）
            
        Returns:
            float: 面积得分 (0-1)
        """
        if area <= 0 or max_area <= 0:
            return 0.0
        # 最大面积归一化：最大者1，按比例递减
        return float(min(1.0, max(0.0, area / max_area)))
    
    def calculate_curvature_score(self, curvature_roughness: float) -> float:
        """
        计算曲率粗糙度得分（满分1分）
        曲率粗糙度越小（越平滑）得分越高
        
        Args:
            curvature_roughness: 曲率粗糙度
            
        Returns:
            float: 曲率得分 (0-1)
        """
        if curvature_roughness == float('inf'):
            return 0.0
        
        # 将粗糙度夹在 [1.0, 3.0] 区间
        r = float(curvature_roughness)
        if r < 1.0:
            r = 1.0
        elif r > 3.0:
            r = 3.0
        
        # 线性映射：r=1 -> 1.0 分，r=3 -> 0 分
        score = (3.0 - r) / 2.0
        # 保证在 [0,1]
        return float(min(1.0, max(0.0, score)))
    
    def calculate_curvature_roughness(self, contour: np.ndarray) -> float:
        """
        计算轮廓的曲率粗糙度
        
        Args:
            contour: 轮廓点集
            
        Returns:
            float: 曲率粗糙度
        """
        try:
            if len(contour) < 10:
                return 2.0  # 点太少，认为是粗糙的
            
            # 计算局部曲率变化
            points = contour.reshape(-1, 2)
            
            # 计算相邻点之间的角度变化
            angle_changes = []
            window_size = min(5, len(points) // 10)  # 自适应窗口大小
            
            for i in range(len(points)):
                # 获取前后窗口的点
                prev_idx = (i - window_size) % len(points)
                next_idx = (i + window_size) % len(points)
                
                # 计算向量
                v1 = points[i] - points[prev_idx]
                v2 = points[next_idx] - points[i]
                
                # 计算角度变化
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.arccos(cos_angle)
                    angle_changes.append(angle_change)
            
            if not angle_changes:
                return 2.0
            
            # 计算角度变化的标准差（平滑边界变化小）
            angle_std = np.std(angle_changes)
            
            # 曲率粗糙度：标准差越大越粗糙
            curvature_roughness = 1.0 + angle_std * 2  # 基础值1，标准差越大越粗糙
            
            return curvature_roughness
            
        except Exception:
            return 2.0


def create_contour_processor() -> ContourBasedMaskProcessor:
    """创建基于轮廓的掩码处理器的便捷函数"""
    return ContourBasedMaskProcessor()


# 向后兼容的函数接口
def filter_mask_contours(mask: np.ndarray, method: str = "dual_connected_components") -> np.ndarray:
    """
    便捷函数：基于轮廓特征过滤掩码
    
    Args:
        mask: 输入掩码
        method: 过滤方法
        
    Returns:
        np.ndarray: 过滤后的掩码
    """
    processor = create_contour_processor()
    return processor.filter_mask_by_contour_quality(mask, method)