#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码后处理模块
包含基于轮廓特征的掩码清理和优化功能，专门处理内部破碎的大面积碎片
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from scipy import ndimage


class ContourBasedMaskProcessor:
    """基于轮廓特征的掩码处理器"""
    
    def __init__(self):
        """初始化处理器"""
        pass
    
    def filter_mask_by_contour_quality(self, mask: np.ndarray, method: str = "best_contour") -> np.ndarray:
        """
        基于轮廓质量过滤掩码
        
        Args:
            mask: 输入掩码
            method: 过滤方法
                - "best_contour": 选择质量最高的单个轮廓
                - "quality_threshold": 基于质量阈值过滤多个轮廓
                - "fireball_optimized": 专门针对火球优化的过滤
                
        Returns:
            np.ndarray: 过滤后的掩码
        """
        if mask is None:
            return None
        
        if method == "best_contour":
            return self.select_best_contour(mask)
        elif method == "quality_threshold":
            return self.filter_by_quality_threshold(mask)
        elif method == "fireball_optimized":
            return self.filter_for_fireball(mask)
        else:
            raise ValueError(f"未知的过滤方法: {method}")
    
    def select_best_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        选择质量最高的单个轮廓，去除所有碎片
        
        Args:
            mask: 输入掩码
            
        Returns:
            np.ndarray: 只包含最佳轮廓的掩码
        """
        if mask is None:
            return None
        
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(mask)
        
        print(f"    分析{len(contours)}个轮廓的质量...")
        
        best_contour = None
        best_score = 0
        best_info = None
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 过滤过小的轮廓
            if area < 50:
                continue
            
            # 计算轮廓质量
            quality_info = self.analyze_contour_quality(contour)
            quality_score = quality_info['overall_score']
            
            print(f"      轮廓{i}: 面积={int(area)}, 质量={quality_score:.3f}, 紧凑性={quality_info['compactness']:.3f}")
            
            if quality_score > best_score:
                best_score = quality_score
                best_contour = contour
                best_info = quality_info
        
        if best_contour is None:
            return np.zeros_like(mask)
        
        print(f"    选择最佳轮廓: 质量分数={best_score:.3f}")
        
        # 重建只包含最佳轮廓的mask
        new_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(new_mask, [best_contour], 1)
        
        return new_mask
    
    def filter_for_fireball(self, mask: np.ndarray) -> np.ndarray:
        """
        专门针对火球的轮廓过滤（优化版）
        火球特征：边界相对平滑、内部空洞少、可能有凹陷但整体连续
        
        Args:
            mask: 输入掩码
            
        Returns:
            np.ndarray: 过滤后的掩码
        """
        if mask is None:
            return None
        
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(mask)
        
        print(f"    火球优化过滤: 分析{len(contours)}个轮廓...")
        
        fireball_candidates = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 火球面积筛选（降低最小面积要求）
            if area < 50:  # 降低阈值，适应不同大小的火球
                continue
            
            # 计算火球相关指标
            quality_info = self.analyze_contour_quality(contour)
            
            # 火球特征评分
            fireball_score = self.calculate_fireball_score(quality_info)
            
            print(f"      轮廓{i}: 面积={int(area)}, 火球分数={fireball_score:.3f}")
            print(f"        平滑度={1/quality_info['roughness']:.3f}, 空洞密度={quality_info['hole_density']:.3f}, 凸包比={quality_info['convexity']:.3f}")
            
            # 降低火球质量阈值，适应有凹陷的火球
            if fireball_score > 0.3:  # 降低阈值从0.4到0.3
                fireball_candidates.append({
                    'contour': contour,
                    'area': area,
                    'fireball_score': fireball_score,
                    'quality_info': quality_info,
                    'index': i
                })
        
        if not fireball_candidates:
            print("    未找到符合火球特征的轮廓，使用面积最大且相对完整的轮廓")
            # 回退策略：选择面积最大且空洞密度不太高的轮廓
            best_contour = self.select_best_fallback_contour(contours)
            new_mask = np.zeros_like(binary_mask)
            cv2.fillPoly(new_mask, [best_contour], 1)
            return new_mask
        
        # 选择火球分数最高的轮廓
        best_candidate = max(fireball_candidates, key=lambda x: x['fireball_score'])
        
        print(f"    选择火球轮廓{best_candidate['index']}: 分数={best_candidate['fireball_score']:.3f}")
        
        new_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(new_mask, [best_candidate['contour']], 1)
        
        return new_mask
    
    def select_best_fallback_contour(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        回退策略：当没有高质量轮廓时，选择最佳的备选轮廓
        
        Args:
            contours: 轮廓列表
            
        Returns:
            np.ndarray: 最佳备选轮廓
        """
        if not contours:
            return None
        
        best_contour = None
        best_score = -1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # 过滤过小轮廓
                continue
            
            # 简单评分：面积权重 + 空洞密度权重
            hole_density = self.calculate_contour_hole_density(contour, area)
            
            # 面积归一化分数
            area_score = min(1.0, area / 1000)
            hole_score = max(0, 1 - hole_density)
            
            fallback_score = area_score * 0.7 + hole_score * 0.3
            
            if fallback_score > best_score:
                best_score = fallback_score
                best_contour = contour
        
        return best_contour or contours[0]
    
    def analyze_contour_quality(self, contour: np.ndarray) -> Dict[str, float]:
        """
        分析单个轮廓的质量指标
        
        Args:
            contour: 轮廓点集
            
        Returns:
            Dict[str, float]: 质量指标字典
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return self._empty_quality_info()
        
        # 1. 紧凑性 (圆形=1, 破碎形状<0.5)
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # 2. 凸包比率 (完整形状接近1)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # 3. 长宽比 (圆形接近1)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        
        # 4. 填充度 (相对于外接矩形)
        rect_area = w * h
        fill_ratio = area / rect_area if rect_area > 0 else 0
        
        # 5. 空洞密度分析
        hole_density = self.calculate_contour_hole_density(contour, area)
        
        # 6. 边界粗糙度
        roughness = self.calculate_boundary_roughness(contour, area)
        
        # 综合质量分数
        overall_score = (
            compactness * 0.25 +      # 紧凑性
            convexity * 0.25 +        # 凸性
            aspect_ratio * 0.15 +     # 长宽比
            fill_ratio * 0.10 +       # 填充度
            (1 - hole_density) * 0.15 + # 空洞密度（越少越好）
            max(0, 2 - roughness) * 0.10  # 边界平滑度
        )
        
        return {
            'area': area,
            'compactness': compactness,
            'convexity': convexity,
            'aspect_ratio': aspect_ratio,
            'fill_ratio': fill_ratio,
            'hole_density': hole_density,
            'roughness': roughness,
            'overall_score': overall_score
        }
    
    def calculate_contour_hole_density(self, contour: np.ndarray, contour_area: float) -> float:
        """
        计算轮廓内部的空洞密度
        
        Args:
            contour: 轮廓点集
            contour_area: 轮廓面积
            
        Returns:
            float: 空洞密度 (0-1)，越小越好
        """
        try:
            # 创建轮廓mask
            x, y, w, h = cv2.boundingRect(contour)
            roi_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            
            # 将轮廓坐标转换为ROI坐标
            contour_roi = contour.copy()
            contour_roi[:, :, 0] -= x
            contour_roi[:, :, 1] -= y
            
            # 填充轮廓
            cv2.fillPoly(roi_mask[1:-1, 1:-1], [contour_roi], 1)
            
            # 计算填充后的面积
            filled_area = np.sum(roi_mask)
            
            # 空洞密度
            if filled_area > 0:
                hole_density = 1 - (contour_area / filled_area)
                return max(0, min(1, hole_density))
            else:
                return 1.0
                
        except Exception:
            return 0.5  # 默认中等密度
    
    def calculate_boundary_roughness(self, contour: np.ndarray, area: float) -> float:
        """
        计算边界粗糙度（优化版，适合火球特征）
        
        Args:
            contour: 轮廓点集
            area: 轮廓面积
            
        Returns:
            float: 粗糙度，平滑边界约为1，破碎边界>2
        """
        try:
            perimeter = cv2.arcLength(contour, True)
            
            if area <= 0 or perimeter <= 0:
                return float('inf')
            
            # 方法1: 相对于等面积圆的周长比
            ideal_perimeter = 2 * np.pi * np.sqrt(area / np.pi)
            perimeter_ratio = perimeter / ideal_perimeter
            
            # 方法2: 轮廓点密度分析（检测边界复杂度）
            # 简化轮廓，看简化前后的差异
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            simplification_ratio = len(simplified) / len(contour) if len(contour) > 0 else 1
            
            # 方法3: 局部曲率变化分析
            curvature_roughness = self.calculate_curvature_roughness(contour)
            
            # 综合粗糙度评估
            # 对于火球：边界可能不规则但应该相对平滑
            combined_roughness = (
                perimeter_ratio * 0.4 +           # 周长比
                (1 - simplification_ratio) * 0.3 + # 复杂度
                curvature_roughness * 0.3          # 曲率变化
            )
            
            return combined_roughness
                
        except Exception:
            return float('inf')
    
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
            mean_angle_change = np.mean(angle_changes)
            
            # 曲率粗糙度：标准差越大越粗糙
            curvature_roughness = 1.0 + angle_std * 2  # 基础值1，标准差越大越粗糙
            
            return curvature_roughness
            
        except Exception:
            return 2.0
    
    def calculate_fireball_score(self, quality_info: Dict[str, float]) -> float:
        """
        计算火球特征分数（优化版）
        火球特点：边界平滑、内部空洞少、可能有凹陷但整体连续
        
        Args:
            quality_info: 轮廓质量信息
            
        Returns:
            float: 火球特征分数 (0-1)，越高越像火球
        """
        # 提取指标
        compactness = quality_info.get('compactness', 0)
        convexity = quality_info.get('convexity', 0)
        aspect_ratio = quality_info.get('aspect_ratio', 0)
        hole_density = quality_info.get('hole_density', 1)
        roughness = quality_info.get('roughness', float('inf'))
        area = quality_info.get('area', 0)
        
        # 1. 边界平滑度评分（最重要）
        # 火球边界应该相对平滑，roughness在1.0-1.8之间为理想
        if roughness == float('inf'):
            smooth_score = 0
        elif roughness <= 1.2:
            smooth_score = 1.0  # 非常平滑
        elif roughness <= 1.8:
            smooth_score = 1.0 - (roughness - 1.2) / 0.6  # 线性衰减
        else:
            smooth_score = max(0, 0.5 - (roughness - 1.8) / 2.0)  # 快速衰减
        
        # 2. 内部完整性评分（重要）
        # 火球内部应该少空洞
        hole_score = max(0, 1 - hole_density * 3)  # 空洞密度惩罚加重
        
        # 3. 面积合理性评分
        # 火球应该有合理的面积，不会太小
        area_score = 1.0 if area > 500 else area / 500
        
        # 4. 形状连续性评分（降低凸性要求）
        # 火球可能有凹陷，所以降低凸性权重
        # 凸性0.6以上就认为是合理的
        convexity_score = min(1.0, convexity / 0.6) if convexity > 0.3 else 0
        
        # 5. 长宽比评分（保持）
        # 火球不应该过于细长
        aspect_score = aspect_ratio
        
        # 6. 基础紧凑性（降低要求）
        # 允许一定程度的不规则，但不能太破碎
        compactness_score = min(1.0, compactness / 0.4) if compactness > 0.2 else 0
        
        # 火球综合分数（调整权重）
        fireball_score = (
            smooth_score * 0.35 +       # 边界平滑度最重要
            hole_score * 0.25 +         # 内部完整性
            area_score * 0.15 +         # 面积合理性
            convexity_score * 0.10 +    # 凸性（降低权重）
            aspect_score * 0.10 +       # 长宽比
            compactness_score * 0.05    # 基础紧凑性（最低权重）
        )
        
        return fireball_score
    
    def filter_by_quality_threshold(self, mask: np.ndarray, quality_threshold: float = 0.5) -> np.ndarray:
        """
        基于质量阈值过滤轮廓，保留所有高质量轮廓
        
        Args:
            mask: 输入掩码
            quality_threshold: 质量阈值
            
        Returns:
            np.ndarray: 过滤后的掩码
        """
        if mask is None:
            return None
        
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(mask)
        
        # 分析所有轮廓
        valid_contours = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < 50:  # 过滤过小轮廓
                continue
            
            quality_info = self.analyze_contour_quality(contour)
            
            if quality_info['overall_score'] > quality_threshold:
                valid_contours.append(contour)
                print(f"      保留轮廓{i}: 面积={int(area)}, 质量={quality_info['overall_score']:.3f}")
        
        if not valid_contours:
            print("    未找到高质量轮廓，使用最大轮廓")
            largest_contour = max(contours, key=cv2.contourArea)
            valid_contours = [largest_contour]
        
        # 重建包含所有有效轮廓的mask
        new_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(new_mask, valid_contours, 1)
        
        print(f"    质量过滤: 保留{len(valid_contours)}/{len(contours)}个轮廓")
        
        return new_mask
    
    def analyze_contour_with_holes(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        分析包含内部空洞的轮廓
        
        Args:
            mask: 输入掩码
            
        Returns:
            Dict[str, Any]: 详细分析结果
        """
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 查找外部轮廓和内部空洞
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"external_contours": 0, "holes": 0, "hole_ratio": 0}
        
        external_contours = []
        holes = []
        
        # 分析层次结构
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # 外部轮廓
                    external_contours.append(contours[i])
                else:  # 内部空洞
                    holes.append(contours[i])
        
        # 计算空洞统计
        total_area = sum(cv2.contourArea(c) for c in external_contours)
        hole_area = sum(cv2.contourArea(c) for c in holes)
        hole_ratio = hole_area / total_area if total_area > 0 else 0
        
        return {
            "external_contours": len(external_contours),
            "holes": len(holes),
            "total_area": total_area,
            "hole_area": hole_area,
            "hole_ratio": hole_ratio
        }
    
    def remove_internal_holes(self, mask: np.ndarray, max_hole_area: int = 100) -> np.ndarray:
        """
        移除内部小空洞
        
        Args:
            mask: 输入掩码
            max_hole_area: 最大允许空洞面积
            
        Returns:
            np.ndarray: 填充空洞后的掩码
        """
        if mask is None:
            return None
        
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 使用scipy填充所有空洞
        filled = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
        
        # 计算被填充的区域
        holes = filled - binary_mask
        
        # 分析每个空洞
        hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_mask = binary_mask.copy()
        filled_holes = 0
        
        for hole_contour in hole_contours:
            hole_area = cv2.contourArea(hole_contour)
            if hole_area <= max_hole_area:
                cv2.fillPoly(result_mask, [hole_contour], 1)
                filled_holes += 1
        
        print(f"    空洞填充: 填充了{filled_holes}个小空洞（<{max_hole_area}像素）")
        
        return result_mask
    
    def _empty_quality_info(self) -> Dict[str, float]:
        """返回空的质量信息"""
        return {
            'area': 0,
            'compactness': 0,
            'convexity': 0,
            'aspect_ratio': 0,
            'fill_ratio': 0,
            'hole_density': 1,
            'roughness': float('inf'),
            'overall_score': 0
        }


def create_contour_processor() -> ContourBasedMaskProcessor:
    """创建基于轮廓的掩码处理器的便捷函数"""
    return ContourBasedMaskProcessor()


# 向后兼容的函数接口
def filter_mask_contours(mask: np.ndarray, method: str = "fireball_optimized") -> np.ndarray:
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


def analyze_mask_contours(mask: np.ndarray) -> Dict[str, Any]:
    """
    便捷函数：分析掩码的轮廓特征
    
    Args:
        mask: 输入掩码
        
    Returns:
        Dict[str, Any]: 轮廓分析结果
    """
    processor = create_contour_processor()
    return processor.analyze_contour_with_holes(mask)
