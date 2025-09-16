#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt点生成模块
负责从参考图片生成目标图片的prompt点，包括采样、映射和筛选功能
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class PromptPointGenerator:
    """Prompt点生成器"""
    
    def __init__(self, very_similar_threshold: float = 8.0, similar_threshold: float = 12.0):
        """
        初始化Prompt点生成器
        
        Args:
            very_similar_threshold: 非常相似的RGB距离阈值
            similar_threshold: 相似的RGB距离阈值
        """
        self.very_similar_threshold = very_similar_threshold
        self.similar_threshold = similar_threshold
    
    def generate_points_with_rgb_similarity(self, reference_image: np.ndarray, reference_mask: np.ndarray, 
                                          target_image: np.ndarray, return_debug_info: bool = False,
                                          predefined_reference_points: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        基于RGB相似性生成目标图片的正负点
        
        Args:
            reference_image: 参考图片 (RGB)
            reference_mask: 参考掩码
            target_image: 目标图片 (RGB)
            
        Returns:
            Tuple[List[Tuple[int, int]], List[int]]: (点坐标列表, 点标签列表)
        """
        try:
            # 1. 使用预定义的采样点或重新采样
            if predefined_reference_points is not None:
                positive_candidates = predefined_reference_points['positive']
                negative_candidates = predefined_reference_points['negative']
                print(f"    使用预定义的参考点: 正{len(positive_candidates)} 负{len(negative_candidates)}")
            else:
                # 从参考图片的掩码内部选取10个正点候选
                positive_candidates = self.sample_points_from_mask(reference_mask, num_points=10, inside_mask=True)
                
                # 从参考图片的掩码外部选取6个负点候选
                negative_candidates = self.sample_points_from_mask(reference_mask, num_points=6, inside_mask=False)
                print(f"    重新采样参考点: 正{len(positive_candidates)} 负{len(negative_candidates)}")
            
            # 3. 将候选点映射到目标图片上
            target_positive_candidates = self.map_points_to_target(reference_image, positive_candidates, target_image)
            target_negative_candidates = self.map_points_to_target(reference_image, negative_candidates, target_image)
            
            # 4. 计算参考图片正点的RGB值
            reference_positive_rgbs = [reference_image[y, x] for x, y in positive_candidates]
            
            # 5. 筛选目标图片的正点
            target_positive_points = self.filter_positive_points(
                target_positive_candidates, target_image, reference_positive_rgbs
            )
            
            # 6. 筛选目标图片的负点
            target_negative_points = self.filter_negative_points(
                target_negative_candidates, target_image, reference_positive_rgbs
            )
            
            # 7. 组合最终的点坐标和标签
            final_points = target_positive_points + target_negative_points
            final_labels = [1] * len(target_positive_points) + [0] * len(target_negative_points)
            
            print(f"    生成了 {len(target_positive_points)} 个正点和 {len(target_negative_points)} 个负点")
            
            if return_debug_info:
                return final_points, final_labels, {
                    'reference_positive': positive_candidates,
                    'reference_negative': negative_candidates,
                    'mapped_positive': target_positive_candidates,
                    'mapped_negative': target_negative_candidates,
                    'filtered_positive': target_positive_points,
                    'filtered_negative': target_negative_points
                }
            else:
                return final_points, final_labels
            
        except Exception as e:
            print(f"    ⚠️ 点生成失败: {e}")
            return [], []
    
    def sample_points_from_mask(self, mask: np.ndarray, num_points: int, inside_mask: bool) -> List[Tuple[int, int]]:
        """从掩码内部或外部采样点，优化分布策略"""
        h, w = mask.shape
        
        if inside_mask:
            # 正点采样：优先选择mask中心区域的点
            return self._sample_positive_points(mask, num_points)
        else:
            # 负点采样：选择远离mask的区域的点
            return self._sample_negative_points(mask, num_points)
    
    def _sample_positive_points(self, mask: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
        """采样正点：优先选择mask中心区域的点"""
        h, w = mask.shape
        
        # 计算距离变换，找到mask的中心区域
        mask_uint8 = mask.astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
        # 创建权重：距离mask边缘越远权重越高
        weights = dist_transform.copy()
        
        # 只考虑mask内部的点
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return []
        
        # 获取每个候选点的权重（距离边缘的距离）
        point_weights = weights[y_coords, x_coords]
        
        # 按权重排序，优先选择中心区域的点
        sorted_indices = np.argsort(point_weights)[::-1]  # 降序排列
        
        # 从高权重区域进行网格化采样
        candidates = np.stack([x_coords[sorted_indices], y_coords[sorted_indices]], axis=1)
        candidate_weights = point_weights[sorted_indices]
        
        # 选择前80%权重的点作为候选池（避免边缘点）
        top_ratio = 0.8
        top_count = max(num_points * 2, int(len(candidates) * top_ratio))
        top_candidates = candidates[:top_count]
        
        if len(top_candidates) == 0:
            return []
        
        # 在高权重区域进行均匀网格采样
        selected = self._grid_sample_from_candidates(top_candidates, num_points, w, h)
        
        return selected
    
    def _sample_negative_points(self, mask: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
        """采样负点：选择远离mask的区域的点"""
        h, w = mask.shape
        
        # 计算距离变换，找到远离mask的区域
        mask_uint8 = mask.astype(np.uint8)
        
        # 创建扩展的mask，排除mask边缘附近的区域
        edge_buffer = max(10, int(min(h, w) * 0.05))  # 动态缓冲区大小
        expanded_mask = cv2.dilate(mask_uint8, np.ones((edge_buffer, edge_buffer), np.uint8), iterations=1)
        
        # 计算到mask的距离
        dist_to_mask = cv2.distanceTransform((1 - expanded_mask).astype(np.uint8), cv2.DIST_L2, 5)
        
        # 只考虑不在扩展mask内的点
        y_coords, x_coords = np.where(expanded_mask == 0)
        
        if len(x_coords) < num_points:
            # 如果远离区域点不够，从图像边缘区域采样
            return self._sample_from_image_edges(mask, num_points, h, w)
        
        # 获取每个候选点到mask的距离
        point_distances = dist_to_mask[y_coords, x_coords]
        
        # 按距离排序，优先选择距离mask最远的点
        sorted_indices = np.argsort(point_distances)[::-1]  # 降序排列
        
        # 选择距离最远的候选点
        candidates = np.stack([x_coords[sorted_indices], y_coords[sorted_indices]], axis=1)
        
        # 选择前70%距离的点作为候选池
        top_ratio = 0.7
        top_count = max(num_points * 2, int(len(candidates) * top_ratio))
        top_candidates = candidates[:top_count]
        
        # 在远离区域进行均匀网格采样
        selected = self._grid_sample_from_candidates(top_candidates, num_points, w, h)
        
        return selected
    
    def _sample_from_image_edges(self, mask: np.ndarray, num_points: int, h: int, w: int) -> List[Tuple[int, int]]:
        """从图像边缘区域采样负点"""
        edge_width = max(5, min(h, w) // 20)  # 边缘区域宽度
        edge_points = []
        
        # 上边缘区域
        for y in range(edge_width):
            for x in range(0, w, max(1, w // (num_points * 2))):
                if mask[y, x] == 0:
                    edge_points.append((x, y))
        
        # 下边缘区域
        for y in range(h - edge_width, h):
            for x in range(0, w, max(1, w // (num_points * 2))):
                if mask[y, x] == 0:
                    edge_points.append((x, y))
        
        # 左边缘区域
        for x in range(edge_width):
            for y in range(0, h, max(1, h // (num_points * 2))):
                if mask[y, x] == 0:
                    edge_points.append((x, y))
        
        # 右边缘区域
        for x in range(w - edge_width, w):
            for y in range(0, h, max(1, h // (num_points * 2))):
                if mask[y, x] == 0:
                    edge_points.append((x, y))
        
        if not edge_points:
            return []
        
        # 从边缘点中均匀选择
        if len(edge_points) <= num_points:
            return edge_points
        else:
            # 均匀采样
            indices = np.linspace(0, len(edge_points) - 1, num_points, dtype=int)
            return [edge_points[i] for i in indices]
    
    def _grid_sample_from_candidates(self, candidates: np.ndarray, num_points: int, w: int, h: int) -> List[Tuple[int, int]]:
        """从候选点中进行网格化均匀采样"""
        if len(candidates) == 0:
            return []
        
        if len(candidates) <= num_points:
            return [(int(x), int(y)) for x, y in candidates]
        
        # 计算网格参数
        grid_rows = max(1, int(np.sqrt(num_points)))
        grid_cols = max(1, int(np.ceil(num_points / grid_rows)))
        
        cell_w = max(1, w // grid_cols)
        cell_h = max(1, h // grid_rows)
        
        selected = []
        used_idx = set()
        
        # 网格采样
        for r in range(grid_rows):
            if len(selected) >= num_points:
                break
            y0 = r * cell_h
            y1 = h if r == grid_rows - 1 else (r + 1) * cell_h
            
            for c in range(grid_cols):
                if len(selected) >= num_points:
                    break
                x0 = c * cell_w
                x1 = w if c == grid_cols - 1 else (c + 1) * cell_w
                
                # 找出落在该网格内的候选点
                in_cell = np.where(
                    (candidates[:, 0] >= x0) & (candidates[:, 0] < x1) &
                    (candidates[:, 1] >= y0) & (candidates[:, 1] < y1)
                )[0]
                
                if len(in_cell) == 0:
                    continue
                
                # 选择距离网格中心最近的点
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                pts = candidates[in_cell]
                distances = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
                best_local_idx = in_cell[np.argmin(distances)]
                
                if best_local_idx not in used_idx:
                    used_idx.add(best_local_idx)
                    x, y = candidates[best_local_idx]
                    selected.append((int(x), int(y)))
        
        # 如果网格采样不足，从剩余候选点中随机补充
        if len(selected) < num_points:
            remaining_indices = [i for i in range(len(candidates)) if i not in used_idx]
            if remaining_indices:
                need = min(num_points - len(selected), len(remaining_indices))
                extra_indices = np.random.choice(remaining_indices, need, replace=False)
                for idx in extra_indices:
                    x, y = candidates[idx]
                    selected.append((int(x), int(y)))
        
        return selected[:num_points]
    
    def map_points_to_target(self, reference_image: np.ndarray, points: List[Tuple[int, int]], 
                            target_image: np.ndarray) -> List[Tuple[int, int]]:
        """将参考图片的点映射到目标图片上"""
        # 简化版本：直接使用相同的坐标
        # 在实际应用中，这里应该使用光流或其他运动估计方法
        ref_h, ref_w = reference_image.shape[:2]
        target_h, target_w = target_image.shape[:2]
        
        # 简单的坐标缩放
        scale_x = target_w / ref_w
        scale_y = target_h / ref_h
        
        mapped_points = []
        for x, y in points:
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)
            
            # 确保坐标在目标图片范围内
            if 0 <= new_x < target_w and 0 <= new_y < target_h:
                mapped_points.append((new_x, new_y))
        
        return mapped_points
    
    def filter_positive_points(self, candidate_points: List[Tuple[int, int]], target_image: np.ndarray, 
                              reference_positive_rgbs: List[np.ndarray]) -> List[Tuple[int, int]]:
        """筛选正点：至少与参考图片的两个正点非常相似"""
        valid_points = []
        
        for x, y in candidate_points:
            target_rgb = target_image[y, x]
            
            # 计算与所有参考正点的相似性
            similar_count = 0
            for ref_rgb in reference_positive_rgbs:
                if self.is_rgb_very_similar(target_rgb, ref_rgb):
                    similar_count += 1
            
            # 至少与两个参考正点非常相似
            if similar_count >= 2:
                valid_points.append((x, y))
        
        return valid_points
    
    def filter_negative_points(self, candidate_points: List[Tuple[int, int]], target_image: np.ndarray, 
                              reference_positive_rgbs: List[np.ndarray]) -> List[Tuple[int, int]]:
        """筛选负点：至多与一个正点相似，但不能与任何正点非常相似"""
        valid_points = []
        
        for x, y in candidate_points:
            target_rgb = target_image[y, x]
            
            # 检查与参考正点的相似性
            similar_count = 0
            very_similar_count = 0
            
            for ref_rgb in reference_positive_rgbs:
                if self.is_rgb_very_similar(target_rgb, ref_rgb):
                    very_similar_count += 1
                elif self.is_rgb_similar(target_rgb, ref_rgb):
                    similar_count += 1
            
            # 负点条件：不能非常相似，至多与一个正点相似
            if very_similar_count == 0 and similar_count <= 1:
                valid_points.append((x, y))
        
        return valid_points
    
    def is_rgb_very_similar(self, rgb1: np.ndarray, rgb2: np.ndarray) -> bool:
        """判断两个RGB是否非常相似（欧几里得距离）"""
        distance = np.sqrt(np.sum((rgb1 - rgb2) ** 2))
        return distance < self.very_similar_threshold
    
    def is_rgb_similar(self, rgb1: np.ndarray, rgb2: np.ndarray) -> bool:
        """判断两个RGB是否相似（欧几里得距离）"""
        distance = np.sqrt(np.sum((rgb1 - rgb2) ** 2))
        return distance < self.similar_threshold


def create_prompt_generator(very_similar_threshold: float = 8.0, similar_threshold: float = 12.0) -> PromptPointGenerator:
    """
    创建Prompt点生成器的便捷函数
    
    Args:
        very_similar_threshold: 非常相似的RGB距离阈值
        similar_threshold: 相似的RGB距离阈值
        
    Returns:
        PromptPointGenerator: Prompt点生成器实例
    """
    return PromptPointGenerator(very_similar_threshold, similar_threshold)


if __name__ == "__main__":
    # 示例用法
    print("Prompt点生成模块")
    print("使用方法:")
    print("1. 创建生成器: generator = create_prompt_generator()")
    print("2. 生成点: points, labels = generator.generate_points_with_rgb_similarity(ref_image, ref_mask, target_image)")
