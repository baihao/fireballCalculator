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
        """从掩码内部或外部采样点"""
        h, w = mask.shape
        
        if inside_mask:
            # 从掩码内部采样
            y_coords, x_coords = np.where(mask > 0)
        else:
            # 从掩码外部采样，但避免边缘区域
            # 创建一个扩展的掩码，排除掩码周围的区域
            mask_uint8 = mask.astype(np.uint8)
            expanded_mask = cv2.dilate(mask_uint8, np.ones((20, 20), np.uint8), iterations=1)
            y_coords, x_coords = np.where(expanded_mask == 0)
            
            # 如果外部点太少，从图像边缘采样
            if len(x_coords) < num_points:
                # 从图像边缘采样
                edge_points = []
                # 上边缘
                edge_points.extend([(x, 0) for x in range(0, w, 10)])
                # 下边缘
                edge_points.extend([(x, h-1) for x in range(0, w, 10)])
                # 左边缘
                edge_points.extend([(0, y) for y in range(0, h, 10)])
                # 右边缘
                edge_points.extend([(w-1, y) for y in range(0, h, 10)])
                
                # 过滤掉掩码内部的点
                valid_edge_points = []
                for x, y in edge_points:
                    if mask[y, x] == 0:  # 不在掩码内部
                        valid_edge_points.append((x, y))
                
                if valid_edge_points:
                    x_coords = np.array([p[0] for p in valid_edge_points])
                    y_coords = np.array([p[1] for p in valid_edge_points])
        
        if len(x_coords) == 0:
            return []
        
        # 基于网格的分层均匀采样（优先均匀分布，其次再随机补齐）
        total_candidates = len(x_coords)
        if total_candidates == 0:
            return []

        # 候选点数组
        candidates = np.stack([x_coords, y_coords], axis=1)

        # 计算网格行列数，使 cell 数约等于 num_points
        grid_rows = max(1, int(np.sqrt(num_points)))
        grid_cols = max(1, int(np.ceil(num_points / grid_rows)))

        cell_w = max(1, w // grid_cols)
        cell_h = max(1, h // grid_rows)

        selected = []
        used_idx = set()

        # 遍历每个网格单元，选择距离单元中心最近的候选点
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

                # 找出落在该 cell 内的候选索引
                in_cell = np.where(
                    (candidates[:, 0] >= x0) & (candidates[:, 0] < x1) &
                    (candidates[:, 1] >= y0) & (candidates[:, 1] < y1)
                )[0]

                if in_cell.size == 0:
                    continue

                # 选择与 cell 中心最近的点
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                pts = candidates[in_cell]
                d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
                pick_local = in_cell[np.argmin(d2)]
                if pick_local not in used_idx:
                    used_idx.add(pick_local)
                    selected.append((int(candidates[pick_local, 0]), int(candidates[pick_local, 1])))

        # 若不足 num_points，再从未用候选中随机补齐
        if len(selected) < num_points:
            remaining = [i for i in range(total_candidates) if i not in used_idx]
            if remaining:
                need = min(num_points - len(selected), len(remaining))
                extra_idx = np.random.choice(remaining, need, replace=False)
                for i in extra_idx:
                    selected.append((int(candidates[i, 0]), int(candidates[i, 1])))

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
