#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt点生成模块
负责从参考图片生成目标图片的prompt点，包括采样、映射和筛选功能
"""

import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# 参考 mask 上采样的正点候选数；滤后在目标图上均匀采样为 SAM 正点
POSITIVE_REFERENCE_CANDIDATE_COUNT = 40
DEFAULT_POSITIVE_POINT_COUNT = 10
DEFAULT_NEGATIVE_POINT_COUNT = 10
# 负点候选池相对最终数量的倍数（filter / 均匀采样前多采一些）
NEGATIVE_CANDIDATE_POOL_FACTOR = 2
# 映射+RGB 滤后至少需要的正负点数，否则视为投射失败
MIN_PROJECTED_POINTS_PER_LABEL = 2
# 目标像素在 RGB 池中至少相似的个数：max(2, 10%·池大小)
POOL_SIMILAR_MATCH_FRACTION = 0.10


class PromptPointGenerator:
    """Prompt点生成器"""
    
    def __init__(self, very_similar_threshold: float = 16.0, similar_threshold: float = 32.0):
        """
        初始化Prompt点生成器
        
        Args:
            very_similar_threshold: 非常相似的RGB距离阈值
            similar_threshold: 相似的RGB距离阈值
        """
        self.very_similar_threshold = very_similar_threshold
        self.similar_threshold = similar_threshold

    @staticmethod
    def _projection_has_enough_points(
        filtered_positive: List[Tuple[int, int]],
        filtered_negative: List[Tuple[int, int]],
    ) -> bool:
        return (
            len(filtered_positive) >= MIN_PROJECTED_POINTS_PER_LABEL
            and len(filtered_negative) >= MIN_PROJECTED_POINTS_PER_LABEL
        )

    def _sample_reference_point_candidates(
        self,
        reference_mask: np.ndarray,
        num_positive: int = POSITIVE_REFERENCE_CANDIDATE_COUNT,
        num_negative: int = DEFAULT_NEGATIVE_POINT_COUNT,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """从参考 mask 内/外采样正负点候选（单 ref 路径共用）。"""
        positive_candidates = self.sample_points_from_mask(
            reference_mask, num_points=num_positive, inside_mask=True
        )
        negative_candidates = self.sample_points_from_mask(
            reference_mask, num_points=num_negative, inside_mask=False
        )
        return positive_candidates, negative_candidates

    def _resolve_reference_candidates(
        self,
        reference_mask: np.ndarray,
        predefined_reference_points: Optional[Dict[str, List[Tuple[int, int]]]],
        num_positive: int = POSITIVE_REFERENCE_CANDIDATE_COUNT,
        num_negative: int = DEFAULT_NEGATIVE_POINT_COUNT,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], bool]:
        """
        参考正点始终从 mask 采 num_positive（默认 40）个；负点可用预定义或 mask 采样。

        Returns:
            (positive_candidates, negative_candidates, used_predefined)
        """
        used_predefined = False
        positive_candidates = self.sample_points_from_mask(
            reference_mask, num_points=num_positive, inside_mask=True
        )
        negative_candidates: List[Tuple[int, int]] = []
        if predefined_reference_points is not None:
            negative_candidates = predefined_reference_points.get('negative') or []
            if negative_candidates:
                used_predefined = True
        if not negative_candidates:
            negative_candidates = self.sample_points_from_mask(
                reference_mask, num_points=num_negative, inside_mask=False
            )
        return positive_candidates, negative_candidates, used_predefined
    
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
            positive_candidates, negative_candidates, used_predefined = (
                self._resolve_reference_candidates(
                    reference_mask, predefined_reference_points
                )
            )
            if used_predefined:
                print(
                    f"    参考正点采样{len(positive_candidates)}个，"
                    f"预定义负点{len(negative_candidates)}个"
                )
            else:
                print(
                    f"    参考点采样: 正{len(positive_candidates)} 负{len(negative_candidates)}"
                )

            target_positive_candidates, projection_source_rgbs = (
                self.map_points_to_target_with_source_rgb(
                    reference_image, positive_candidates, target_image
                )
            )
            target_negative_candidates = self.map_points_to_target(
                reference_image, negative_candidates, target_image
            )

            reference_positive_rgbs = [
                reference_image[y, x] for x, y in positive_candidates
            ]
            reference_negative_rgbs = [
                reference_image[y, x] for x, y in negative_candidates
            ]

            target_positive_points = self.filter_positive_points(
                target_positive_candidates,
                target_image,
                reference_positive_rgbs,
                reference_negative_rgbs,
                projection_source_rgbs=projection_source_rgbs,
            )
            target_negative_points = self.filter_negative_points(
                target_negative_candidates,
                target_image,
                reference_negative_rgbs,
                reference_positive_rgbs,
            )

            if not self._projection_has_enough_points(
                target_positive_points, target_negative_points
            ):
                print(
                    f"    ❌ 投射失败: 滤后正点 {len(target_positive_points)}、"
                    f"负点 {len(target_negative_points)}，"
                    f"均需 ≥ {MIN_PROJECTED_POINTS_PER_LABEL}"
                )
                debug = {
                    'reference_positive': positive_candidates,
                    'reference_negative': negative_candidates,
                    'mapped_positive': target_positive_candidates,
                    'mapped_negative': target_negative_candidates,
                    'filtered_positive': target_positive_points,
                    'filtered_negative': target_negative_points,
                    'final_positive': [],
                    'final_negative': target_negative_points,
                    'projection_failed': True,
                }
                if return_debug_info:
                    return [], [], debug
                return [], []

            h, w = target_image.shape[:2]
            final_positive = self.uniform_sample_points(
                target_positive_points, DEFAULT_POSITIVE_POINT_COUNT, w, h
            )
            if len(final_positive) < MIN_PROJECTED_POINTS_PER_LABEL:
                print(
                    f"    ❌ 投射失败: 滤后均匀采样正点 {len(final_positive)}，"
                    f"需 ≥ {MIN_PROJECTED_POINTS_PER_LABEL}"
                )
                debug = {
                    'reference_positive': positive_candidates,
                    'reference_negative': negative_candidates,
                    'mapped_positive': target_positive_candidates,
                    'mapped_negative': target_negative_candidates,
                    'filtered_positive': target_positive_points,
                    'filtered_negative': target_negative_points,
                    'final_positive': final_positive,
                    'final_negative': target_negative_points,
                    'projection_failed': True,
                }
                if return_debug_info:
                    return [], [], debug
                return [], []

            final_points = final_positive + target_negative_points
            final_labels = [1] * len(final_positive) + [0] * len(target_negative_points)

            print(
                f"    参考正{len(positive_candidates)}→滤{len(target_positive_points)}"
                f"→匀{len(final_positive)}，负点滤后 {len(target_negative_points)} 个"
            )

            if return_debug_info:
                return final_points, final_labels, {
                    'reference_positive': positive_candidates,
                    'reference_negative': negative_candidates,
                    'mapped_positive': target_positive_candidates,
                    'mapped_negative': target_negative_candidates,
                    'filtered_positive': target_positive_points,
                    'filtered_negative': target_negative_points,
                    'final_positive': final_positive,
                    'final_negative': target_negative_points,
                    'projection_failed': False,
                }
            return final_points, final_labels
            
        except Exception as e:
            print(f"    ⚠️ 点生成失败: {e}")
            if return_debug_info:
                return [], [], {}
            return [], []

    def generate_points_multi_reference(
        self,
        reference_entries: List[Dict[str, Any]],
        primary_entry: Dict[str, Any],
        target_image: np.ndarray,
        num_positive: int = DEFAULT_POSITIVE_POINT_COUNT,
        num_negative: int = DEFAULT_NEGATIVE_POINT_COUNT,
        return_debug_info: bool = False,
    ):
        """
        多参考图传播：正点来自各 ref 缓存正点映射合并；负点来自多帧背景 mask 交集。
        """
        try:
            h, w = target_image.shape[:2]
            mapped_positive: List[Tuple[int, int]] = []
            reference_positive: List[Tuple[int, int]] = []
            reference_positive_rgbs: List[np.ndarray] = []
            seen_pos: set = set()

            for entry in reference_entries:
                ref_image = entry['image_rgb']
                pos_pts = entry.get('positive_points') or []
                reference_positive.extend(pos_pts)
                for x, y in pos_pts:
                    reference_positive_rgbs.append(ref_image[y, x])
                for pt in self.map_points_to_target(ref_image, pos_pts, target_image):
                    if pt not in seen_pos:
                        seen_pos.add(pt)
                        mapped_positive.append(pt)

            ref_masks = [e['mask'] for e in reference_entries]
            primary_image = primary_entry['image_rgb']
            primary_shape = primary_image.shape[:2]
            bg_intersection = self.compute_background_mask_intersection(ref_masks, primary_shape)
            intersection_pixels = int(np.sum(bg_intersection))
            pool_n = num_negative * NEGATIVE_CANDIDATE_POOL_FACTOR
            negative_candidates = self.sample_points_from_background_mask(
                bg_intersection, pool_n
            )
            if len(negative_candidates) < num_negative:
                fallback_mask = (primary_entry['mask'] == 0)
                extra = self.sample_points_from_background_mask(fallback_mask, pool_n)
                for pt in extra:
                    if pt not in negative_candidates:
                        negative_candidates.append(pt)

            reference_negative_rgbs = [
                primary_image[y, x] for x, y in negative_candidates
            ]

            filtered_positive = self.filter_positive_points(
                mapped_positive,
                target_image,
                reference_positive_rgbs,
                reference_negative_rgbs,
            )
            final_positive = self.uniform_sample_points(
                filtered_positive, num_positive, w, h
            )
            mapped_negative = self.map_points_to_target(
                primary_image, negative_candidates, target_image
            )
            filtered_negative = self.filter_negative_points(
                mapped_negative,
                target_image,
                reference_negative_rgbs,
                reference_positive_rgbs,
            )

            if not self._projection_has_enough_points(
                filtered_positive, filtered_negative
            ):
                print(
                    f"    ❌ 投射失败: 滤后正点 {len(filtered_positive)}、"
                    f"负点 {len(filtered_negative)}，"
                    f"均需 ≥ {MIN_PROJECTED_POINTS_PER_LABEL}"
                )
                debug = {
                    'reference_positive': reference_positive,
                    'reference_negative': negative_candidates,
                    'mapped_positive': mapped_positive,
                    'mapped_negative': mapped_negative,
                    'filtered_positive': filtered_positive,
                    'filtered_negative': filtered_negative,
                    'final_positive': [],
                    'final_negative': [],
                    'projection_failed': True,
                    'primary_ref_idx': primary_entry['idx'],
                    'background_intersection_pixels': intersection_pixels,
                }
                if return_debug_info:
                    return [], [], debug
                return [], []

            final_negative = self.uniform_sample_points(
                filtered_negative, num_negative, w, h
            )

            final_points = final_positive + final_negative
            final_labels = [1] * len(final_positive) + [0] * len(final_negative)

            chain_ids = [e['idx'] for e in reference_entries]
            print(
                f"    多参考传播链 {chain_ids}: "
                f"映射正{len(mapped_positive)}→滤{len(filtered_positive)}→匀{len(final_positive)}，"
                f"背景交集像素={intersection_pixels}，"
                f"负点滤{len(filtered_negative)}→匀{len(final_negative)}"
            )

            if return_debug_info:
                debug = {
                    'reference_positive': reference_positive,
                    'reference_negative': negative_candidates,
                    'mapped_positive': mapped_positive,
                    'mapped_negative': mapped_negative,
                    'filtered_positive': filtered_positive,
                    'filtered_negative': filtered_negative,
                    'final_positive': final_positive,
                    'final_negative': final_negative,
                    'primary_ref_idx': primary_entry['idx'],
                    'background_intersection_pixels': intersection_pixels,
                    'projection_failed': False,
                }
                return final_points, final_labels, debug
            return final_points, final_labels

        except Exception as e:
            print(f"    ⚠️ 多参考点生成失败: {e}")
            if return_debug_info:
                return [], [], {}
            return [], []

    @staticmethod
    def compute_background_mask_intersection(
        masks: List[np.ndarray], shape: Tuple[int, int]
    ) -> np.ndarray:
        """多帧前景 mask 的背景区域交集（True 表示稳定背景）。"""
        h, w = shape
        intersection = np.ones((h, w), dtype=bool)
        for mask in masks:
            if mask.shape[:2] != (h, w):
                aligned = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            else:
                aligned = mask.astype(np.uint8)
            intersection &= (aligned == 0)
        return intersection

    def sample_points_from_background_mask(
        self, background: np.ndarray, num_points: int
    ) -> List[Tuple[int, int]]:
        """在背景 bool mask（True=背景）上采样负点。"""
        if not np.any(background):
            return []
        foreground = (~background).astype(np.uint8)
        return self._sample_negative_points(foreground, num_points)

    def uniform_sample_points(
        self,
        points: List[Tuple[int, int]],
        num_points: int,
        w: int,
        h: int,
    ) -> List[Tuple[int, int]]:
        """对目标侧已筛选点做空间均匀采样。"""
        if not points:
            return []
        if len(points) <= num_points:
            return list(points)
        candidates = np.array(points, dtype=np.int32)
        return self._grid_sample_from_candidates(candidates, num_points, w, h)
    
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

    def map_points_to_target_with_source_rgb(
        self,
        reference_image: np.ndarray,
        points: List[Tuple[int, int]],
        target_image: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """映射到目标图，并返回与每个目标点一一对应的参考图 RGB（P_o）。"""
        ref_h, ref_w = reference_image.shape[:2]
        target_h, target_w = target_image.shape[:2]
        scale_x = target_w / ref_w
        scale_y = target_h / ref_h

        mapped_points: List[Tuple[int, int]] = []
        source_rgbs: List[np.ndarray] = []
        for x, y in points:
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)
            if 0 <= new_x < target_w and 0 <= new_y < target_h:
                mapped_points.append((new_x, new_y))
                source_rgbs.append(reference_image[y, x])
        return mapped_points, source_rgbs

    @staticmethod
    def _required_similar_matches_in_pool(pool_size: int) -> int:
        """同一 target 像素在 RGB 池中需相似的个数：max(2, 10%·池大小)（向上取整）。"""
        if pool_size <= 0:
            return MIN_PROJECTED_POINTS_PER_LABEL
        quota = max(2.0, pool_size * POOL_SIMILAR_MATCH_FRACTION)
        return int(math.ceil(quota - 1e-9))

    @staticmethod
    def _rgb_distance(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        a = np.asarray(rgb1, dtype=np.float64)
        b = np.asarray(rgb2, dtype=np.float64)
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def _min_rgb_distance(self, rgb: np.ndarray, ref_rgbs: List[np.ndarray]) -> float:
        if not ref_rgbs:
            return float("inf")
        return min(self._rgb_distance(rgb, ref) for ref in ref_rgbs)

    def filter_positive_points(
        self,
        candidate_points: List[Tuple[int, int]],
        target_image: np.ndarray,
        reference_positive_rgbs: List[np.ndarray],
        reference_negative_rgbs: Optional[List[np.ndarray]] = None,
        projection_source_rgbs: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[int, int]]:
        """
        正点：不像参考负点，且更接近正池；并满足下列之一：
        - P_o（参考点）与 P_t（映射目标点）RGB 极度相似 → 直接合法；
        - 否则 target 在正点 RGB 池中 very_similar 个数 ≥ max(2, 10%·池大小)。
        """
        ref_neg = reference_negative_rgbs or []
        need = self._required_similar_matches_in_pool(len(reference_positive_rgbs))
        valid_points = []
        paired_source = projection_source_rgbs or []

        for i, (x, y) in enumerate(candidate_points):
            target_rgb = target_image[y, x]

            if ref_neg and any(
                self.is_rgb_very_similar(target_rgb, nr) for nr in ref_neg
            ):
                continue

            d_pos = self._min_rgb_distance(target_rgb, reference_positive_rgbs)
            d_neg = self._min_rgb_distance(target_rgb, ref_neg)
            if ref_neg and d_pos >= d_neg:
                continue

            if i < len(paired_source) and self.is_rgb_very_similar(
                target_rgb, paired_source[i]
            ):
                valid_points.append((x, y))
                continue

            similar_count = sum(
                1
                for ref_rgb in reference_positive_rgbs
                if self.is_rgb_very_similar(target_rgb, ref_rgb)
            )
            if similar_count >= need:
                valid_points.append((x, y))

        return valid_points

    def filter_negative_points(
        self,
        candidate_points: List[Tuple[int, int]],
        target_image: np.ndarray,
        reference_negative_rgbs: List[np.ndarray],
        reference_positive_rgbs: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[int, int]]:
        """
        负点：与正点对称，在参考负点 RGB 池中凑够 max(2, 10%·池大小) 个相似，
        且不像参考正点（火球）。
        """
        ref_pos = reference_positive_rgbs or []
        need = self._required_similar_matches_in_pool(len(reference_negative_rgbs))
        valid_points = []

        for x, y in candidate_points:
            target_rgb = target_image[y, x]

            if ref_pos and any(
                self.is_rgb_very_similar(target_rgb, pr) for pr in ref_pos
            ):
                continue

            d_neg = self._min_rgb_distance(target_rgb, reference_negative_rgbs)
            d_pos = self._min_rgb_distance(target_rgb, ref_pos)
            if ref_pos and d_neg >= d_pos:
                continue

            similar_count = sum(
                1
                for ref_rgb in reference_negative_rgbs
                if self.is_rgb_very_similar(target_rgb, ref_rgb)
            )
            if similar_count >= need:
                valid_points.append((x, y))

        return valid_points

    def is_rgb_very_similar(self, rgb1: np.ndarray, rgb2: np.ndarray) -> bool:
        """判断两个RGB是否非常相似（欧几里得距离，float 计算避免 uint8 溢出）"""
        return self._rgb_distance(rgb1, rgb2) < self.very_similar_threshold

    def is_rgb_similar(self, rgb1: np.ndarray, rgb2: np.ndarray) -> bool:
        """判断两个RGB是否相似（欧几里得距离）"""
        return self._rgb_distance(rgb1, rgb2) < self.similar_threshold


def create_prompt_generator(very_similar_threshold: float = 16.0, similar_threshold: float = 32.0) -> PromptPointGenerator:
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
