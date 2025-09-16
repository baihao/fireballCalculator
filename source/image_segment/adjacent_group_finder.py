#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相邻图片组查找工具模块
用于迭代掩码传播过程中查找相邻的已处理和未处理图片组
"""

from typing import List, Dict, Any, Optional, Set


class AdjacentGroupFinder:
    """相邻图片组查找器"""
    
    def __init__(self, processed_indices: Set[int], failed_indices: Set[int]):
        """
        初始化相邻组查找器
        
        Args:
            processed_indices: 已处理图片索引集合
            failed_indices: 处理失败图片索引集合
        """
        self.processed_indices = processed_indices
        self.failed_indices = failed_indices
    
    def find_adjacent_groups(self, total_images: int) -> List[Dict[str, Any]]:
        """
        找到相邻图片组，每个组包含一个已处理图片和其相邻的未处理图片
        确保每个未处理图片只被一个组包含，跳过已失败的图片
        如果直接相邻的图片失败，继续寻找更远的相邻图片
        
        Args:
            total_images: 总图片数
            
        Returns:
            List[Dict]: 相邻图片组列表，格式为:
            [
                {
                    'processed': 7,
                    'unprocessed': [6, 8],
                    'group_id': 0
                },
                ...
            ]
        """
        adjacent_groups = []
        processed_indices = list(self.processed_indices)
        used_unprocessed = set()  # 记录已经加入组的未处理图片
        
        # 为每个已处理的图片创建组
        for processed_idx in processed_indices:
            # 找到该已处理图片的相邻未处理图片
            unprocessed_neighbors = []
            
            # 向前寻找相邻的未处理图片
            prev_idx = self._find_next_available_index(processed_idx, -1, total_images, used_unprocessed)
            if prev_idx is not None:
                unprocessed_neighbors.append(prev_idx)
                used_unprocessed.add(prev_idx)
            
            # 向后寻找相邻的未处理图片
            next_idx = self._find_next_available_index(processed_idx, 1, total_images, used_unprocessed)
            if next_idx is not None:
                unprocessed_neighbors.append(next_idx)
                used_unprocessed.add(next_idx)
            
            # 如果该已处理图片有相邻的未处理图片，创建组
            if unprocessed_neighbors:
                group = {
                    'processed': processed_idx,
                    'unprocessed': unprocessed_neighbors,
                    'group_id': len(adjacent_groups)
                }
                adjacent_groups.append(group)
        
        return adjacent_groups
    
    def _find_next_available_index(self, start_idx: int, direction: int, total_images: int, used_unprocessed: Set[int]) -> Optional[int]:
        """
        从起始索引开始，按指定方向寻找下一个可用的未处理图片索引
        只有在遇到失败的图片时才继续寻找，已处理或已使用的图片会停止寻找
        
        Args:
            start_idx: 起始索引
            direction: 方向，-1表示向前，1表示向后
            total_images: 总图片数
            used_unprocessed: 已经使用的未处理图片索引集合
            
        Returns:
            Optional[int]: 找到的可用索引，如果没找到返回None
        """
        current_idx = start_idx + direction
        
        # 在有效范围内寻找
        while 0 <= current_idx < total_images:
            # 检查当前索引是否可用
            if (current_idx not in self.processed_indices and 
                current_idx not in self.failed_indices and 
                current_idx not in used_unprocessed):
                return current_idx
            
            # 如果当前索引已处理或已使用，停止寻找
            if current_idx in self.processed_indices or current_idx in used_unprocessed:
                return None
            
            # 如果当前索引失败，跳过它继续寻找
            if current_idx in self.failed_indices:
                current_idx += direction
            else:
                # 其他情况，继续寻找
                current_idx += direction
        
        return None


def create_adjacent_group_finder(processed_indices: Set[int], failed_indices: Set[int]) -> AdjacentGroupFinder:
    """
    创建相邻组查找器的便捷函数
    
    Args:
        processed_indices: 已处理图片索引集合
        failed_indices: 处理失败图片索引集合
        
    Returns:
        AdjacentGroupFinder: 相邻组查找器实例
    """
    return AdjacentGroupFinder(processed_indices, failed_indices)
