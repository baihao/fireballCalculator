#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：可视化候选点（top_candidates）的权重分布

功能：
1. 加载mask图像
2. 计算距离变换和权重
3. 获取top_candidates候选点（类似prompt_generation.py中的逻辑）
4. 可视化：候选点中前80%涂成红色，后20%涂成蓝色
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from typing import List, Tuple


def load_mask(mask_path: str) -> np.ndarray:
    """
    加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        np.ndarray: 二值化mask (0或1)
    """
    # 读取图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法加载mask图像: {mask_path}")
    
    # 二值化：大于127的为1，否则为0
    mask_binary = (mask > 127).astype(np.float32)
    
    return mask_binary


def get_mask_weights(mask: np.ndarray):
    """
    获取mask内所有点的权重
    
    Args:
        mask: 二值化mask (0或1)
        
    Returns:
        tuple: (weights_map, point_weights)
            - weights_map: 完整的权重图（与mask同尺寸）
            - point_weights: mask内所有点的权重数组
    """
    h, w = mask.shape
    
    # 计算距离变换，找到mask的中心区域
    mask_uint8 = mask.astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    
    # 创建权重：距离mask边缘越远权重越高
    weights_map = dist_transform.copy()
    
    # 获取mask内所有点的权重
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return weights_map, np.array([])
    
    point_weights = weights_map[y_coords, x_coords]
    
    return weights_map, point_weights


def _grid_sample_from_candidates(candidates: np.ndarray, num_points: int, w: int, h: int) -> List[Tuple[int, int]]:
    """
    从候选点中进行网格化均匀采样（复现prompt_generation.py中的逻辑）
    
    Args:
        candidates: 候选点坐标数组 (N, 2)
        num_points: 需要采样的点数
        w: 图像宽度
        h: 图像高度
        
    Returns:
        List[Tuple[int, int]]: 采样后的点坐标列表
    """
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


def sample_positive_points(mask: np.ndarray, num_points: int = 10, top_ratio: float = 0.8):
    """
    采样正点：优先选择mask中心区域的点（复现prompt_generation.py中的逻辑）
    
    Args:
        mask: 二值化mask (0或1)
        num_points: 采样点数
        top_ratio: 选择前top_ratio比例的候选点（默认0.8，即80%）
        
    Returns:
        tuple: (selected_points, top_candidates, candidate_weights, weights_map)
            - selected_points: 最终选择的点坐标列表
            - top_candidates: 候选点坐标数组 (N, 2)
            - candidate_weights: 候选点对应的权重数组 (N,)
            - weights_map: 完整的权重图
    """
    h, w = mask.shape
    
    # 计算距离变换，找到mask的中心区域
    mask_uint8 = mask.astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    
    # 创建权重：距离mask边缘越远权重越高
    weights = dist_transform.copy()
    
    # 只考虑mask内部的点
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return [], np.array([]), np.array([]), weights
    
    # 获取每个候选点的权重（距离边缘的距离）
    point_weights = weights[y_coords, x_coords]
    
    # 按权重排序，优先选择中心区域的点
    sorted_indices = np.argsort(point_weights)[::-1]  # 降序排列
    
    # 从高权重区域进行网格化采样
    candidates = np.stack([x_coords[sorted_indices], y_coords[sorted_indices]], axis=1)
    candidate_weights = point_weights[sorted_indices]
    
    # 选择前top_ratio权重的点作为候选池（避免边缘点）
    top_count = max(num_points * 2, int(len(candidates) * top_ratio))
    top_candidates = candidates[:top_count]
    top_weights = candidate_weights[:top_count]
    
    if len(top_candidates) == 0:
        return [], np.array([]), np.array([]), weights
    
    # 在高权重区域进行均匀网格采样
    selected = _grid_sample_from_candidates(top_candidates, num_points, w, h)
    
    return selected, top_candidates, top_weights, weights


def visualize_positive_points_selection(mask: np.ndarray, selected_points: List[Tuple[int, int]],
                                       top_candidates: np.ndarray, candidate_weights: np.ndarray,
                                       weights_map: np.ndarray, output_path: str = None,
                                       top_ratio: float = 0.8, cross_size: int = 8):
    """
    可视化正点选择策略的结果
    
    Args:
        mask: 二值化mask
        selected_points: 最终选择的点坐标列表（黄色十字标记）
        top_candidates: 候选点坐标数组 (N, 2)
        candidate_weights: 候选点对应的权重数组 (N,)
        weights_map: 完整的权重图（与mask同尺寸）
        output_path: 输出图像路径，如果为None则显示图像
        top_ratio: mask中前top_ratio比例的点为候选点集合（默认0.8）
        cross_size: 黄色十字的大小（像素），默认8
    """
    h, w = mask.shape
    
    # 创建RGB图像：背景为黑色
    visualization = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 获取mask内所有点的权重
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) > 0:
        point_weights = weights_map[y_coords, x_coords]
        
        # 计算权重阈值：用于区分前80%和后20%
        # 注意：sorted_weights是从小到大排序的，所以前80%应该是后20%的点（权重高的）
        sorted_weights = np.sort(point_weights)
        threshold_idx = int(len(sorted_weights) * (1 - top_ratio))  # 修正：应该是后80%，即前20%的点
        weight_threshold = sorted_weights[threshold_idx] if threshold_idx < len(sorted_weights) else sorted_weights[0]
        
        print(f"   Mask中总点数: {len(x_coords)}")
        print(f"   权重阈值（区分前{int(top_ratio*100)}%和后{int((1-top_ratio)*100)}%）: {weight_threshold:.2f}")
        print(f"   前{int(top_ratio*100)}%权重范围（高权重，红色）: [{sorted_weights[threshold_idx]:.2f}, {sorted_weights[-1]:.2f}]")
        print(f"   后{int((1-top_ratio)*100)}%权重范围（低权重，蓝色）: [{sorted_weights[0]:.2f}, {sorted_weights[threshold_idx]:.2f}]")
        print(f"   候选点总数: {len(top_candidates)}")
        print(f"   候选点权重范围: [{candidate_weights.min():.2f}, {candidate_weights.max():.2f}]")
        print(f"   最终选择点数: {len(selected_points)}")
        
        # 对mask内的每个点进行颜色映射
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            weight = weights_map[y, x]
            
            if weight >= weight_threshold:
                # 前top_ratio%的点（候选点集合）：标记为红色
                # 归一化到0-1范围（相对于前top_ratio%的权重范围）
                if sorted_weights[-1] > sorted_weights[threshold_idx]:
                    normalized = (weight - sorted_weights[threshold_idx]) / (sorted_weights[-1] - sorted_weights[threshold_idx])
                else:
                    normalized = 0.5
                # 红色强度：权重越高，红色越深
                red_intensity = max(100, int(normalized * 255))  # 最小100，确保可见
                visualization[y, x] = [0, 0, red_intensity]  # BGR格式：[B, G, R] - 红色
            else:
                # 后(1-top_ratio)%的点：标记为蓝色
                # 归一化到0-1范围（相对于后(1-top_ratio)%的权重范围）
                if sorted_weights[threshold_idx] > sorted_weights[0]:
                    normalized = (weight - sorted_weights[0]) / (sorted_weights[threshold_idx] - sorted_weights[0])
                else:
                    normalized = 0.5
                # 蓝色强度：权重越低，蓝色越深
                blue_intensity = max(100, int((1.0 - normalized) * 255))  # 最小100，确保可见
                visualization[y, x] = [blue_intensity, 0, 0]  # BGR格式：[B, G, R] - 蓝色
    
    # 在最终选择的点上绘制黄色十字
    for x, y in selected_points:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            # 绘制黄色十字
            # 水平线
            x_start = max(0, x - cross_size)
            x_end = min(w, x + cross_size + 1)
            visualization[y, x_start:x_end] = [0, 255, 255]  # BGR格式：[B, G, R] - 黄色
            
            # 垂直线
            y_start = max(0, y - cross_size)
            y_end = min(h, y + cross_size + 1)
            visualization[y_start:y_end, x] = [0, 255, 255]  # BGR格式：[B, G, R] - 黄色
    
    # 保存或显示图像
    if output_path:
        # 使用matplotlib保存，注意matplotlib使用RGB格式
        visualization_rgb = visualization[:, :, ::-1]  # BGR转RGB
        plt.figure(figsize=(12, 8))
        plt.imshow(visualization_rgb)
        plt.axis('off')
        plt.title(f'正点选择策略可视化\n(红色：mask中权重前{int(top_ratio*100)}%%的点（候选点集合）；蓝色：mask中其余点；黄色十字：最终选择的点)', 
                 fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化结果已保存到: {output_path}")
        plt.close()
    else:
        # 使用matplotlib显示，注意matplotlib使用RGB格式
        visualization_rgb = visualization[:, :, ::-1]  # BGR转RGB
        plt.figure(figsize=(12, 8))
        plt.imshow(visualization_rgb)
        plt.axis('off')
        plt.title(f'正点选择策略可视化\n(红色：mask中权重前{int(top_ratio*100)}%%的点（候选点集合）；蓝色：mask中其余点；黄色十字：最终选择的点)',
                 fontsize=12, pad=10)
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='可视化候选点（top_candidates）的权重分布',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（显示图像）
  python test_top_candidates_visualization.py mask.png
  
  # 保存到文件
  python test_top_candidates_visualization.py mask.png -o output.png
  
  # 指定采样点数和top_ratio
  python test_top_candidates_visualization.py mask.png -n 20 -r 0.8 -o output.png
  
  # 指定后20%%的比例（例如改为后30%%）
  python test_top_candidates_visualization.py mask.png -b 0.3 -o output.png
        """
    )
    
    parser.add_argument('mask_path', type=str, help='mask图像路径')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='输出图像路径（如果未指定则显示图像）')
    parser.add_argument('-n', '--num_points', type=int, default=10,
                       help='采样点数（用于计算top_count，默认10）')
    parser.add_argument('-r', '--top_ratio', type=float, default=None,
                       help='选择前top_ratio比例的候选点（默认0.8，与-w参数互斥）')
    parser.add_argument('-w', '--weight_percent', type=int, default=None,
                       help='从权重排序前weight_percent%%的点中选择（例如：80表示前80%%，60表示前60%%）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.mask_path):
        print(f"❌ 错误: 找不到mask文件: {args.mask_path}")
        return
    
    print(f"📁 加载mask: {args.mask_path}")
    
    try:
        # 加载mask
        mask = load_mask(args.mask_path)
        print(f"✅ Mask尺寸: {mask.shape}")
        print(f"   Mask中非零像素数: {np.sum(mask > 0)}")
        
        # 使用与prompt_generation.py相同的策略选择正点
        print(f"\n🔍 使用正点选择策略...")
        print(f"   采样点数: {args.num_points}")
        
        # 确定top_ratio：优先使用-w参数，如果未指定则使用-r参数，默认0.8
        if args.weight_percent is not None:
            if args.top_ratio is not None:
                print(f"   ⚠️ 警告: 同时指定了-r和-w参数，优先使用-w参数")
            top_ratio = args.weight_percent / 100.0
            print(f"   权重百分比: {args.weight_percent}% (top_ratio={top_ratio})")
        elif args.top_ratio is not None:
            top_ratio = args.top_ratio
            print(f"   Top比例: {top_ratio}")
        else:
            top_ratio = 0.8
            print(f"   Top比例: {top_ratio} (默认)")
        
        selected_points, top_candidates, candidate_weights, weights_map = sample_positive_points(
            mask, args.num_points, top_ratio
        )
        
        if len(top_candidates) == 0:
            print("❌ 错误: mask中没有有效像素，无法计算候选点")
            return
        
        if len(selected_points) == 0:
            print("❌ 错误: 无法选择正点")
            return
        
        print(f"✅ 候选点数量: {len(top_candidates)}")
        print(f"   最终选择点数: {len(selected_points)}")
        print(f"   权重范围: [{candidate_weights.min():.2f}, {candidate_weights.max():.2f}]")
        
        # 可视化
        print(f"\n🎨 生成可视化...")
        visualize_positive_points_selection(mask, selected_points, top_candidates, 
                                           candidate_weights, weights_map, args.output, 
                                           top_ratio, cross_size=8)
        
        print("\n✅ 完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

