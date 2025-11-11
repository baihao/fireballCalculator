#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列图像组合器
根据分割结果在序列图像上绘制火球轮廓和最大直径，并支持保存为图像文件
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


def draw_segmentation_on_image(image: np.ndarray, segmentation_result: Dict[str, Any]) -> np.ndarray:
    """
    在图像上绘制分割结果（按照 interactive_image_widget.py 的规则）
    
    Args:
        image: 原始图像数组（RGB格式）
        segmentation_result: 分割结果字典，包含 contours、centroid、max_radius 等信息
        
    Returns:
        np.ndarray: 绘制后的图像数组（RGB格式）
    """
    try:
        # 创建图像副本，避免修改原始图像
        display_image = image.copy()
        
        # 检查分割结果是否有效
        if segmentation_result is None or not segmentation_result.get('success', False):
            return display_image
        
        # 1. 绘制蓝色轮廓
        contours = segmentation_result.get('contours', [])
        if contours:
            for contour_points in contours:
                if len(contour_points) > 2:
                    # 转换为numpy数组格式
                    contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
                    # 绘制蓝色轮廓，线宽3（RGB格式，蓝色 (0, 0, 255)）
                    cv2.drawContours(display_image, [contour_array], -1, (0, 0, 255), 3)
        
        # 2. 绘制绿色质心到最大半径的箭头
        centroid_data = segmentation_result.get('centroid', {})
        max_radius_data = segmentation_result.get('max_radius', {})
        
        if centroid_data and max_radius_data:
            # 获取质心坐标
            cx = int(centroid_data.get('x', 0))
            cy = int(centroid_data.get('y', 0))
            
            # 获取最大半径端点坐标
            endpoint_data = max_radius_data.get('endpoint', {})
            ex = int(endpoint_data.get('x', 0))
            ey = int(endpoint_data.get('y', 0))
            
            # 确保坐标在图像范围内
            h, w = display_image.shape[:2]
            if (0 <= cx < w and 0 <= cy < h and 
                0 <= ex < w and 0 <= ey < h):
                
                # 绘制绿色箭头，从质心指向最大半径端点（RGB格式，绿色 (0, 255, 0)）
                cv2.arrowedLine(display_image, (cx, cy), (ex, ey), (0, 255, 0), 3, tipLength=0.1)
                
                # 在质心绘制紫色圆点（RGB格式，紫色 (128, 0, 128)）
                cv2.circle(display_image, (cx, cy), 5, (128, 0, 128), -1)  # 填充的紫色圆点
                
                # 在最大半径端点绘制小圆圈（RGB格式，绿色 (0, 255, 0)）
                cv2.circle(display_image, (ex, ey), 3, (0, 255, 0), 2)  # 绿色圆圈
        
        return display_image
        
    except Exception as e:
        print(f"❌ 绘制分割结果失败: {e}")
        return image  # 出错时返回原始图像


def compose_sequence_images(
    image_paths: List[str],
    segmentation_results: List[Dict[str, Any]]
) -> List[np.ndarray]:
    """
    根据分割结果在序列图像上绘制火球轮廓和最大直径
    
    Args:
        image_paths: 图像文件路径列表
        segmentation_results: 分割结果列表，每个元素对应一张图像的分割结果
        
    Returns:
        List[np.ndarray]: 处理后的图像数据列表（RGB格式）
    """
    composed_images = []
    
    try:
        # 确保图像路径和分割结果数量一致
        num_images = len(image_paths)
        num_results = len(segmentation_results)
        
        if num_images != num_results:
            print(f"⚠️ 警告: 图像数量 ({num_images}) 与分割结果数量 ({num_results}) 不一致")
        
        # 处理每张图像
        for i, image_path in enumerate(image_paths):
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ 无法加载图像: {image_path}")
                    continue
                
                # 转换为RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 获取对应的分割结果
                if i < len(segmentation_results):
                    segmentation_result = segmentation_results[i]
                else:
                    segmentation_result = {'success': False}
                
                # 在图像上绘制分割结果
                composed_image = draw_segmentation_on_image(image_rgb, segmentation_result)
                composed_images.append(composed_image)
                
            except Exception as e:
                print(f"❌ 处理图像 {image_path} 失败: {e}")
                continue
        
        print(f"✓ 成功组合 {len(composed_images)} 张图像")
        return composed_images
        
    except Exception as e:
        print(f"❌ 组合序列图像失败: {e}")
        return []


def save_composed_images(
    composed_images: List[np.ndarray],
    output_dir: str,
    base_name: str = "composed",
    image_format: str = "jpg",
    start_index: int = 0
) -> Tuple[bool, List[str]]:
    """
    保存组合后的图像到文件
    
    Args:
        composed_images: 处理后的图像数据列表（RGB格式）
        output_dir: 输出目录路径
        base_name: 基础文件名（不含扩展名）
        image_format: 图像格式（'jpg', 'png', 'bmp' 等）
        start_index: 起始索引（用于文件名编号）
        
    Returns:
        Tuple[bool, List[str]]: (是否成功, 保存的文件路径列表)
    """
    try:
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # 保存每张图像
        for i, image in enumerate(composed_images):
            try:
                # 转换为BGR格式（OpenCV保存需要）
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 生成文件名
                file_name = f"{base_name}_{start_index + i:04d}.{image_format}"
                file_path = output_path / file_name
                
                # 保存图像
                success = cv2.imwrite(str(file_path), image_bgr)
                if success:
                    saved_paths.append(str(file_path))
                else:
                    print(f"⚠️ 保存图像失败: {file_path}")
                    
            except Exception as e:
                print(f"❌ 保存图像 {i} 失败: {e}")
                continue
        
        if saved_paths:
            print(f"✓ 成功保存 {len(saved_paths)} 张图像到 {output_dir}")
            return True, saved_paths
        else:
            print(f"❌ 没有成功保存任何图像")
            return False, []
            
    except Exception as e:
        print(f"❌ 保存组合图像失败: {e}")
        return False, []


def compose_and_save(
    image_paths: List[str],
    segmentation_results: List[Dict[str, Any]],
    output_dir: str,
    base_name: str = "composed",
    image_format: str = "jpg",
    start_index: int = 0
) -> Tuple[bool, List[str]]:
    """
    组合序列图像并保存（便捷函数）
    
    Args:
        image_paths: 图像文件路径列表
        segmentation_results: 分割结果列表
        output_dir: 输出目录路径
        base_name: 基础文件名（不含扩展名）
        image_format: 图像格式（'jpg', 'png', 'bmp' 等）
        start_index: 起始索引（用于文件名编号）
        
    Returns:
        Tuple[bool, List[str]]: (是否成功, 保存的文件路径列表)
    """
    # 组合图像
    composed_images = compose_sequence_images(image_paths, segmentation_results)
    
    if not composed_images:
        return False, []
    
    # 保存图像
    return save_composed_images(
        composed_images,
        output_dir,
        base_name,
        image_format,
        start_index
    )


if __name__ == "__main__":
    # 测试模块功能
    import sys
    
    print("序列图像组合器测试")
    print("=" * 50)
    
    # 示例用法
    if len(sys.argv) < 3:
        print("用法: python sequence_image_composer.py <图像目录> <分割结果JSON文件> [输出目录]")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    json_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "output_composed"
    
    # 这里需要实现加载JSON文件的逻辑
    # 示例代码：
    # import json
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     sequence_data = json.load(f)
    # image_paths = [str(Path(image_dir) / f) for f in sorted(Path(image_dir).glob("*.jpg"))]
    # segmentation_results = sequence_data.get('image_sequence_segmentation', [])
    # success, saved_paths = compose_and_save(image_paths, segmentation_results, output_dir)
    
    print(f"图像目录: {image_dir}")
    print(f"分割结果文件: {json_file}")
    print(f"输出目录: {output_dir}")

