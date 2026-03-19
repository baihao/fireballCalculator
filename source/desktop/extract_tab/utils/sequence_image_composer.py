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
from .draw_utils import draw_segmentation_on_image


def _cv_imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    兼容中文/非 ASCII 路径的图像读取。
    
    Windows 下部分 OpenCV 版本对包含中文/特殊字符的路径使用 cv2.imread 会返回 None。
    使用 np.fromfile + cv2.imdecode 的方式避免编码问题。
    
    Args:
        path: 图像文件路径
        flags: OpenCV 读取标志，例如 cv2.IMREAD_COLOR / cv2.IMREAD_GRAYSCALE
        
    Returns:
        np.ndarray 或 None：成功返回图像数组，失败返回 None
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def _cv_imwrite_unicode(path: str, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """
    兼容中文/非 ASCII 路径的图像保存。
    
    Windows 下部分 OpenCV 版本对包含中文/特殊字符的路径使用 cv2.imwrite 会失败。
    使用 cv2.imencode + 文件写入的方式避免编码问题。
    
    Args:
        path: 图像文件路径
        image: 图像数据（BGR格式）
        params: OpenCV 编码参数（例如 [cv2.IMWRITE_JPEG_QUALITY, 95]）
        
    Returns:
        bool: 保存成功返回 True，失败返回 False
    """
    try:
        # 根据文件扩展名确定编码格式
        ext = Path(path).suffix.lower()
        if ext == '.jpg' or ext == '.jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            encode_ext = '.jpg'
        elif ext == '.png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
            encode_ext = '.png'
        elif ext == '.bmp':
            encode_params = None
            encode_ext = '.bmp'
        else:
            # 默认使用 JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            encode_ext = '.jpg'
        
        # 使用用户指定的参数（如果提供）
        if params is not None:
            encode_params = params
        
        # 编码图像数据
        if encode_params:
            success, encoded_data = cv2.imencode(encode_ext, image, encode_params)
        else:
            success, encoded_data = cv2.imencode(encode_ext, image)
        
        if not success:
            return False
        
        # 写入文件（使用二进制模式，支持中文路径）
        with open(path, 'wb') as f:
            f.write(encoded_data.tobytes())
        
        return True
    except Exception:
        return False


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
                # 读取图像（兼容中文路径）
                image = _cv_imread_unicode(image_path, cv2.IMREAD_COLOR)
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
                
                # 保存图像（兼容中文路径）
                success = _cv_imwrite_unicode(str(file_path), image_bgr)
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

