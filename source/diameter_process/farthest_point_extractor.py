#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最远点数据提取模块

提供从分割序列数据中提取每张图片的最远点坐标和像素半径的功能
"""

import sys
import os
import csv
from typing import Dict, Any, Optional

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
source_path = os.path.join(project_root, 'source')
sys.path.insert(0, source_path)
# 确保 desktop 目录也在路径中，以便能找到 framework 模块
desktop_path = os.path.join(source_path, 'desktop')
if desktop_path not in sys.path:
    sys.path.insert(0, desktop_path)


def extract_farthest_point_data(sequence_data: Dict[str, Any], output_dir: Optional[str], 
                                file_path: str) -> Optional[str]:
    """
    从序列数据中提取每张图片的最远点坐标和像素半径，并输出CSV文件
    
    Args:
        sequence_data: 序列数据字典
        output_dir: 输出目录（可选）
        file_path: 输入文件路径（用于生成输出文件名）
        
    Returns:
        Optional[str]: CSV文件路径，如果失败则返回None
    """
    # 延迟导入，只在需要时导入
    from desktop.extract_tab.utils.sequence_manager import SequenceManager
    
    try:
        manager = SequenceManager()
        
        # 获取分割结果
        segmentation_results = manager.get_segmentation_results_from_sequence(sequence_data)
        if not segmentation_results:
            print("⚠️ 无法获取分割结果，跳过最远点数据提取")
            return None
        
        # 获取爆炸时长参数
        parameters = manager.get_parameters_from_sequence(sequence_data)
        explosion_duration_str = parameters.get('explosion_duration', '100')
        
        try:
            explosion_duration_ms = float(explosion_duration_str)
        except ValueError:
            explosion_duration_ms = 140  # 默认140毫秒
        
        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(file_path) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_output_path = os.path.join(output_dir, f"{base_name}_farthest_points.csv")
        
        # 提取最远点数据
        total_images = len(segmentation_results)
        farthest_point_data = []
        
        for i, result in enumerate(segmentation_results):
            # 计算时间（线性映射）
            time_ms = (i / (total_images - 1) * explosion_duration_ms) if total_images > 1 else 0.0
            
            if not result or not result.get("success", False):
                # 分割失败，记录空值
                farthest_point_data.append({
                    'image_index': i,
                    'time_ms': time_ms,
                    'farthest_x': None,
                    'farthest_y': None,
                    'pixel_radius': None
                })
                continue
            
            # 提取最远点信息
            max_radius = result.get("max_radius")
            if max_radius is None:
                farthest_point_data.append({
                    'image_index': i,
                    'time_ms': time_ms,
                    'farthest_x': None,
                    'farthest_y': None,
                    'pixel_radius': None
                })
                continue
            
            # 解析max_radius数据
            if isinstance(max_radius, dict):
                pixel_radius = max_radius.get("value", None)
                endpoint = max_radius.get("endpoint", {})
                farthest_x = endpoint.get("x", None)
                farthest_y = endpoint.get("y", None)
            else:
                # 如果max_radius是数字，没有endpoint信息
                pixel_radius = float(max_radius) if max_radius else None
                farthest_x = None
                farthest_y = None
            
            farthest_point_data.append({
                'image_index': i,
                'time_ms': time_ms,
                'farthest_x': farthest_x,
                'farthest_y': farthest_y,
                'pixel_radius': pixel_radius
            })
        
        # 保存CSV文件
        try:
            with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['图片索引', '时间(ms)', '最远点X坐标(像素)', '最远点Y坐标(像素)', '像素半径(像素)'])
                for data in farthest_point_data:
                    writer.writerow([
                        data['image_index'],
                        f'{data["time_ms"]:.6f}',
                        f'{data["farthest_x"]:.2f}' if data["farthest_x"] is not None else '',
                        f'{data["farthest_y"]:.2f}' if data["farthest_y"] is not None else '',
                        f'{data["pixel_radius"]:.2f}' if data["pixel_radius"] is not None else ''
                    ])
            
            successful_count = sum(1 for d in farthest_point_data if d['pixel_radius'] is not None)
            print(f"✓ 最远点数据已保存: {csv_output_path}")
            print(f"  共 {total_images} 张图片，其中 {successful_count} 张包含有效的最远点数据")
            return csv_output_path
            
        except Exception as e:
            print(f"⚠️ 保存最远点CSV文件失败: {str(e)}")
            return None
        
    except Exception as e:
        print(f"⚠️ 提取最远点数据失败: {str(e)}")
        return None

