#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出管理模块
包含分割结果的保存和统计信息输出功能
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Set, Any, Dict, Tuple


class SegmentationOutputManager:
    """分割结果输出管理器"""
    
    def __init__(self, mask_analyzer=None):
        """
        初始化输出管理器
        
        Args:
            mask_analyzer: 掩码分析器，用于计算统计信息
        """
        self.mask_analyzer = mask_analyzer
    
    def save_mask_results(self, image_path: str, mask: np.ndarray, image_idx: int, 
                         output_dir: str, save_masks: bool, prefix: str = ""):
        """
        保存分割掩码文件
        
        Args:
            image_path: 原始图像路径
            mask: 分割掩码
            image_idx: 图像索引
            output_dir: 输出目录
            save_masks: 是否保存掩码文件
            prefix: 文件名前缀
        """
        if not save_masks:
            return
        
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            
            # 创建掩码子目录
            mask_dir = os.path.join(output_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            
            # 生成掩码文件名
            if prefix:
                mask_filename = f"{base_name}_{prefix}_mask.png"
            else:
                mask_filename = f"{base_name}_mask.png"
            
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # 保存掩码（转换为0-255的灰度图）
            mask_image = (mask * 255).astype(np.uint8)
            success = cv2.imwrite(mask_path, mask_image)
            
            if success:
                print(f"    💾 掩码已保存: {mask_filename}")
            else:
                print(f"    ⚠️ 掩码保存失败: {mask_filename}")
                
        except Exception as e:
            print(f"    ❌ 保存掩码时出错: {e}")
    
    def save_batch_masks(self, image_paths: List[str], masks: List[Optional[np.ndarray]], 
                        output_dir: str, save_masks: bool = True, prefix: str = ""):
        """
        批量保存掩码文件
        
        Args:
            image_paths: 图像路径列表
            masks: 掩码列表
            output_dir: 输出目录
            save_masks: 是否保存掩码文件
            prefix: 文件名前缀
        """
        if not save_masks:
            return
        
        print(f"\n📁 批量保存掩码到: {output_dir}")
        saved_count = 0
        
        for i, (image_path, mask) in enumerate(zip(image_paths, masks)):
            if mask is not None:
                self.save_mask_results(image_path, mask, i, output_dir, True, prefix)
                saved_count += 1
        
        print(f"✅ 批量保存完成: {saved_count}/{len(masks)} 个掩码已保存")
    
    def print_segmentation_statistics(self, all_masks: List[Optional[np.ndarray]], 
                                    processed_indices: Set[int], 
                                    failed_indices: Set[int],
                                    prompt_indices: Set[int] = None):
        """
        打印分割统计信息
        
        Args:
            all_masks: 所有掩码列表
            processed_indices: 已处理图片索引集合
            failed_indices: 处理失败图片索引集合
            prompt_indices: 有prompt点的图片索引集合
        """
        print(f"\n{'='*60}")
        print(f"分割完成统计")
        print(f"{'='*60}")
        
        # 基本统计
        total_images = len(all_masks)
        processed_images = len(processed_indices)
        failed_images = len(failed_indices)
        successful_images = sum(1 for mask in all_masks if mask is not None)
        
        print(f"📊 基本统计:")
        print(f"   总图片数: {total_images}")
        print(f"   已处理图片数: {processed_images}")
        print(f"   处理失败图片数: {failed_images}")
        print(f"   成功分割图片数: {successful_images}")
        print(f"   处理成功率: {successful_images/total_images*100:.1f}%")
        
        # Prompt统计
        if prompt_indices:
            prompt_count = len(prompt_indices)
            propagated_count = successful_images - prompt_count
            print(f"\n🎯 传播统计:")
            print(f"   Prompt图片数: {prompt_count}")
            print(f"   传播成功图片数: {propagated_count}")
            if prompt_count > 0:
                propagation_rate = propagated_count / (total_images - prompt_count) * 100 if total_images > prompt_count else 0
                print(f"   传播成功率: {propagation_rate:.1f}%")
        
        # 失败统计
        if failed_images > 0:
            print(f"\n❌ 失败详情:")
            print(f"   失败图片索引: {sorted(failed_indices)}")
        
        # 掩码质量统计
        if successful_images > 0:
            print(f"\n📏 掩码统计:")
            
            # 面积统计
            if self.mask_analyzer:
                areas = [self.mask_analyzer.calculate_mask_area(mask) 
                        for mask in all_masks if mask is not None]
            else:
                areas = [int(np.sum(mask)) for mask in all_masks if mask is not None]
            
            if areas:
                avg_area = np.mean(areas)
                min_area = min(areas)
                max_area = max(areas)
                std_area = np.std(areas)
                
                print(f"   平均掩码面积: {avg_area:.0f} 像素")
                print(f"   面积范围: {min_area} - {max_area} 像素")
                print(f"   面积标准差: {std_area:.0f} 像素")
        
        print(f"{'='*60}")
    
    def print_processing_summary(self, stage: str, processed_count: int, 
                               total_count: int, details: str = ""):
        """
        打印处理阶段摘要
        
        Args:
            stage: 处理阶段名称
            processed_count: 已处理数量
            total_count: 总数量
            details: 额外详情
        """
        percentage = processed_count / total_count * 100 if total_count > 0 else 0
        print(f"\n📋 {stage}:")
        print(f"   进度: {processed_count}/{total_count} ({percentage:.1f}%)")
        if details:
            print(f"   详情: {details}")
    
    def save_processing_log(self, output_dir: str, log_data: dict):
        """
        保存处理日志到文件
        
        Args:
            output_dir: 输出目录
            log_data: 日志数据字典
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, "processing_log.txt")
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("图像序列分割处理日志\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in log_data.items():
                    if isinstance(value, (list, set)):
                        f.write(f"{key}: {list(value)}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                f.write(f"\n生成时间: {Path().resolve()}\n")
            
            print(f"📄 处理日志已保存: {log_path}")
            
        except Exception as e:
            print(f"⚠️ 保存日志失败: {e}")
    
    def create_results_summary(self, image_paths: List[str], 
                              masks: List[Optional[np.ndarray]],
                              processing_details: dict = None) -> dict:
        """
        创建结果摘要
        
        Args:
            image_paths: 图像路径列表
            masks: 掩码列表
            processing_details: 处理详情
            
        Returns:
            dict: 结果摘要字典
        """
        summary = {
            "total_images": len(image_paths),
            "successful_masks": sum(1 for mask in masks if mask is not None),
            "failed_masks": sum(1 for mask in masks if mask is None),
            "image_files": [Path(p).name for p in image_paths],
            "success_indices": [i for i, mask in enumerate(masks) if mask is not None],
            "failed_indices": [i for i, mask in enumerate(masks) if mask is None]
        }
        
        # 添加掩码面积信息
        if self.mask_analyzer:
            areas = [self.mask_analyzer.calculate_mask_area(mask) 
                    for mask in masks if mask is not None]
        else:
            areas = [int(np.sum(mask)) for mask in masks if mask is not None]
        
        if areas:
            summary["mask_areas"] = areas
            summary["average_area"] = float(np.mean(areas))
            summary["area_std"] = float(np.std(areas))
        
        # 添加处理详情
        if processing_details:
            summary.update(processing_details)
        
        return summary
    
    def extract_contour_from_mask(self, mask: np.ndarray) -> Optional[List[List[Tuple[int, int]]]]:
        """
        从掩码中提取轮廓坐标
        
        Args:
            mask: 输入掩码
            
        Returns:
            List[List[Tuple[int, int]]]: 轮廓点列表，每个轮廓是点坐标的列表
        """
        if mask is None or np.sum(mask) == 0:
            return None
        
        try:
            # 转换为uint8格式
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 转换轮廓格式：从numpy数组转换为坐标点列表
            contour_list = []
            for contour in contours:
                # 将轮廓点转换为 (x, y) 坐标列表
                points = [(int(point[0][0]), int(point[0][1])) for point in contour]
                contour_list.append(points)
            
            return contour_list
            
        except Exception as e:
            print(f"    ⚠️ 轮廓提取失败: {e}")
            return None
    
    def export_segmentation_results_to_json(self, json_path: str, image_paths: List[str], 
                                          masks: List[Optional[np.ndarray]], 
                                          geometries: List[Optional[Dict[str, Any]]]) -> bool:
        """
        将分割结果导出到原JSON文件中
        
        Args:
            json_path: 原JSON文件路径
            image_paths: 图像路径列表
            masks: 掩码列表
            geometries: 几何信息列表
            
        Returns:
            bool: 是否成功导出
        """
        try:
            # 读取原始JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建分割结果数据结构
            segmentation_results = []
            
            for i, (image_path, mask, geometry) in enumerate(zip(image_paths, masks, geometries)):
                result_item = {
                    "success": mask is not None
                }
                
                if mask is not None and geometry is not None:
                    # 提取轮廓
                    contours = self.extract_contour_from_mask(mask)
                    
                    # 获取质心坐标
                    centroid = geometry.get('centroid', (0.0, 0.0))
                    
                    # 获取最大半径信息
                    max_radius = geometry.get('max_radius', 0.0)
                    max_radius_point = geometry.get('max_radius_point', (0.0, 0.0))
                    
                    result_item.update({
                        "contours": contours,
                        "centroid": {
                            "x": float(centroid[0]),
                            "y": float(centroid[1])
                        },
                        "max_radius": {
                            "value": float(max_radius),
                            "endpoint": {
                                "x": float(max_radius_point[0]),
                                "y": float(max_radius_point[1])
                            }
                        }
                    })
                else:
                    # 分割失败的情况
                    result_item.update({
                        "contours": None,
                        "centroid": None,
                        "max_radius": None
                    })
                
                segmentation_results.append(result_item)
            
            # 添加分割结果到JSON数据中
            data["image_sequence_segmentation"] = segmentation_results
            
            # 导出文件名：{序列stem}_{当量}_{含铝}_segmented.json（与训练数据命名一致）
            try:
                from export_naming import segmented_sequence_filename_from_data
            except ImportError:
                import sys
                _src = Path(__file__).resolve().parent.parent
                if str(_src) not in sys.path:
                    sys.path.insert(0, str(_src))
                from export_naming import segmented_sequence_filename_from_data

            export_path = segmented_sequence_filename_from_data(json_path, data)
            
            # 保存为新文件，避免覆盖原始序列文件
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 分割结果已导出到: {export_path}")
            successful_count = sum(1 for result in segmentation_results if result['success'])
            print(f"   成功分割: {successful_count}/{len(segmentation_results)} 张图片")
            
            return True
            
        except Exception as e:
            print(f"❌ 导出分割结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_output_manager(mask_analyzer=None) -> SegmentationOutputManager:
    """
    创建输出管理器的便捷函数
    
    Args:
        mask_analyzer: 掩码分析器
        
    Returns:
        SegmentationOutputManager: 输出管理器实例
    """
    return SegmentationOutputManager(mask_analyzer)


def save_masks_simple(image_paths: List[str], masks: List[Optional[np.ndarray]], 
                     output_dir: str, prefix: str = "") -> int:
    """
    简单的掩码保存函数
    
    Args:
        image_paths: 图像路径列表
        masks: 掩码列表
        output_dir: 输出目录
        prefix: 文件名前缀
        
    Returns:
        int: 成功保存的掩码数量
    """
    manager = create_output_manager()
    manager.save_batch_masks(image_paths, masks, output_dir, True, prefix)
    return sum(1 for mask in masks if mask is not None)


def print_simple_statistics(masks: List[Optional[np.ndarray]]):
    """
    打印简单统计信息
    
    Args:
        masks: 掩码列表
    """
    total = len(masks)
    successful = sum(1 for mask in masks if mask is not None)
    failed = total - successful
    
    print(f"\n📊 分割结果统计:")
    print(f"   总计: {total} 张图片")
    print(f"   成功: {successful} 张 ({successful/total*100:.1f}%)")
    print(f"   失败: {failed} 张 ({failed/total*100:.1f}%)")


if __name__ == "__main__":
    # 测试模块功能
    print("输出管理模块测试")
    print("=" * 40)
    
    # 创建测试数据
    test_masks = [
        np.ones((100, 100), dtype=np.uint8) * 255,  # 成功掩码
        None,  # 失败
        np.ones((150, 150), dtype=np.uint8) * 255,  # 成功掩码
        None   # 失败
    ]
    
    test_paths = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
    
    # 测试输出管理器
    manager = create_output_manager()
    
    print("1. 测试统计信息:")
    processed_indices = {0, 2}
    failed_indices = {1, 3}
    prompt_indices = {0}
    
    manager.print_segmentation_statistics(
        test_masks, processed_indices, failed_indices, prompt_indices
    )
    
    print("\n2. 测试简单统计:")
    print_simple_statistics(test_masks)
    
    print("\n3. 测试结果摘要:")
    summary = manager.create_results_summary(test_paths, test_masks)
    print("   摘要字段:", list(summary.keys()))
    
    print("\n✅ 测试完成!")
