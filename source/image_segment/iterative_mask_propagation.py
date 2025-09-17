#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SAM的迭代掩码传播图像序列分割模块
支持部分图片有prompt点，通过迭代掩码传播处理所有图片
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

# 导入SAM初始化模块
try:
    from .sam_initialization import create_sam_manager, check_sam_availability
except ImportError:
    from sam_initialization import create_sam_manager, check_sam_availability

try:
    from .prompt_generation import create_prompt_generator
    from .adjacent_group_finder import create_adjacent_group_finder
    from .mask_utils import create_mask_analyzer, create_failure_analyzer
    from .mask_postprocessing import create_contour_processor
    from .output_manager import create_output_manager
except ImportError:
    from prompt_generation import create_prompt_generator
    from adjacent_group_finder import create_adjacent_group_finder
    from mask_utils import create_mask_analyzer, create_failure_analyzer
    from mask_postprocessing import create_contour_processor
    from output_manager import create_output_manager


class IterativeMaskPropagationSegmenter:
    """迭代掩码传播分割器"""
    
    def __init__(self, model_type: str = "vit_l", 
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto",
                 enable_postprocessing: bool = True):
        """
        初始化分割器
        
        Args:
            model_type: SAM模型类型
            checkpoint_path: 模型检查点路径
            device: 设备类型
            enable_postprocessing: 是否启用掩码后处理（轮廓过滤）
        """
        if not check_sam_availability():
            raise ImportError("Segment Anything未安装，请先运行 setup.sh 安装SAM")
        
        # 创建SAM管理器
        self.sam_manager = create_sam_manager(model_type, checkpoint_path, device)
        self.enable_postprocessing = enable_postprocessing
        
        # 保持向后兼容的属性
        self.model_type = self.sam_manager.model_type
        self.device = self.sam_manager.device
        self.checkpoint_path = self.sam_manager.checkpoint_path
        self._load_model()
        
        # 创建工具类
        self.prompt_generator = create_prompt_generator()
        self.mask_analyzer = create_mask_analyzer()
        self.failure_analyzer = create_failure_analyzer(self.mask_analyzer, self.prompt_generator)
        self.contour_processor = create_contour_processor()
        self.output_manager = create_output_manager(self.mask_analyzer)
        
        # 存储分割结果
        self.all_masks = []
        self.processed_indices = set()  # 成功处理的图片索引
        self.failed_indices = set()     # 处理失败的图片索引
        self.prompt_indices = set()     # 有prompt点的图片索引
    
    def _load_model(self):
        """加载SAM模型"""
        self.sam_manager.load_model()
        # 保持向后兼容
        self.predictor = self.sam_manager.get_predictor()
    
    def segment_sequence_with_iterative_propagation(self, 
                                                  image_paths: List[str],
                                                  prompt_data: Dict[int, Dict[str, Any]],
                                                  output_dir: Optional[str] = None,
                                                  save_masks: bool = True,
                                                  save_visualization: bool = True) -> List[Optional[np.ndarray]]:
        """
        对图像序列进行迭代掩码传播分割
        
        Args:
            image_paths: 图像文件路径列表
            prompt_data: 包含prompt信息的字典，格式为:
                {
                    image_index: {
                        'points': [(x, y), ...],  # 点坐标列表
                        'labels': [1, 0, ...],    # 点标签列表 (1=正点, 0=负点)
                        'boxes': [(x, y, w, h), ...]  # 可选：矩形prompt
                    },
                    ...
                }
            output_dir: 输出目录
            save_masks: 是否保存分割掩码
            save_visualization: 是否保存可视化结果
            
        Returns:
            List[Optional[np.ndarray]]: 每张图片的分割掩码列表，None表示未处理或处理失败
        """
        if not image_paths:
            raise ValueError("图像路径列表不能为空")
        
        print(f"\n{'='*60}")
        print(f"开始迭代掩码传播分割")
        print(f"图像序列长度: {len(image_paths)}")
        print(f"有prompt点的图片: {len(prompt_data)} 张")
        print(f"有prompt点的索引: {sorted(prompt_data.keys())}")
        print(f"{'='*60}")
        
        # 初始化结果列表
        self.all_masks = [None] * len(image_paths)
        self.processed_indices = set()
        self.failed_indices = set()
        self.prompt_indices = set(prompt_data.keys())
        
        # 初始化采样点管理
        self.image_paths = image_paths  # 保存路径供采样点管理使用
        self.current_iteration = 0
        self._init_sampling_data(image_paths)
        
        # 第一阶段：处理有prompt点的图片
        print(f"\n=== 第一阶段：处理有prompt点的图片 ===")
        self._process_prompted_images(image_paths, prompt_data, output_dir, save_masks, save_visualization)
        
        # 第二阶段：迭代掩码传播
        print(f"\n=== 第二阶段：迭代掩码传播 ===")
        self._iterative_mask_propagation(image_paths, output_dir, save_masks, save_visualization)
        
        # 输出最终统计
        self.output_manager.print_segmentation_statistics(
            self.all_masks, self.processed_indices, self.failed_indices, self.prompt_indices
        )
        
        return self.all_masks
    
    def _init_sampling_data(self, image_paths: List[str]):
        """初始化采样点数据结构"""
        self.propagation_details = {}
        for i in range(len(image_paths)):
            self.propagation_details[i] = {
                'reference_points': None,       # 参考图片的采样点
                'mapped_points': None,          # 映射到目标图片的点
                'filtered_points': None,        # 筛选后的点
                'next_iteration_points': None,  # 为下次迭代准备的采样点
                'reference_image_idx': None,    # 参考图片索引
                'iteration': None,              # 传播迭代次数
                'mask_quality': None,           # 掩码质量
                'status': 'unprocessed'         # 处理状态
            }
    
    
    def _save_prompted_image_details(self, idx: int, mask: np.ndarray, prompt_info: Dict[str, Any]):
        """保存有prompt点图片的详情"""
        try:
            # 生成下次迭代的采样点
            next_positive_points = self.prompt_generator.sample_points_from_mask(mask, 10, True)
            next_negative_points = self.prompt_generator.sample_points_from_mask(mask, 6, False)
            
            # 提取prompt点
            points = prompt_info.get('points', [])
            labels = prompt_info.get('labels', [])
            positive_points = [p for p, l in zip(points, labels) if l == 1]
            negative_points = [p for p, l in zip(points, labels) if l == 0]
            
            self.propagation_details[idx] = {
                'reference_points': None,
                'mapped_points': None,
                'filtered_points': {
                    'positive': positive_points,
                    'negative': negative_points
                },
                'next_iteration_points': {
                    'positive': next_positive_points,
                    'negative': next_negative_points
                },
                'reference_image_idx': None,
                'iteration': 0,
                'mask_quality': None,  # 不再使用自定义质量分数
                'status': 'prompted'
            }
            
        except Exception as e:
            print(f"    ⚠️ 保存prompt图片详情失败: {e}")
    
    def _save_propagated_image_details(self, target_idx: int, reference_image_path: str, debug_info: Dict[str, Any], mask: Optional[np.ndarray] = None):
        """保存传播图片的完整详情"""
        try:
            # 获取参考图片索引
            ref_idx = None
            for i, path in enumerate(self.image_paths):
                if path == reference_image_path:
                    ref_idx = i
                    break
            
            # 保存完整的传播详情
            self.propagation_details[target_idx] = {
                'reference_points': {
                    'positive': debug_info['reference_positive'],
                    'negative': debug_info['reference_negative']
                },
                'mapped_points': {
                    'positive': debug_info['mapped_positive'],
                    'negative': debug_info['mapped_negative']
                },
                'filtered_points': {
                    'positive': debug_info['filtered_positive'],
                    'negative': debug_info['filtered_negative']
                },
                'next_iteration_points': None,  # 将在分割成功后填充
                'reference_image_idx': ref_idx,
                'iteration': self.current_iteration,
                'mask_quality': None,  # 将在分割成功后填充
                'status': 'processing'
            }
            
            # 如果提供了mask，则完成详情保存
            if mask is not None:
                next_positive_points = self.prompt_generator.sample_points_from_mask(mask, 10, True)
                next_negative_points = self.prompt_generator.sample_points_from_mask(mask, 6, False)
                
                self.propagation_details[target_idx]['next_iteration_points'] = {
                    'positive': next_positive_points,
                    'negative': next_negative_points
                }
                self.propagation_details[target_idx]['mask_quality'] = None  # 不再使用自定义质量分数
                self.propagation_details[target_idx]['status'] = 'completed'
            
        except Exception as e:
            print(f"    ⚠️ 保存传播图片详情失败: {e}")
    
    def _complete_propagated_image_details(self, target_idx: int, mask: np.ndarray):
        """完成传播图片详情（添加最终结果）"""
        try:
            if target_idx in self.propagation_details:
                next_positive_points = self.prompt_generator.sample_points_from_mask(mask, 10, True)
                next_negative_points = self.prompt_generator.sample_points_from_mask(mask, 6, False)
                
                self.propagation_details[target_idx]['next_iteration_points'] = {
                    'positive': next_positive_points,
                    'negative': next_negative_points
                }
                self.propagation_details[target_idx]['mask_quality'] = None  # 不再使用自定义质量分数
                self.propagation_details[target_idx]['status'] = 'completed'
            
        except Exception as e:
            print(f"    ⚠️ 完成传播详情失败: {e}")
    
    def _process_prompted_images(self, image_paths: List[str], prompt_data: Dict[int, Dict[str, Any]], 
                                output_dir: Optional[str], save_masks: bool, save_visualization: bool):
        """处理有prompt点的图片"""
        for idx in sorted(prompt_data.keys()):
            if idx >= len(image_paths):
                print(f"⚠️ 索引 {idx} 超出图像序列范围，跳过")
                continue
            
            print(f"处理第 {idx+1}/{len(image_paths)} 张图片 (有prompt点): {os.path.basename(image_paths[idx])}")
            
            # 分割图片
            result = self._segment_with_prompts(image_paths[idx], prompt_data[idx])
            
            if result is not None:
                masks = result['masks']
                scores = result['scores']
                
                # 选择最佳掩码（使用SAM的scores）
                best_mask, sam_quality = self._select_best_mask_with_sam_score(masks, scores)
                self.all_masks[idx] = best_mask
                self.processed_indices.add(idx)
                
                # 保存prompt图片详情和下次迭代的采样点
                self._save_prompted_image_details(idx, best_mask, prompt_data[idx])
                
                print(f"  ✓ 分割成功，选择最佳掩码 (SAM质量分数: {sam_quality:.3f})")
                
                # 保存结果
                if output_dir:
                    self.output_manager.save_mask_results(image_paths[idx], best_mask, idx, output_dir, save_masks, "prompted")
            else:
                self.failed_indices.add(idx)
                print(f"  ❌ 分割失败，标记为失败")
    
    def _iterative_mask_propagation(self, image_paths: List[str], output_dir: Optional[str], 
                                   save_masks: bool, save_visualization: bool):
        """迭代掩码传播"""
        iteration = 1
        max_iterations = len(image_paths)  # 防止无限循环
        
        while len(self.processed_indices) < len(image_paths) and iteration <= max_iterations:
            # 更新当前迭代次数用于debug
            self.current_iteration = iteration
            print(f"\n--- 第 {iteration} 次迭代 ---")
            print(f"已处理: {len(self.processed_indices)}/{len(image_paths)} 张图片")
            print(f"已处理索引: {sorted(self.processed_indices)}")
            print(f"已失败索引: {sorted(self.failed_indices)}")
            
            # 找到相邻图片组
            group_finder = create_adjacent_group_finder(self.processed_indices, self.failed_indices)
            adjacent_groups = group_finder.find_adjacent_groups(len(image_paths))
            
            if not adjacent_groups:
                print("  没有相邻的未处理图片，停止迭代")
                break
            
            print(f"  找到 {len(adjacent_groups)} 个相邻图片组")
            for group in adjacent_groups:
                print(f"    组 {group['group_id']}: 参考图片 {group['processed']+1}, 目标图片 {[idx+1 for idx in group['unprocessed']]}")
            
            # 处理每个相邻组
            processed_this_iteration = 0
            failed_this_iteration = 0
            
            for group in adjacent_groups:
                processed_idx = group['processed']
                unprocessed_indices = group['unprocessed']
                
                print(f"  处理组 {group['group_id']}: 参考图片 {processed_idx+1}, 目标图片 {[idx+1 for idx in unprocessed_indices]}")
                
                # 获取参考掩码
                reference_mask = self.all_masks[processed_idx]
                if reference_mask is None:
                    print(f"    ❌ 参考图片 {processed_idx+1} 的掩码为空，跳过该组")
                    continue
                
                # 处理该组中的所有未处理图片
                for unprocessed_idx in unprocessed_indices:
                    print(f"    处理第 {unprocessed_idx+1} 张图片")
                    
                    # 获取预定义的采样点
                    predefined_points = None
                    if (processed_idx in self.propagation_details and 
                        'next_iteration_points' in self.propagation_details[processed_idx]):
                        predefined_points = self.propagation_details[processed_idx]['next_iteration_points']
                        print(f"      使用预定义的采样点")
                    
                    # 使用掩码传播进行分割
                    mask = self._propagate_mask_from_reference(
                        image_paths[unprocessed_idx], 
                        image_paths[processed_idx],
                        reference_mask,
                        unprocessed_idx,
                        predefined_points
                    )
                    
                    if mask is not None and self.mask_analyzer.validate_mask_quality(mask):
                        self.all_masks[unprocessed_idx] = mask
                        self.processed_indices.add(unprocessed_idx)
                        processed_this_iteration += 1
                        
                        print(f"      ✓ 掩码传播成功")
                        
                        # 保存结果
                        if output_dir:
                            self.output_manager.save_mask_results(image_paths[unprocessed_idx], mask, 
                                                                unprocessed_idx, output_dir, save_masks, "propagated")
                    else:
                        self.failed_indices.add(unprocessed_idx)
                        failed_this_iteration += 1
                        
                        # 详细分析失败原因
                        failure_reason = self.failure_analyzer.analyze_failure(
                            image_paths[unprocessed_idx], 
                            image_paths[processed_idx],
                            self.all_masks[processed_idx],
                            mask
                        )
                        print(f"      ❌ 掩码传播失败: {failure_reason}")
            
            print(f"  本次迭代处理了 {processed_this_iteration} 张图片，失败 {failed_this_iteration} 张图片")
            
            if processed_this_iteration == 0:
                print("  本次迭代没有处理任何图片，停止迭代")
                break
            
            iteration += 1
        
        if iteration > max_iterations:
            print(f"⚠️ 达到最大迭代次数 {max_iterations}，停止传播")
    
    def _select_best_mask_with_sam_score(self, masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用SAM的原生质量分数选择最佳掩码
        
        Args:
            masks: SAM返回的掩码数组
            scores: SAM返回的质量分数数组
            
        Returns:
            Tuple[np.ndarray, float]: (最佳掩码, SAM质量分数)
        """
        if len(masks) == 0:
            return None, 0.0
        
        if len(masks) == 1:
            return masks[0], float(scores[0])
        
        # 使用SAM的scores选择最佳掩码
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        sam_quality = float(scores[best_idx])
        
        print(f"    📊 SAM生成{len(masks)}个掩码，选择质量最高的（SAM分数: {sam_quality:.3f}）")
        
        return best_mask, sam_quality
    
    def _segment_with_prompts(self, image_path: str, prompt_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """使用prompt点进行分割"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            
            # 准备prompt数据
            points = prompt_info.get('points', [])
            labels = prompt_info.get('labels', [])
            boxes = prompt_info.get('boxes', [])
            
            point_coords = np.array(points) if points else None
            point_labels = np.array(labels) if labels else None
            
            # 转换矩形prompt
            input_boxes = None
            if boxes:
                input_boxes = []
                for x, y, w, h in boxes:
                    input_boxes.append([x, y, x + w, y + h])
                input_boxes = np.array(input_boxes)
            
            # 进行预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_boxes,
                multimask_output=True,
            )
            
            # 返回masks和对应的scores
            if len(masks) > 0:
                return {'masks': masks, 'scores': scores, 'logits': logits}
            else:
                return None
            
        except Exception as e:
            print(f"    ⚠️ 分割失败: {e}")
            return None
    
    def _propagate_mask_from_reference(self, target_image_path: str, reference_image_path: str, 
                                     reference_mask: np.ndarray, target_idx: int, 
                                     predefined_reference_points: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> Optional[np.ndarray]:
        """从参考图片传播掩码到目标图片"""
        try:
            # 读取目标图片
            target_image = cv2.imread(target_image_path)
            if target_image is None:
                print(f"    ⚠️ 无法读取目标图片: {target_image_path}")
                return None
            
            target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            
            # 读取参考图片
            reference_image = cv2.imread(reference_image_path)
            if reference_image is None:
                print(f"    ⚠️ 无法读取参考图片: {reference_image_path}")
                return None
            
            reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            
            # 使用基于RGB相似性的点映射和筛选
            target_points, target_labels, debug_info = self.prompt_generator.generate_points_with_rgb_similarity(
                reference_image_rgb, reference_mask, target_image_rgb,
                return_debug_info=True, predefined_reference_points=predefined_reference_points
            )
            
            # 保存传播详情（不包含最终结果）
            self._save_propagated_image_details(target_idx, reference_image_path, debug_info)
            
            if len(target_points) == 0:
                print(f"    ⚠️ 未找到有效的正负点，跳过")
                return None
            
            # 记录点生成信息
            positive_count = sum(target_labels)
            negative_count = len(target_labels) - positive_count
            print(f"    📍 生成了 {positive_count} 个正点和 {negative_count} 个负点")
            
            # 使用筛选后的点进行SAM分割
            self.predictor.set_image(target_image_rgb)
            
            # 准备点坐标和标签
            point_coords = np.array(target_points)
            point_labels = np.array(target_labels)
            
            # 进行预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            
            if len(masks) > 0:
                # 选择最佳掩码（使用SAM的scores）
                best_mask, sam_quality = self._select_best_mask_with_sam_score(masks, scores)
                
                # 记录原始掩码信息
                original_area = np.sum(best_mask)
                print(f"    📊 原始掩码: 面积={original_area}, SAM质量分数={sam_quality:.3f}")
                
                # 可选的后处理：基于轮廓特征过滤碎片
                if self.enable_postprocessing:
                    cleaned_mask = self.contour_processor.filter_mask_by_contour_quality(best_mask, "fireball_optimized")
                    
                    # 记录清理后的信息
                    cleaned_area = np.sum(cleaned_mask)
                    area_retention = cleaned_area / original_area if original_area > 0 else 0
                    
                    print(f"    🧹 清理后掩码: 面积={cleaned_area}, 保留率={area_retention:.3f}")
                    
                    # 保存后处理对比信息到传播详情
                    if target_idx in self.propagation_details:
                        self.propagation_details[target_idx]['original_mask'] = best_mask
                        self.propagation_details[target_idx]['cleaned_mask'] = cleaned_mask
                        self.propagation_details[target_idx]['postprocessing_stats'] = {
                            'original_area': original_area,
                            'cleaned_area': cleaned_area,
                            'area_retention': area_retention,
                            'sam_quality': sam_quality  # SAM原生质量分数
                        }
                    
                    final_mask = cleaned_mask
                else:
                    print(f"    ⚡ 跳过后处理，直接使用原始掩码")
                    
                    # 不进行后处理时的信息保存
                    if target_idx in self.propagation_details:
                        self.propagation_details[target_idx]['original_mask'] = best_mask
                        self.propagation_details[target_idx]['cleaned_mask'] = best_mask  # 与原始掩码相同
                        self.propagation_details[target_idx]['postprocessing_stats'] = {
                            'original_area': original_area,
                            'cleaned_area': original_area,  # 未清理，面积相同
                            'area_retention': 1.0,  # 100%保留
                            'sam_quality': sam_quality
                        }
                    
                    final_mask = best_mask
                
                # 完成传播详情保存（使用最终掩码）
                self._complete_propagated_image_details(target_idx, final_mask)
                
                return final_mask
            else:
                print(f"    ⚠️ SAM未生成任何掩码")
                return None
            
        except Exception as e:
            print(f"    ⚠️ 掩码传播异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    


def create_iterative_segmenter(model_type: str = "vit_l", 
                               checkpoint_path: Optional[str] = None,
                               device: str = "auto",
                               enable_postprocessing: bool = True) -> IterativeMaskPropagationSegmenter:
    """
    创建迭代掩码传播分割器的便捷函数
    
    Args:
        model_type: SAM模型类型
        checkpoint_path: 模型检查点路径
        device: 设备类型
        enable_postprocessing: 是否启用掩码后处理（轮廓过滤）
        
    Returns:
        IterativeMaskPropagationSegmenter: 分割器实例
    """
    return IterativeMaskPropagationSegmenter(model_type, checkpoint_path, device, enable_postprocessing)


if __name__ == "__main__":
    # 示例用法
    print("迭代掩码传播图像序列分割模块")
    print("使用方法:")
    print("1. 创建分割器: segmenter = create_iterative_segmenter()")
    print("2. 准备prompt数据:")
    print("   prompt_data = {")
    print("       0: {'points': [(400, 300)], 'labels': [1]},  # 第1张图片有prompt")
    print("       2: {'points': [(420, 320)], 'labels': [1]},  # 第3张图片有prompt")
    print("   }")
    print("3. 分割序列: masks = segmenter.segment_sequence_with_iterative_propagation(image_paths, prompt_data)")
