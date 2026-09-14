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
import time

try:
    from .image_io import imread_unicode
except ImportError:
    from image_io import imread_unicode

# 导入SAM初始化模块
try:
    from .sam_initialization import create_sam_manager, check_sam_availability
except ImportError:
    from sam_initialization import create_sam_manager, check_sam_availability

try:
    from .prompt_generation import (
        create_prompt_generator,
        DEFAULT_POSITIVE_POINT_COUNT,
        DEFAULT_NEGATIVE_POINT_COUNT,
    )
    from .adjacent_group_finder import create_adjacent_group_finder
    from .mask_utils import create_mask_analyzer
    from .failure_analyzer import create_failure_analyzer
    from .mask_postprocessing import create_contour_processor
    from .output_manager import create_output_manager
    from .weighted_mask import (
        create_weighted_mask_processor,
        DEFAULT_MASK_FUSION_ALPHA,
        DISPLAY_GAUSSIAN_SIGMA,
        DISPLAY_WEIGHT_THRESHOLD,
    )
except ImportError:
    from prompt_generation import (
        create_prompt_generator,
        DEFAULT_POSITIVE_POINT_COUNT,
        DEFAULT_NEGATIVE_POINT_COUNT,
    )
    from adjacent_group_finder import create_adjacent_group_finder
    from mask_utils import create_mask_analyzer
    from failure_analyzer import create_failure_analyzer
    from mask_postprocessing import create_contour_processor
    from output_manager import create_output_manager
    from weighted_mask import (
        create_weighted_mask_processor,
        DEFAULT_MASK_FUSION_ALPHA,
        DISPLAY_GAUSSIAN_SIGMA,
        DISPLAY_WEIGHT_THRESHOLD,
    )


class IterativeMaskPropagationSegmenter:
    """迭代掩码传播分割器"""

    POSITIVE_POINTS_PER_MASK = DEFAULT_POSITIVE_POINT_COUNT
    NEGATIVE_POINTS_PER_MASK = DEFAULT_NEGATIVE_POINT_COUNT
    
    def __init__(self, model_type: str = "vit_b", 
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto",
                 enable_postprocessing: bool = True,
                 fast_mode: bool = True,
                 mask_fusion_alpha: float = DEFAULT_MASK_FUSION_ALPHA,
                 display_gaussian_sigma: float = DISPLAY_GAUSSIAN_SIGMA,
                 display_weight_threshold: float = DISPLAY_WEIGHT_THRESHOLD):
        """
        初始化分割器
        
        Args:
            model_type: SAM模型类型
            checkpoint_path: 模型检查点路径
            device: 设备类型
            enable_postprocessing: 是否启用掩码后处理（轮廓过滤）
            fast_mode: 是否启用快速模式（限制正负点数量，默认启用）
        """
        if not check_sam_availability():
            raise ImportError("Segment Anything未安装，请先运行 setup.sh 安装SAM")
        
        # 创建SAM管理器
        self.sam_manager = create_sam_manager(model_type, checkpoint_path, device)
        self.enable_postprocessing = enable_postprocessing
        self.fast_mode = fast_mode
        self.mask_fusion_alpha = mask_fusion_alpha
        self.display_gaussian_sigma = display_gaussian_sigma
        self.display_weight_threshold = display_weight_threshold
        self.weighted_mask = create_weighted_mask_processor(
            alpha=mask_fusion_alpha,
            sigma=display_gaussian_sigma,
            threshold=display_weight_threshold,
        )
        
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
        
        # 存储分割结果（all_masks 为展示二值；all_mask_weights 为传播用 float 权重）
        self.all_masks = []
        self.all_mask_weights: List[Optional[np.ndarray]] = []
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
                                                  save_visualization: bool = True,
                                                  target_centre: Optional[Tuple[float, float]] = None) -> List[Optional[np.ndarray]]:
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
            target_centre: 目标质心坐标，如果提供则在计算几何信息时使用，否则自动计算质心
            
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
        self.all_mask_weights = [None] * len(image_paths)
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
        
        # 计算每个成功分割掩码的几何信息
        mask_geometries = []
        for i, mask in enumerate(self.all_masks):
            if mask is not None:
                # 优先使用后处理保存的信息
                if i in self.propagation_details and 'postprocessing_details' in self.propagation_details[i]:
                    pp_details = self.propagation_details[i]['postprocessing_details']
                    area = pp_details['area']
                    contour = pp_details['contour']
                    
                    # 如果提供了target_centre，则将其作为centroid，否则使用后处理的值
                    if target_centre is not None:
                        centroid = target_centre
                        print(f"  🔄 图片 {i+1}: 使用后处理数据计算几何信息 (面积={area}, 质心=target_centre=({centroid[0]:.1f}, {centroid[1]:.1f}))")
                    else:
                        centroid = pp_details['centroid']
                        print(f"  🔄 图片 {i+1}: 使用后处理数据计算几何信息 (面积={area}, 质心=后处理=({centroid[0]:.1f}, {centroid[1]:.1f}))")
                    
                    # 只需要计算最大半径
                    max_radius, max_radius_point = self.mask_analyzer.calculate_max_radius_with_point(mask, centroid)
                    
                    geometry = {
                        'image_index': i,
                        'area': area,
                        'centroid': centroid,
                        'max_radius': max_radius,
                        'max_radius_point': max_radius_point
                    }
                else:
                    # 回退到完整计算（兼容没有后处理的情况）
                    geometry = self.mask_analyzer.analyze_mask_geometry(mask, target_centre)
                    geometry['image_index'] = i
                
                mask_geometries.append(geometry)
            else:
                mask_geometries.append(None)
        
        return self.all_masks, mask_geometries
    
    def _init_sampling_data(self, image_paths: List[str]):
        """初始化采样点数据结构"""
        self.positive_mask_point_cache: Dict[int, List[Tuple[int, int]]] = {}
        self.propagation_details = {}
        for i in range(len(image_paths)):
            self.propagation_details[i] = {
                'reference_points': None,       # 参考图片的采样点
                'mapped_points': None,          # 映射到目标图片的点
                'filtered_points': None,        # 筛选后的点
                'next_iteration_points': None,  # 为下次迭代准备的采样点
                'cached_positive_points': None, # 掩码正点采样缓存（与 positive_mask_point_cache 同步）
                'reference_image_idx': None,    # 参考图片索引
                'iteration': None,              # 传播迭代次数
                'mask_quality': None,           # 掩码质量
                'status': 'unprocessed'         # 处理状态
            }

    def _cache_positive_mask_points(self, idx: int, mask: np.ndarray) -> List[Tuple[int, int]]:
        """从 mask 采样正点并写入缓存；每张图只采样一次。"""
        points = self.prompt_generator.sample_points_from_mask(
            mask, self.POSITIVE_POINTS_PER_MASK, inside_mask=True
        )
        self.positive_mask_point_cache[idx] = points
        if idx in self.propagation_details:
            self.propagation_details[idx]['cached_positive_points'] = points
        return points

    def _get_cached_positive_mask_points(
        self, idx: int, mask: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int]]:
        """获取已缓存的正点；若无缓存且提供了 mask 则采样并缓存。"""
        cached = self.positive_mask_point_cache.get(idx)
        if cached is not None:
            return cached
        if mask is None and idx < len(self.all_masks):
            mask = self.all_masks[idx]
        if mask is None:
            return []
        return self._cache_positive_mask_points(idx, mask)

    def _save_prompted_image_details(self, idx: int, mask: np.ndarray, prompt_info: Dict[str, Any]):
        """保存有prompt点图片的详情"""
        try:
            # 正点写入缓存；负点仍按 mask 采样供下轮传播
            next_positive_points = self._cache_positive_mask_points(idx, mask)
            next_negative_points = self.prompt_generator.sample_points_from_mask(
                mask, self.NEGATIVE_POINTS_PER_MASK, inside_mask=False
            )
            
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
    
    def _save_propagated_image_details(
        self, target_idx: int, ref_indices: List[int], debug_info: Dict[str, Any]
    ):
        """保存传播图片的初始详情（不包含最终结果）"""
        try:
            primary_idx = debug_info.get('primary_ref_idx')
            if primary_idx is None and ref_indices:
                primary_idx = ref_indices[0]
            self.propagation_details[target_idx] = {
                'reference_points': {
                    'positive': debug_info.get('reference_positive', []),
                    'negative': debug_info.get('reference_negative', []),
                },
                'mapped_points': {
                    'positive': debug_info.get('mapped_positive', []),
                    'negative': debug_info.get('mapped_negative', []),
                },
                'filtered_points': {
                    'positive': debug_info.get('final_positive', debug_info.get('filtered_positive', [])),
                    'negative': debug_info.get('final_negative', debug_info.get('filtered_negative', [])),
                },
                'next_iteration_points': None,
                'reference_image_idx': primary_idx,
                'reference_chain_indices': list(ref_indices),
                'iteration': self.current_iteration,
                'mask_quality': None,
                'status': 'processing',
            }
        except Exception as e:
            print(f"    ⚠️ 保存传播图片详情失败: {e}")
    
    def _complete_propagated_image_details(self, target_idx: int, mask: np.ndarray):
        """完成传播图片详情（添加最终结果）"""
        try:
            if target_idx in self.propagation_details:
                next_positive_points = self._cache_positive_mask_points(target_idx, mask)
                next_negative_points = self.prompt_generator.sample_points_from_mask(
                    mask, self.NEGATIVE_POINTS_PER_MASK, inside_mask=False
                )
                
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
            
            print(f"处理第 {idx+1}/{len(image_paths)} 张图片 (有prompt点): {os.path.basename(image_paths[idx])}", flush=True)
            
            # 记录开始时间
            start_time = time.time()
            
            # 分割图片
            result = self._segment_with_prompts(image_paths[idx], prompt_data[idx])
            
            if result is not None:
                masks = result['masks']
                scores = result['scores']
                
                # 选择最佳掩码（使用SAM的scores）
                best_mask, sam_quality = self._select_best_mask_with_sam_score(masks, scores)
                original_area = int(np.sum(best_mask)) if best_mask is not None else 0
                print(f"  📊 Prompt掩码: 面积={original_area}, SAM质量分数={sam_quality:.3f}")
                
                final_mask = self._commit_weighted_frame_mask(
                    best_mask, idx, sam_quality, original_area, ref_idx=None
                )
                
                # 记录结果
                self.all_masks[idx] = final_mask
                self.processed_indices.add(idx)
                
                # 保存prompt图片详情和下次迭代的采样点（使用后处理后的掩码）
                self._save_prompted_image_details(idx, final_mask, prompt_data[idx])
                
                # 计算处理时间
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"  ✓ 分割成功，选择最佳掩码 (SAM质量分数: {sam_quality:.3f})")
                print(f"  ⏱️ 处理时间: {processing_time:.2f} 秒")
                
                # 保存结果
                if output_dir and save_masks:
                    self.output_manager.save_mask_results(image_paths[idx], final_mask, idx, output_dir, save_masks, "prompted")
            else:
                # 计算处理时间（即使失败也要记录时间）
                end_time = time.time()
                processing_time = end_time - start_time
                
                self.failed_indices.add(idx)
                print(f"  ❌ 分割失败，标记为失败")
                print(f"  ⏱️ 处理时间: {processing_time:.2f} 秒")
    
    def _iterative_mask_propagation(self, image_paths: List[str], output_dir: Optional[str], 
                                   save_masks: bool, save_visualization: bool):
        """迭代掩码传播"""
        iteration = 1
        max_iterations = len(image_paths)  # 防止无限循环
        
        while len(self.processed_indices) < len(image_paths) and iteration <= max_iterations:
            # 更新当前迭代次数用于debug
            self.current_iteration = iteration
            print(f"\n--- 第 {iteration} 次迭代 ---")
            print(f"已处理: {len(self.processed_indices)}/{len(image_paths)} 张图片", flush=True)
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
                    print(f"    处理第 {unprocessed_idx+1} 张图片", flush=True)
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    print(f"      参考图片: {processed_idx + 1}")

                    mask = self._propagate_mask_from_reference(
                        image_paths[unprocessed_idx],
                        processed_idx,
                        unprocessed_idx,
                    )
                    
                    # 计算处理时间
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    if mask is not None and self.mask_analyzer.validate_mask_quality(mask):
                        self.all_masks[unprocessed_idx] = mask
                        self.processed_indices.add(unprocessed_idx)
                        processed_this_iteration += 1
                        
                        print(f"      ✓ 掩码传播成功")
                        print(f"      ⏱️ 处理时间: {processing_time:.2f} 秒")
                        
                        # 保存结果
                        if output_dir and save_masks:
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
                        print(f"      ⏱️ 处理时间: {processing_time:.2f} 秒")
            
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
            image = imread_unicode(image_path, cv2.IMREAD_COLOR)
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
            
            # 注意：用户选择的prompt点不进行drop，保持原样
            
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
                multimask_output=False,
            )
            
            # 返回masks和对应的scores
            if len(masks) > 0:
                return {'masks': masks, 'scores': scores, 'logits': logits}
            else:
                return None
            
        except Exception as e:
            print(f"    ⚠️ 分割失败: {e}")
            return None
    
    def _propagate_mask_from_reference(
        self,
        target_image_path: str,
        ref_idx: int,
        target_idx: int,
    ) -> Optional[np.ndarray]:
        """从单张参考图传播掩码到目标图。"""
        try:
            target_image = imread_unicode(target_image_path, cv2.IMREAD_COLOR)
            if target_image is None:
                print(f"    ⚠️ 无法读取目标图片: {target_image_path}")
                return None

            target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

            ref_path = self.image_paths[ref_idx]
            ref_image = imread_unicode(ref_path, cv2.IMREAD_COLOR)
            if ref_image is None:
                print(f"    ⚠️ 无法读取参考图片: {ref_path}")
                return None

            ref_mask = self.all_masks[ref_idx]
            if ref_mask is None:
                print(f"    ⚠️ 参考图 {ref_idx + 1} 掩码为空")
                return None

            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            negative_points = self.prompt_generator.sample_points_from_mask(
                ref_mask, self.NEGATIVE_POINTS_PER_MASK, inside_mask=False
            )
            predefined = {'negative': negative_points}

            result = self.prompt_generator.generate_points_with_rgb_similarity(
                ref_image_rgb,
                ref_mask,
                target_image_rgb,
                return_debug_info=True,
                predefined_reference_points=predefined,
            )
            if len(result) == 3:
                target_points, target_labels, debug_info = result
            else:
                target_points, target_labels = result
                debug_info = {}

            ref_indices = [ref_idx]
            self._save_propagated_image_details(target_idx, ref_indices, debug_info)
            
            positive_count = sum(target_labels)
            negative_count = len(target_labels) - positive_count
            print(f"    📍 生成了 {positive_count} 个正点和 {negative_count} 个负点")

            filtered_pos = debug_info.get('filtered_positive') or []
            filtered_neg = debug_info.get('filtered_negative') or []
            if debug_info.get('projection_failed') or len(filtered_pos) < 2 or len(filtered_neg) < 2:
                print(
                    f"    ❌ 投射失败（滤后正 {len(filtered_pos)} / 负 {len(filtered_neg)}），"
                    f"不进行 SAM 分割"
                )
                return None

            if positive_count < 2 or negative_count < 2:
                print(
                    f"    ❌ 最终 prompt 正负点不足（正 {positive_count} / 负 {negative_count}），"
                    f"标记为分割失败"
                )
                return None
            
            # 使用筛选后的点进行SAM分割
            self.predictor.set_image(target_image_rgb)
            
            # 准备点坐标和标签
            point_coords = np.array(target_points)
            point_labels = np.array(target_labels)
            
            # Fast模式：随机drop点，使得正点和负点个数都分别不超过3个
            if self.fast_mode:
                point_coords, point_labels = self._apply_fast_mode_drop(point_coords, point_labels)
            
            # 进行预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            
            if len(masks) > 0:
                # 选择最佳掩码（使用SAM的scores）
                best_mask, sam_quality = self._select_best_mask_with_sam_score(masks, scores)
                
                # 记录原始掩码信息
                original_area = np.sum(best_mask)
                print(f"    📊 原始掩码: 面积={original_area}, SAM质量分数={sam_quality:.3f}")
                
                final_mask = self._commit_weighted_frame_mask(
                    best_mask,
                    target_idx,
                    float(sam_quality),
                    int(original_area),
                    ref_idx=ref_idx,
                )
                
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
    
    def _apply_fast_mode_drop(self, point_coords: np.ndarray, point_labels: np.ndarray, 
                              max_points_per_label: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast模式：随机drop点，使得正点和负点个数都分别不超过指定数量
        
        Args:
            point_coords: 点坐标数组
            point_labels: 点标签数组 (1=正点, 0=负点)
            max_points_per_label: 每个标签的最大点数（默认3）
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 筛选后的点坐标和标签
        """
        if len(point_coords) == 0:
            return point_coords, point_labels
        
        # 分离正点和负点
        positive_indices = np.where(point_labels == 1)[0]
        negative_indices = np.where(point_labels == 0)[0]
        
        # 随机选择不超过max_points_per_label个点
        selected_indices = []
        
        # 处理正点
        if len(positive_indices) > max_points_per_label:
            np.random.shuffle(positive_indices)
            selected_indices.extend(positive_indices[:max_points_per_label])
        else:
            selected_indices.extend(positive_indices)
        
        # 处理负点
        if len(negative_indices) > max_points_per_label:
            np.random.shuffle(negative_indices)
            selected_indices.extend(negative_indices[:max_points_per_label])
        else:
            selected_indices.extend(negative_indices)
        
        # 按照原始顺序排序（保持相对顺序）
        selected_indices = sorted(selected_indices)
        
        # 提取筛选后的点和标签
        filtered_coords = point_coords[selected_indices]
        filtered_labels = point_labels[selected_indices]
        
        # 打印筛选信息
        original_positive = len(positive_indices)
        original_negative = len(negative_indices)
        filtered_positive = np.sum(filtered_labels == 1)
        filtered_negative = np.sum(filtered_labels == 0)
        
        if original_positive > max_points_per_label or original_negative > max_points_per_label:
            print(f"    ⚡ Fast模式: 从 {original_positive}正/{original_negative}负 筛选到 {filtered_positive}正/{filtered_negative}负 个点")
        
        return filtered_coords, filtered_labels
    
    def _reference_weights_for_fusion(self, ref_idx: int) -> Optional[np.ndarray]:
        w = None
        if ref_idx < len(self.all_mask_weights):
            w = self.all_mask_weights[ref_idx]
        if w is not None:
            return w
        ref_display = self.all_masks[ref_idx] if ref_idx < len(self.all_masks) else None
        if ref_display is not None:
            return self.weighted_mask.sam_mask_to_weights(ref_display)
        return None

    def _commit_weighted_frame_mask(
        self,
        sam_mask: np.ndarray,
        target_idx: int,
        sam_quality: float,
        original_area: int,
        ref_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        SAM 二值掩码 → 与参考权重融合 → 存 float 权重 → 高斯+阈值得展示 mask。
        """
        wm = self.weighted_mask
        p_sam = wm.sam_mask_to_weights(sam_mask)

        if ref_idx is not None:
            ref_w = self._reference_weights_for_fusion(ref_idx)
            if ref_w is not None:
                p_o = wm.resize_mask_weights(ref_w, p_sam.shape[:2])
                weights = wm.fuse_sam_with_reference_weights(
                    p_sam, p_o, alpha=self.mask_fusion_alpha
                )
                print(
                    f"    🔗 权重融合 α={self.mask_fusion_alpha:.2f}: "
                    f"SAM + 参考帧 {ref_idx + 1}"
                )
            else:
                weights = p_sam
        else:
            weights = p_sam

        if target_idx < len(self.all_mask_weights):
            self.all_mask_weights[target_idx] = weights

        if self.enable_postprocessing:
            display_mask, pp = wm.weights_to_display_mask_and_contour(
                weights,
                sigma=self.display_gaussian_sigma,
                threshold=self.display_weight_threshold,
            )
        else:
            display_mask = sam_mask.astype(bool)
            pp = {
                "area": float(np.sum(display_mask)),
                "centroid": self.mask_analyzer.calculate_mask_centroid(
                    display_mask.astype(np.uint8)
                ),
                "contour": None,
            }

        if display_mask is None:
            display_mask = np.zeros(sam_mask.shape[:2], dtype=bool)

        cleaned_area = int(np.sum(display_mask))
        area_retention = cleaned_area / original_area if original_area > 0 else 0.0
        print(
            f"    🧹 展示掩码: 面积={cleaned_area}, 保留率={area_retention:.3f} "
            f"(σ={self.display_gaussian_sigma}, thr={self.display_weight_threshold})"
        )

        if target_idx in self.propagation_details:
            self.propagation_details[target_idx]['original_mask'] = sam_mask
            self.propagation_details[target_idx]['cleaned_mask'] = display_mask
            self.propagation_details[target_idx]['mask_weights'] = weights
            self.propagation_details[target_idx]['postprocessing_stats'] = {
                'original_area': original_area,
                'cleaned_area': cleaned_area,
                'area_retention': area_retention,
                'sam_quality': sam_quality,
                'fusion_alpha': self.mask_fusion_alpha,
                'display_sigma': self.display_gaussian_sigma,
                'display_threshold': self.display_weight_threshold,
            }
            self.propagation_details[target_idx]['postprocessing_details'] = {
                'area': pp.get('area', 0.0),
                'centroid': pp.get('centroid', (0.0, 0.0)),
                'contour': pp.get('contour', None),
                'scores': {},
            }

        return display_mask
    
    


def create_iterative_segmenter(model_type: str = "vit_b", 
                               checkpoint_path: Optional[str] = None,
                               device: str = "auto",
                               enable_postprocessing: bool = True,
                               fast_mode: bool = True,
                               mask_fusion_alpha: float = DEFAULT_MASK_FUSION_ALPHA,
                               display_gaussian_sigma: float = DISPLAY_GAUSSIAN_SIGMA,
                               display_weight_threshold: float = DISPLAY_WEIGHT_THRESHOLD) -> IterativeMaskPropagationSegmenter:
    """
    创建迭代掩码传播分割器的便捷函数
    
    Args:
        model_type: SAM模型类型
        checkpoint_path: 模型检查点路径
        device: 设备类型
        enable_postprocessing: 是否启用掩码后处理（轮廓过滤）
        fast_mode: 是否启用快速模式（限制正负点数量，默认启用）
        
    Returns:
        IterativeMaskPropagationSegmenter: 分割器实例
    """
    return IterativeMaskPropagationSegmenter(
        model_type,
        checkpoint_path,
        device,
        enable_postprocessing,
        fast_mode,
        mask_fusion_alpha,
        display_gaussian_sigma,
        display_weight_threshold,
    )


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
