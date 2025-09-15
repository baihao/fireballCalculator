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
from scipy import ndimage
from scipy.interpolate import interp1d

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("警告: Segment Anything未安装，请先运行 setup.sh 安装SAM")

try:
    from .prompt_generation import create_prompt_generator
except ImportError:
    from prompt_generation import create_prompt_generator


class IterativeMaskPropagationSegmenter:
    """迭代掩码传播分割器"""
    
    def __init__(self, model_type: str = "vit_l", 
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto"):
        """
        初始化分割器
        
        Args:
            model_type: SAM模型类型
            checkpoint_path: 模型检查点路径
            device: 设备类型
        """
        if not SAM_AVAILABLE:
            raise ImportError("Segment Anything未安装，请先运行 setup.sh 安装SAM")
        
        self.model_type = model_type
        self.device = self._get_device(device)
        self.predictor = None
        
        # 设置默认检查点路径
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path()
        
        self.checkpoint_path = checkpoint_path
        self._load_model()
        
        # 创建prompt点生成器
        self.prompt_generator = create_prompt_generator()
        
        # 存储分割结果
        self.all_masks = []
        self.processed_indices = set()  # 成功处理的图片索引
        self.failed_indices = set()     # 处理失败的图片索引
        self.prompt_indices = set()     # 有prompt点的图片索引
    
    def _get_device(self, device: str) -> str:
        """获取可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_default_checkpoint_path(self) -> str:
        """获取默认检查点路径"""
        sam_dir = Path(__file__).parent.parent / "third_party" / "segment-anything"
        checkpoint_dir = sam_dir / "checkpoints"
        
        checkpoint_files = [
            f"sam_{self.model_type}_0b3195.pth",
            f"sam_{self.model_type}.pth",
            f"sam_{self.model_type}_01ec64.pth"
        ]
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = checkpoint_dir / checkpoint_file
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        return str(checkpoint_dir / f"sam_{self.model_type}.pth")
    
    def _load_model(self):
        """加载SAM模型"""
        try:
            print(f"正在加载SAM模型: {self.model_type}")
            print(f"设备: {self.device}")
            
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            
            print("✓ SAM模型加载成功")
            
        except Exception as e:
            print(f"❌ SAM模型加载失败: {e}")
            raise
    
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
        
        # 第一阶段：处理有prompt点的图片
        print(f"\n=== 第一阶段：处理有prompt点的图片 ===")
        self._process_prompted_images(image_paths, prompt_data, output_dir, save_masks, save_visualization)
        
        # 第二阶段：迭代掩码传播
        print(f"\n=== 第二阶段：迭代掩码传播 ===")
        self._iterative_mask_propagation(image_paths, output_dir, save_masks, save_visualization)
        
        # 输出最终统计
        self._print_final_statistics()
        
        return self.all_masks
    
    def _process_prompted_images(self, image_paths: List[str], prompt_data: Dict[int, Dict[str, Any]], 
                                output_dir: Optional[str], save_masks: bool, save_visualization: bool):
        """处理有prompt点的图片"""
        for idx in sorted(prompt_data.keys()):
            if idx >= len(image_paths):
                print(f"⚠️ 索引 {idx} 超出图像序列范围，跳过")
                continue
            
            print(f"处理第 {idx+1}/{len(image_paths)} 张图片 (有prompt点): {os.path.basename(image_paths[idx])}")
            
            # 分割图片
            masks = self._segment_with_prompts(image_paths[idx], prompt_data[idx])
            
            if masks is not None:
                # 选择最佳掩码
                best_mask = self._select_best_mask(masks)
                self.all_masks[idx] = best_mask
                self.processed_indices.add(idx)
                
                print(f"  ✓ 分割成功，选择最佳掩码 (质量分数: {self._calculate_mask_quality(best_mask):.3f})")
                
                # 保存结果
                if output_dir:
                    self._save_results(image_paths[idx], best_mask, idx, output_dir, save_masks, save_visualization, "prompted")
            else:
                self.failed_indices.add(idx)
                print(f"  ❌ 分割失败，标记为失败")
    
    def _iterative_mask_propagation(self, image_paths: List[str], output_dir: Optional[str], 
                                   save_masks: bool, save_visualization: bool):
        """迭代掩码传播"""
        iteration = 1
        max_iterations = len(image_paths)  # 防止无限循环
        
        while len(self.processed_indices) < len(image_paths) and iteration <= max_iterations:
            print(f"\n--- 第 {iteration} 次迭代 ---")
            print(f"已处理: {len(self.processed_indices)}/{len(image_paths)} 张图片")
            print(f"已处理索引: {sorted(self.processed_indices)}")
            print(f"已失败索引: {sorted(self.failed_indices)}")
            
            # 找到相邻图片组
            adjacent_groups = self._find_adjacent_groups(len(image_paths))
            
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
                    
                    # 使用掩码传播进行分割
                    mask = self._propagate_mask_from_reference(
                        image_paths[unprocessed_idx], 
                        image_paths[processed_idx],
                        reference_mask,
                        unprocessed_idx
                    )
                    
                    if mask is not None and self._validate_mask_quality(mask):
                        self.all_masks[unprocessed_idx] = mask
                        self.processed_indices.add(unprocessed_idx)
                        processed_this_iteration += 1
                        
                        print(f"      ✓ 掩码传播成功 (质量分数: {self._calculate_mask_quality(mask):.3f})")
                        
                        # 保存结果
                        if output_dir:
                            self._save_results(image_paths[unprocessed_idx], mask, unprocessed_idx, 
                                             output_dir, save_masks, save_visualization, "propagated")
                    else:
                        self.failed_indices.add(unprocessed_idx)
                        failed_this_iteration += 1
                        
                        # 详细分析失败原因
                        failure_reason = self._analyze_propagation_failure(
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
    
    def _find_adjacent_groups(self, total_images: int) -> List[Dict[str, Any]]:
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
    
    def _find_next_available_index(self, start_idx: int, direction: int, total_images: int, used_unprocessed: set) -> Optional[int]:
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
            
            return masks if len(masks) > 0 else None
            
        except Exception as e:
            print(f"    ⚠️ 分割失败: {e}")
            return None
    
    def _propagate_mask_from_reference(self, target_image_path: str, reference_image_path: str, 
                                     reference_mask: np.ndarray, target_idx: int) -> Optional[np.ndarray]:
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
            target_points, target_labels = self.prompt_generator.generate_points_with_rgb_similarity(
                reference_image_rgb, reference_mask, target_image_rgb
            )
            
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
                # 选择最佳掩码
                best_mask = self._select_best_mask(masks)
                
                # 记录掩码信息
                mask_area = np.sum(best_mask)
                mask_quality = self._calculate_mask_quality(best_mask)
                print(f"    📊 生成掩码: 面积={mask_area}, 质量分数={mask_quality:.3f}")
                
                return best_mask
            else:
                print(f"    ⚠️ SAM未生成任何掩码")
                return None
            
        except Exception as e:
            print(f"    ⚠️ 掩码传播异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_propagation_failure(self, target_image_path: str, reference_image_path: str, 
                                   reference_mask: np.ndarray, result_mask: Optional[np.ndarray]) -> str:
        """
        分析掩码传播失败的原因
        
        Args:
            target_image_path: 目标图片路径
            reference_image_path: 参考图片路径
            reference_mask: 参考掩码
            result_mask: 传播结果掩码（可能为None）
            
        Returns:
            str: 失败原因描述
        """
        try:
            reasons = []
            
            # 1. 检查结果掩码是否为空
            if result_mask is None:
                reasons.append("结果掩码为空")
            else:
                # 2. 检查掩码质量
                if not self._validate_mask_quality(result_mask):
                    quality_score = self._calculate_mask_quality(result_mask)
                    reasons.append(f"掩码质量不佳 (质量分数: {quality_score:.3f})")
                
                # 3. 检查掩码面积
                if result_mask is not None:
                    area = np.sum(result_mask)
                    ref_area = np.sum(reference_mask)
                    area_ratio = area / ref_area if ref_area > 0 else 0
                    
                    if area_ratio < 0.1:
                        reasons.append(f"掩码面积过小 (面积比: {area_ratio:.3f})")
                    elif area_ratio > 5.0:
                        reasons.append(f"掩码面积过大 (面积比: {area_ratio:.3f})")
            
            # 4. 检查图像差异
            try:
                target_image = cv2.imread(target_image_path)
                reference_image = cv2.imread(reference_image_path)
                
                if target_image is not None and reference_image is not None:
                    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                    reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                    
                    # 计算图像差异
                    diff = cv2.absdiff(target_rgb, reference_rgb)
                    mean_diff = np.mean(diff) / 255.0
                    
                    if mean_diff > 0.5:
                        reasons.append(f"图像差异过大 (差异度: {mean_diff:.3f})")
                    
                    # 5. 检查点生成情况
                    points, labels = self.prompt_generator.generate_points_with_rgb_similarity(
                        reference_rgb, reference_mask, target_rgb
                    )
                    
                    positive_points = [p for p, l in zip(points, labels) if l == 1]
                    negative_points = [p for p, l in zip(points, labels) if l == 0]
                    
                    if len(positive_points) == 0:
                        reasons.append("未生成有效正点")
                    elif len(positive_points) < 3:
                        reasons.append(f"正点数量过少 ({len(positive_points)}个)")
                    
                    if len(negative_points) == 0:
                        reasons.append("未生成负点")
                    
                    # 6. 检查RGB相似性
                    if len(positive_points) > 0:
                        # 计算正点的RGB相似性
                        ref_positive_rgbs = [reference_rgb[y, x] for x, y in 
                                           self.prompt_generator.sample_points_from_mask(reference_mask, 10, True)]
                        
                        similar_count = 0
                        for x, y in positive_points:
                            target_rgb_point = target_rgb[y, x]
                            for ref_rgb in ref_positive_rgbs:
                                if self.prompt_generator.is_rgb_very_similar(target_rgb_point, ref_rgb):
                                    similar_count += 1
                                    break
                        
                        similarity_ratio = similar_count / len(positive_points) if len(positive_points) > 0 else 0
                        if similarity_ratio < 0.3:
                            reasons.append(f"RGB相似性过低 (相似度: {similarity_ratio:.3f})")
                
            except Exception as e:
                reasons.append(f"图像分析失败: {str(e)}")
            
            # 7. 检查参考掩码质量
            ref_quality = self._calculate_mask_quality(reference_mask)
            if ref_quality < 0.5:
                reasons.append(f"参考掩码质量不佳 (质量分数: {ref_quality:.3f})")
            
            # 返回失败原因
            if not reasons:
                return "未知原因"
            else:
                return "; ".join(reasons)
                
        except Exception as e:
            return f"分析过程出错: {str(e)}"

    def _propagate_mask_with_optical_flow(self, reference_image: np.ndarray, reference_mask: np.ndarray, 
                                        target_image: np.ndarray) -> Optional[np.ndarray]:
        """使用光流传播掩码"""
        try:
            # 转换为灰度图
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowPyrLK(
                ref_gray, target_gray, 
                None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # 简化版本：使用简单的仿射变换
            # 在实际应用中，这里应该使用更复杂的光流算法
            h, w = reference_mask.shape
            target_h, target_w = target_image.shape[:2]
            
            # 调整掩码尺寸
            if (h, w) != (target_h, target_w):
                propagated_mask = cv2.resize(reference_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                propagated_mask = reference_mask.copy()
            
            return propagated_mask
            
        except Exception as e:
            print(f"    ⚠️ 光流传播失败: {e}")
            return None
    
    def _prepare_mask_input(self, mask: np.ndarray) -> np.ndarray:
        """准备掩码输入，调整到SAM要求的尺寸"""
        # 调整到256x256
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # 添加batch维度
        mask_input = mask_resized[None, :, :]
        
        return mask_input
    
    def _select_best_mask(self, masks: np.ndarray) -> np.ndarray:
        """选择最佳掩码"""
        if len(masks) == 1:
            return masks[0]
        
        # 简单的选择策略：选择面积最大的掩码
        areas = [np.sum(mask) for mask in masks]
        best_idx = np.argmax(areas)
        
        return masks[best_idx]
    
    def _validate_mask_quality(self, mask: np.ndarray) -> bool:
        """验证掩码质量"""
        if mask is None:
            return False
        
        # 检查掩码是否为空
        if np.sum(mask) == 0:
            return False
        
        # 检查掩码面积是否合理
        area = np.sum(mask)
        total_area = mask.shape[0] * mask.shape[1]
        
        # 面积应该在合理范围内（1%到90%）
        if area < total_area * 0.01 or area > total_area * 0.9:
            return False
        
        return True
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """计算掩码质量分数"""
        if mask is None:
            return 0.0
        
        # 简单的质量分数：基于面积和形状
        area = np.sum(mask)
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = area / total_area
        
        # 计算形状的紧凑性
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0
        else:
            compactness = 0
        
        # 综合质量分数
        quality = 0.6 * min(area_ratio * 10, 1.0) + 0.4 * compactness
        
        return quality
    
    def _save_results(self, image_path: str, mask: np.ndarray, image_idx: int, output_dir: str,
                     save_masks: bool, save_visualization: bool, prefix: str = ""):
        """保存分割结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        if save_masks:
            # 保存掩码
            mask_dir = os.path.join(output_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            
            mask_path = os.path.join(mask_dir, f"{base_name}_{prefix}_mask.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        if save_visualization:
            # 保存可视化结果
            vis_dir = os.path.join(output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 读取原图
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 显示原图
            axes[0].imshow(image_rgb)
            axes[0].set_title(f"原图 {image_idx + 1}")
            axes[0].axis('off')
            
            # 显示分割结果
            axes[1].imshow(image_rgb)
            axes[1].imshow(mask, alpha=0.5, cmap='Reds')
            axes[1].set_title(f"分割结果 {image_idx + 1} ({prefix})")
            axes[1].axis('off')
            
            plt.tight_layout()
            vis_path = os.path.join(vis_dir, f"{base_name}_{prefix}_segmentation.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        print(f"\n{'='*60}")
        print(f"分割完成统计")
        print(f"{'='*60}")
        
        total_images = len(self.all_masks)
        processed_images = len(self.processed_indices)
        failed_images = len(self.failed_indices)
        successful_images = sum(1 for mask in self.all_masks if mask is not None)
        
        print(f"总图片数: {total_images}")
        print(f"已处理图片数: {processed_images}")
        print(f"处理失败图片数: {failed_images}")
        print(f"成功分割图片数: {successful_images}")
        print(f"处理成功率: {successful_images/total_images*100:.1f}%")
        
        if failed_images > 0:
            print(f"失败图片索引: {sorted(self.failed_indices)}")
        
        if successful_images > 0:
            # 计算平均质量分数
            quality_scores = [self._calculate_mask_quality(mask) for mask in self.all_masks if mask is not None]
            avg_quality = np.mean(quality_scores)
            print(f"平均质量分数: {avg_quality:.3f}")
        
        print(f"{'='*60}")


def create_iterative_segmenter(model_type: str = "vit_l", 
                             checkpoint_path: Optional[str] = None,
                             device: str = "auto") -> IterativeMaskPropagationSegmenter:
    """
    创建迭代掩码传播分割器的便捷函数
    
    Args:
        model_type: SAM模型类型
        checkpoint_path: 模型检查点路径
        device: 设备类型
        
    Returns:
        IterativeMaskPropagationSegmenter: 分割器实例
    """
    return IterativeMaskPropagationSegmenter(model_type, checkpoint_path, device)


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
