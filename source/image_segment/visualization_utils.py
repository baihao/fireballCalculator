#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
包含各种分割结果的可视化功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from .image_io import imread_unicode
except ImportError:
    from image_io import imread_unicode


class SegmentationVisualizer:
    """分割可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _draw_centroid_and_radius(self, ax, mask: np.ndarray, color: str = 'yellow', 
                                 show_radius: bool = True, mask_analyzer=None):
        """
        在图像上绘制质心和最大半径
        
        Args:
            ax: matplotlib轴对象
            mask: 掩码数组
            color: 绘制颜色
            show_radius: 是否显示半径圆
            mask_analyzer: 掩码分析器
        """
        if mask is None or np.sum(mask) == 0:
            return
        
        try:
            # 导入掩码分析器（如果没有提供）
            if mask_analyzer is None:
                try:
                    from .mask_utils import create_mask_analyzer
                except ImportError:
                    from mask_utils import create_mask_analyzer
                mask_analyzer = create_mask_analyzer()
            
            # 计算质心和半径
            centroid = mask_analyzer.calculate_mask_centroid(mask)
            max_radius, max_radius_point = mask_analyzer.calculate_max_radius_with_point(mask, centroid)
            
            cx, cy = centroid
            
            # 绘制十字标记
            cross_size = 8
            ax.plot([cx-cross_size, cx+cross_size], [cy, cy], '-', color=color, linewidth=2)
            ax.plot([cx, cx], [cy-cross_size, cy+cross_size], '-', color=color, linewidth=2)
            
            # 绘制最大半径箭头
            if show_radius and max_radius > 0:
                if max_radius_point != (0.0, 0.0):
                    # 使用计算出的最大半径点
                    max_x, max_y = max_radius_point
                    ax.annotate('', xy=(max_x, max_y), xytext=(cx, cy),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                             connectionstyle="arc3", alpha=0.8),
                               label=f'Max Radius: {max_radius:.1f}')
            
            return cx, cy, max_radius
            
        except Exception as e:
            print(f"    ⚠️ 绘制质心和半径失败: {e}")
            return None, None, None
    
    def generate_merged_debug_visualization(self, segmenter, image_paths: List[str], 
                                          masks: List[np.ndarray], prompt_data: Dict[int, Dict[str, Any]], 
                                          output_dir: str = "test_output"):
        """
        生成合并的debug可视化图片
        有prompt点的图片: prompt_points, segmentation_result, next_iteration_sampling
        传播的图片: reference_segmentation, reference_points, mapped_points, filtered_points, segmentation_result, debug_info
        
        Args:
            segmenter: 分割器实例
            image_paths: 图像路径列表
            masks: 掩码列表
            prompt_data: prompt数据
            output_dir: 输出目录
        """
        try:
            # 为每张图片生成合并的debug可视化
            for i, image_path in enumerate(image_paths):
                print(f"   为图片 {i+1} 生成合并debug可视化...")
                
                # 读取目标图片
                image = imread_unicode(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 根据是否为prompt图片创建不同的布局
                if i in prompt_data:
                    self._create_prompted_image_visualization(
                        i, image_rgb, masks[i], prompt_data[i], segmenter, image_paths, output_dir
                    )
                else:
                    self._create_propagated_image_visualization(
                        i, image_rgb, masks[i], segmenter, image_paths, output_dir
                    )
                
        except Exception as e:
            print(f"   ⚠️ 生成合并debug可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_prompted_image_visualization(self, idx: int, image_rgb: np.ndarray, 
                                           mask: Optional[np.ndarray], prompt_info: Dict[str, Any],
                                           segmenter, image_paths: List[str], output_dir: str):
        """创建有prompt点图片的可视化（1x3布局）"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. prompt points (显示prompt点)
        points = prompt_info['points']
        labels = prompt_info['labels']
        pos_points = [p for p, l in zip(points, labels) if l == 1]
        neg_points = [p for p, l in zip(points, labels) if l == 0]
        
        axes[0].imshow(image_rgb)
        self._draw_points(axes[0], pos_points, neg_points)
        axes[0].set_title(f"Prompt Points\nImage {idx+1}\nPos: {len(pos_points)}, Neg: {len(neg_points)}")
        axes[0].axis('off')
        
        # 2. segmentation result (显示分割结果)
        axes[1].imshow(image_rgb)
        if mask is not None:
            axes[1].imshow(mask, alpha=0.5, cmap='Reds')
            mask_area = int(np.sum(mask))
            mask_quality = 0.0  # 不再显示自定义质量分数
            
            # 绘制质心和半径
            cx, cy, radius = self._draw_centroid_and_radius(axes[1], mask, color='yellow')
            if cx is not None:
                axes[1].set_title(f"Segmentation Result\nImage {idx+1}\nArea: {mask_area}, Centroid: ({cx:.1f}, {cy:.1f}), Radius: {radius:.1f}")
            else:
                axes[1].set_title(f"Segmentation Result\nImage {idx+1}\nArea: {mask_area}, Quality: {mask_quality:.3f}")
        else:
            axes[1].text(0.5, 0.5, "Segmentation Failed", ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f"Segmentation Result\nImage {idx+1}\nFailed")
        axes[1].axis('off')
        
        # 3. sampling points for next iteration (为下次迭代选定的采样点)
        axes[2].imshow(image_rgb)
        if mask is not None:
            # 使用保存的下次迭代采样点
            if (hasattr(segmenter, 'propagation_details') and 
                idx in segmenter.propagation_details and 
                'next_iteration_points' in segmenter.propagation_details[idx]):
                
                next_pos_points = segmenter.propagation_details[idx]['next_iteration_points']['positive']
                next_neg_points = segmenter.propagation_details[idx]['next_iteration_points']['negative']
                print(f"     使用保存的下次迭代采样点")
            else:
                # 回退：重新采样
                next_pos_points = segmenter.prompt_generator.sample_points_from_mask(mask, 10, True)
                next_neg_points = segmenter.prompt_generator.sample_points_from_mask(mask, 6, False)
                print(f"     重新采样下次迭代点")
            
            self._draw_points(axes[2], next_pos_points, next_neg_points)
            axes[2].set_title(f"Next Iteration Sampling\nImage {idx+1}\nPos: {len(next_pos_points)}, Neg: {len(next_neg_points)}")
        else:
            axes[2].text(0.5, 0.5, "No mask for sampling", ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title(f"Next Iteration Sampling\nImage {idx+1}\nNo mask")
        axes[2].axis('off')
        
        # 保存图片
        self._save_debug_image(fig, image_paths[idx], output_dir, "merged_debug")
    
    def _create_propagated_image_visualization(self, idx: int, image_rgb: np.ndarray,
                                             mask: Optional[np.ndarray], segmenter,
                                             image_paths: List[str], output_dir: str):
        """创建传播图片的可视化（2x4布局，包含后处理对比）"""
        if not hasattr(segmenter, 'propagation_details') or idx not in segmenter.propagation_details:
            return
        debug_data = segmenter.propagation_details[idx]
        if debug_data['status'] == 'unprocessed':
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # 1. reference_segmentation (参考图片分割结果)
        ref_idx = debug_data['reference_image_idx']
        if ref_idx is not None and 0 <= ref_idx < len(image_paths):
            ref_image = imread_unicode(image_paths[ref_idx], cv2.IMREAD_COLOR)
            if ref_image is not None:
                ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                axes[0, 0].imshow(ref_image_rgb)
                if ref_idx < len(segmenter.all_masks) and segmenter.all_masks[ref_idx] is not None:
                    axes[0, 0].imshow(segmenter.all_masks[ref_idx], alpha=0.5, cmap='Reds')
                axes[0, 0].set_title(f"Reference Segmentation\nImage {ref_idx+1}")
            else:
                axes[0, 0].text(0.5, 0.5, "Failed to load ref image", ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title("Reference Segmentation")
        else:
            axes[0, 0].text(0.5, 0.5, "No reference image", ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("Reference Segmentation")
        axes[0, 0].axis('off')
        
        # 2. reference_points (参考图片的采样点)
        if ref_idx is not None and 0 <= ref_idx < len(image_paths):
            ref_image = imread_unicode(image_paths[ref_idx], cv2.IMREAD_COLOR)
            if ref_image is not None:
                ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                axes[0, 1].imshow(ref_image_rgb)
                
                ref_pos = debug_data['reference_points']['positive']
                ref_neg = debug_data['reference_points']['negative']
                self._draw_points(axes[0, 1], ref_pos, ref_neg)
                axes[0, 1].set_title(f"Reference Points\nImage {ref_idx+1}\nPos: {len(ref_pos)}, Neg: {len(ref_neg)}")
            else:
                axes[0, 1].text(0.5, 0.5, "Failed to load ref image", ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title("Reference Points")
        else:
            axes[0, 1].text(0.5, 0.5, "No reference image", ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Reference Points")
        axes[0, 1].axis('off')
        
        # 3. mapped_points (映射到目标图片的点)
        axes[0, 2].imshow(image_rgb)
        mapped_pos = debug_data['mapped_points']['positive']
        mapped_neg = debug_data['mapped_points']['negative']
        self._draw_points(axes[0, 2], mapped_pos, mapped_neg)
        axes[0, 2].set_title(f"Mapped Points\nImage {idx+1}\nPos: {len(mapped_pos)}, Neg: {len(mapped_neg)}")
        axes[0, 2].axis('off')
        
        # 4. filtered_points (筛选后的点)
        axes[1, 0].imshow(image_rgb)
        filtered_pos = debug_data['filtered_points']['positive']
        filtered_neg = debug_data['filtered_points']['negative']
        self._draw_points(axes[1, 0], filtered_pos, filtered_neg)
        axes[1, 0].set_title(f"Filtered Points\nImage {idx+1}\nPos: {len(filtered_pos)}, Neg: {len(filtered_neg)}")
        axes[1, 0].axis('off')
        
        # 5. original_mask (后处理前的原始掩码)
        axes[1, 1].imshow(image_rgb)
        if 'original_mask' in debug_data and debug_data['original_mask'] is not None:
            original_mask = debug_data['original_mask']
            axes[1, 1].imshow(original_mask, alpha=0.5, cmap='Oranges')
            original_area = int(np.sum(original_mask))
            original_quality = debug_data.get('postprocessing_stats', {}).get('original_quality', 0.0)
            axes[1, 1].set_title(f"Original Mask\nImage {idx+1}\nArea: {original_area}, Quality: {original_quality:.3f}")
        else:
            axes[1, 1].text(0.5, 0.5, "No Original Mask", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f"Original Mask\nImage {idx+1}")
        axes[1, 1].axis('off')
        
        # 6. cleaned_mask (后处理后的清理掩码)
        axes[1, 2].imshow(image_rgb)
        if mask is not None:
            axes[1, 2].imshow(mask, alpha=0.5, cmap='Greens')
            mask_area = int(np.sum(mask))
            mask_quality = debug_data.get('mask_quality', 0.0) or 0.0
            
            # 绘制质心和半径
            cx, cy, radius = self._draw_centroid_and_radius(axes[1, 2], mask, color='cyan')
            
            # 显示保留率信息，根据是否实际进行了后处理调整标题
            retention = debug_data.get('postprocessing_stats', {}).get('area_retention', 1.0)
            if cx is not None:
                if retention < 1.0:
                    axes[1, 2].set_title(f"Cleaned Mask\nImage {idx+1}\nArea: {mask_area}, Centroid: ({cx:.1f}, {cy:.1f}), Radius: {radius:.1f}\nRetention: {retention:.3f}")
                else:
                    axes[1, 2].set_title(f"Final Mask (No Processing)\nImage {idx+1}\nArea: {mask_area}, Centroid: ({cx:.1f}, {cy:.1f}), Radius: {radius:.1f}")
            else:
                if retention < 1.0:
                    axes[1, 2].set_title(f"Cleaned Mask\nImage {idx+1}\nArea: {mask_area}, Quality: {mask_quality:.3f}\nRetention: {retention:.3f}")
                else:
                    axes[1, 2].set_title(f"Final Mask (No Processing)\nImage {idx+1}\nArea: {mask_area}, Quality: {mask_quality:.3f}")
        else:
            axes[1, 2].text(0.5, 0.5, "Processing Failed", ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title(f"Final Mask\nImage {idx+1}\nFailed")
        axes[1, 2].axis('off')
        
        # 7. debug信息文字
        debug_text = self._format_debug_text_with_postprocessing(debug_data, filtered_pos, filtered_neg)
        axes[1, 3].text(0.05, 0.95, debug_text, transform=axes[1, 3].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, 3].set_title("Debug Information")
        axes[1, 3].axis('off')
        
        # 保存图片
        self._save_debug_image(fig, image_paths[idx], output_dir, "merged_debug")
    
    def _draw_points(self, ax, positive_points: List, negative_points: List):
        """在图像上绘制正负点"""
        for x, y in positive_points:
            ax.scatter(x, y, c='red', s=50, marker='o', edgecolors='white', linewidth=2)
        for x, y in negative_points:
            ax.scatter(x, y, c='blue', s=50, marker='o', edgecolors='white', linewidth=2)
    
    def _format_debug_text(self, debug_data: Dict[str, Any], filtered_pos: List, filtered_neg: List) -> str:
        """格式化debug信息文本"""
        return f"""Debug Info:
Status: {debug_data['status']}
Iteration: {debug_data['iteration']}
Ref Image: {debug_data['reference_image_idx'] + 1 if debug_data['reference_image_idx'] is not None else 'None'}

Point Statistics:
- Ref Positive: {len(debug_data['reference_points']['positive']) if debug_data['reference_points'] else 0}
- Ref Negative: {len(debug_data['reference_points']['negative']) if debug_data['reference_points'] else 0}
- Mapped Positive: {len(debug_data['mapped_points']['positive']) if debug_data['mapped_points'] else 0}
- Mapped Negative: {len(debug_data['mapped_points']['negative']) if debug_data['mapped_points'] else 0}
- Filtered Positive: {len(filtered_pos)}
- Filtered Negative: {len(filtered_neg)}"""
    
    def _format_debug_text_with_postprocessing(self, debug_data: Dict[str, Any], filtered_pos: List, filtered_neg: List) -> str:
        """格式化包含后处理信息的debug文本"""
        base_text = f"""Debug Info:
Status: {debug_data['status']}
Iteration: {debug_data['iteration']}
Ref Image: {debug_data['reference_image_idx'] + 1 if debug_data['reference_image_idx'] is not None else 'None'}

Point Statistics:
- Ref Positive: {len(debug_data['reference_points']['positive']) if debug_data['reference_points'] else 0}
- Ref Negative: {len(debug_data['reference_points']['negative']) if debug_data['reference_points'] else 0}
- Mapped Positive: {len(debug_data['mapped_points']['positive']) if debug_data['mapped_points'] else 0}
- Mapped Negative: {len(debug_data['mapped_points']['negative']) if debug_data['mapped_points'] else 0}
- Filtered Positive: {len(filtered_pos)}
- Filtered Negative: {len(filtered_neg)}"""
        
        # 添加后处理统计信息
        if 'postprocessing_stats' in debug_data:
            stats = debug_data['postprocessing_stats']
            area_retention = stats.get('area_retention', 0)
            if area_retention < 1.0:  # 只有在有实际后处理时才显示详细信息
                postprocess_text = f"""

Postprocessing:
- Original Area: {int(stats.get('original_area', 0))}
- Cleaned Area: {int(stats.get('cleaned_area', 0))}
- Retention: {area_retention:.3f}
- SAM Quality: {stats.get('sam_quality', 0):.3f}"""
            else:  # 后处理被跳过或保留率100%
                postprocess_text = f"""

Postprocessing: Skipped
- SAM Quality: {stats.get('sam_quality', 0):.3f}"""
            base_text += postprocess_text
        
        return base_text
    
    def _save_debug_image(self, fig, image_path: str, output_dir: str, suffix: str):
        """保存debug图片"""
        vis_dir = Path(output_dir) / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(image_path).stem
        debug_path = str(vis_dir / f"{base_name}_{suffix}.png")
        
        plt.tight_layout()
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     Saved merged debug image: {debug_path}")
    
    def save_contour_visualization(self, image_paths: List[str], masks: List[np.ndarray], 
                                 output_dir: str = "test_output", geometries: Optional[List[Dict[str, Any]]] = None):
        """
        生成带有蓝色轮廓的原图可视化
        
        Args:
            image_paths: 图像路径列表
            masks: 掩码列表
            output_dir: 输出目录
            geometries: 几何信息列表，如果提供则使用这些信息绘制质心和半径，否则重新计算
        """
        try:
            # 创建轮廓可视化目录
            contour_dir = Path(output_dir) / "contour_visualization"
            contour_dir.mkdir(parents=True, exist_ok=True)
            
            print("生成蓝色轮廓可视化...")
            
            for i, (image_path, mask) in enumerate(zip(image_paths, masks)):
                if mask is None:
                    print(f"   图片 {i+1}: 跳过（无掩码）")
                    continue
                
                # 读取原图
                image = imread_unicode(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"   图片 {i+1}: 跳过（读取失败）")
                    continue
                
                # 创建结果图像的副本
                result_image = image.copy()
                
                # 找到掩码的轮廓
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 绘制蓝色轮廓
                if contours:
                    cv2.drawContours(result_image, contours, -1, (255, 0, 0), 2)  # 蓝色轮廓，线宽2
                    
                    # 计算轮廓信息
                    total_area = sum(cv2.contourArea(contour) for contour in contours)
                    contour_count = len(contours)
                    
                    # 使用几何信息绘制质心和半径（如果没有几何信息则跳过）
                    if geometries is not None and i < len(geometries) and geometries[i] is not None:
                        # 使用提供的几何信息
                        centroid = geometries[i]['centroid']
                        max_radius = geometries[i]['max_radius']
                        max_radius_point = geometries[i].get('max_radius_point', (0.0, 0.0))
                        
                        if centroid != (0.0, 0.0) and max_radius > 0:
                            cx, cy = int(centroid[0]), int(centroid[1])
                            
                            # 绘制十字标记
                            cross_size = 8
                            cv2.line(result_image, (cx-cross_size, cy), (cx+cross_size, cy), (0, 255, 255), 2)
                            cv2.line(result_image, (cx, cy-cross_size), (cx, cy+cross_size), (0, 255, 255), 2)
                            
                            # 绘制最大半径箭头
                            if max_radius_point != (0.0, 0.0):
                                max_x, max_y = int(max_radius_point[0]), int(max_radius_point[1])
                                cv2.arrowedLine(result_image, (cx, cy), (max_x, max_y), (0, 255, 255), 2, tipLength=0.08)
                    else:
                        # 没有几何信息，跳过质心和半径绘制
                        centroid = (0.0, 0.0)
                        max_radius = 0.0
                    
                    print(f"   图片 {i+1}: 绘制了 {contour_count} 个轮廓，总面积 {int(total_area)} 像素")
                    print(f"      质心: ({centroid[0]:.1f}, {centroid[1]:.1f}), 最大半径: {max_radius:.1f}")
                else:
                    print(f"   图片 {i+1}: 未找到有效轮廓")
                
                # 保存结果
                base_name = Path(image_path).stem
                output_path = contour_dir / f"{base_name}_contour.png"
                cv2.imwrite(str(output_path), result_image)
                
            print(f"✓ 轮廓可视化保存到: {contour_dir}")
            
        except Exception as e:
            print(f"❌ 生成轮廓可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_simple_mask_visualization(self, image_paths: List[str], masks: List[np.ndarray],
                                     output_dir: str = "test_output", alpha: float = 0.5):
        """
        生成简单的掩码叠加可视化
        
        Args:
            image_paths: 图像路径列表
            masks: 掩码列表
            output_dir: 输出目录
            alpha: 掩码透明度
        """
        try:
            # 创建简单可视化目录
            simple_dir = Path(output_dir) / "simple_visualization"
            simple_dir.mkdir(parents=True, exist_ok=True)
            
            print("生成简单掩码可视化...")
            
            for i, (image_path, mask) in enumerate(zip(image_paths, masks)):
                if mask is None:
                    print(f"   图片 {i+1}: 跳过（无掩码）")
                    continue
                
                # 读取原图
                image = cv2.imread(image_path)
                if image is None:
                    print(f"   图片 {i+1}: 跳过（读取失败）")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 创建可视化
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # 原图
                axes[0].imshow(image_rgb)
                axes[0].set_title(f"Original Image {i+1}")
                axes[0].axis('off')
                
                # 掩码叠加
                axes[1].imshow(image_rgb)
                axes[1].imshow(mask, alpha=alpha, cmap='Reds')
                mask_area = int(np.sum(mask))
                axes[1].set_title(f"Segmentation Result {i+1}\nArea: {mask_area} pixels")
                axes[1].axis('off')
                
                # 保存图片
                base_name = Path(image_path).stem
                output_path = simple_dir / f"{base_name}_simple.png"
                
                plt.tight_layout()
                plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   图片 {i+1}: 保存到 {output_path}")
                
            print(f"✓ 简单可视化保存到: {simple_dir}")
            
        except Exception as e:
            print(f"❌ 生成简单可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def create_summary_visualization(self, image_paths: List[str], masks: List[np.ndarray],
                                   output_dir: str = "test_output", max_cols: int = 5):
        """
        创建所有结果的汇总可视化
        
        Args:
            image_paths: 图像路径列表
            masks: 掩码列表
            output_dir: 输出目录
            max_cols: 最大列数
        """
        try:
            valid_results = [(path, mask) for path, mask in zip(image_paths, masks) if mask is not None]
            if not valid_results:
                print("没有有效的分割结果，跳过汇总可视化")
                return
            
            num_images = len(valid_results)
            cols = min(max_cols, num_images)
            rows = (num_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            
            # 统一axes处理：确保axes总是一个列表
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = list(axes)
            else:
                axes = axes.flatten()
            
            for i, (image_path, mask) in enumerate(valid_results):
                if i >= len(axes):
                    break
                
                # 读取图像
                image = imread_unicode(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 显示结果
                ax = axes[i]
                ax.imshow(image_rgb)
                ax.imshow(mask, alpha=0.4, cmap='Reds')
                
                base_name = Path(image_path).stem
                mask_area = int(np.sum(mask))
                ax.set_title(f"{base_name}\nArea: {mask_area}")
                ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(len(valid_results), len(axes)):
                axes[i].axis('off')
            
            # 保存汇总图
            summary_path = Path(output_dir) / "segmentation_summary.png"
            plt.tight_layout()
            plt.savefig(str(summary_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 汇总可视化保存到: {summary_path}")
            
        except Exception as e:
            print(f"❌ 生成汇总可视化失败: {e}")
            import traceback
            traceback.print_exc()


def create_visualizer() -> SegmentationVisualizer:
    """创建可视化器的便捷函数"""
    return SegmentationVisualizer()
