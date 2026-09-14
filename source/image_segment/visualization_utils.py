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
    from .prompt_generation import (
        DEFAULT_POSITIVE_POINT_COUNT,
        DEFAULT_NEGATIVE_POINT_COUNT,
    )
except ImportError:
    from image_io import imread_unicode
    from prompt_generation import (
        DEFAULT_POSITIVE_POINT_COUNT,
        DEFAULT_NEGATIVE_POINT_COUNT,
    )


class SegmentationVisualizer:
    """分割可视化器"""

    DEBUG_CELL_MAX_WIDTH = 640
    DEBUG_TITLE_HEIGHT = 44
    DEBUG_POINT_RADIUS = 5
    
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
                
                image_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                if i in prompt_data:
                    self._create_prompted_image_visualization(
                        i, image_rgb, masks[i], prompt_data[i], segmenter, image_paths, output_dir
                    )
                else:
                    self._create_propagated_image_visualization(
                        i, image_bgr, masks[i], segmenter, image_paths, output_dir
                    )
                
        except Exception as e:
            print(f"   ⚠️ 生成合并debug可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_prompted_image_visualization(self, idx: int, image_rgb: np.ndarray, 
                                           mask: Optional[np.ndarray], prompt_info: Dict[str, Any],
                                           segmenter, image_paths: List[str], output_dir: str):
        """创建有 prompt 点图片的可视化（1x3，OpenCV 拼图）。"""
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        points = prompt_info['points']
        labels = prompt_info['labels']
        pos_points = [p for p, l in zip(points, labels) if l == 1]
        neg_points = [p for p, l in zip(points, labels) if l == 0]

        p0 = self._draw_points_bgr(image_bgr.copy(), pos_points, neg_points)
        p0 = self._labeled_panel(
            p0, f"Prompt Points | Img {idx + 1} | Pos {len(pos_points)} Neg {len(neg_points)}"
        )

        p1 = image_bgr.copy()
        if mask is not None:
            p1 = self._blend_mask_bgr(p1, mask, (0, 0, 255), 0.45)
            area = int(np.sum(mask))
            p1 = self._labeled_panel(p1, f"Segmentation | Img {idx + 1} | Area {area}")
        else:
            p1 = self._labeled_panel(p1, f"Segmentation Failed | Img {idx + 1}")

        p2 = image_bgr.copy()
        if mask is not None:
            next_pos, next_neg = self._next_iteration_sample_points(segmenter, idx, mask)
            p2 = self._draw_points_bgr(p2, next_pos, next_neg)
            p2 = self._labeled_panel(
                p2,
                f"Next Iter Sampling | Img {idx + 1} | Pos {len(next_pos)} Neg {len(next_neg)}",
            )
        else:
            p2 = self._labeled_panel(p2, f"No mask for sampling | Img {idx + 1}")

        grid = self._assemble_grid([p0, p1, p2], rows=1, cols=3)
        self._save_cv2_debug_image(grid, image_paths[idx], output_dir, "merged_debug")

    def _next_iteration_sample_points(self, segmenter, idx: int, mask: np.ndarray):
        if hasattr(segmenter, 'propagation_details') and idx in segmenter.propagation_details:
            details = segmenter.propagation_details[idx]
            cached_pos = details.get('cached_positive_points')
            next_pts = details.get('next_iteration_points')
            if cached_pos is not None:
                return cached_pos, (next_pts or {}).get('negative') or []
            if next_pts:
                return next_pts['positive'], next_pts['negative']
            return (
                segmenter._get_cached_positive_mask_points(idx, mask),
                segmenter.prompt_generator.sample_points_from_mask(
                    mask, DEFAULT_NEGATIVE_POINT_COUNT, False
                ),
            )
        return (
            segmenter.prompt_generator.sample_points_from_mask(
                mask, DEFAULT_POSITIVE_POINT_COUNT, True
            ),
            segmenter.prompt_generator.sample_points_from_mask(
                mask, DEFAULT_NEGATIVE_POINT_COUNT, False
            ),
        )
    
    def _create_propagated_image_visualization(self, idx: int, image_bgr: np.ndarray,
                                             mask: Optional[np.ndarray], segmenter,
                                             image_paths: List[str], output_dir: str):
        """传播 debug：2x3 OpenCV 拼图（无 Debug Information 文本面板）。"""
        if not hasattr(segmenter, 'propagation_details') or idx not in segmenter.propagation_details:
            return
        debug_data = segmenter.propagation_details[idx]
        if debug_data['status'] == 'unprocessed':
            return

        target_bgr = image_bgr
        ref_idx = debug_data['reference_image_idx']
        ref_bgr = None
        if ref_idx is not None and 0 <= ref_idx < len(image_paths):
            ref_bgr = imread_unicode(image_paths[ref_idx], cv2.IMREAD_COLOR)

        ref_pos = (debug_data.get('reference_points') or {}).get('positive') or []
        ref_neg = (debug_data.get('reference_points') or {}).get('negative') or []
        mapped_pos = (debug_data.get('mapped_points') or {}).get('positive') or []
        mapped_neg = (debug_data.get('mapped_points') or {}).get('negative') or []
        filtered_pos = (debug_data.get('filtered_points') or {}).get('positive') or []
        filtered_neg = (debug_data.get('filtered_points') or {}).get('negative') or []
        stats = debug_data.get('postprocessing_stats') or {}

        # Row 1: reference seg | reference points | mapped points
        if ref_bgr is not None:
            p_ref_seg = ref_bgr.copy()
            if ref_idx is not None and ref_idx < len(segmenter.all_masks):
                ref_mask = segmenter.all_masks[ref_idx]
                if ref_mask is not None:
                    p_ref_seg = self._blend_mask_bgr(p_ref_seg, ref_mask, (0, 0, 255), 0.45)
            p_ref_seg = self._labeled_panel(p_ref_seg, f"Reference Seg | Img {ref_idx + 1}")

            p_ref_pts = self._draw_points_bgr(ref_bgr.copy(), ref_pos, ref_neg)
            p_ref_pts = self._labeled_panel(
                p_ref_pts,
                f"Reference Points | Img {ref_idx + 1} | Pos {len(ref_pos)} Neg {len(ref_neg)}",
            )
        else:
            blank = np.zeros_like(target_bgr)
            p_ref_seg = self._labeled_panel(blank, "No reference image")
            p_ref_pts = self._labeled_panel(blank.copy(), "No reference image")

        p_mapped = self._draw_points_bgr(target_bgr.copy(), mapped_pos, mapped_neg)
        p_mapped = self._labeled_panel(
            p_mapped,
            f"Mapped Points | Img {idx + 1} | Pos {len(mapped_pos)} Neg {len(mapped_neg)}",
        )

        # Row 2: filtered points | SAM mask | EMA processed display mask
        p_filtered = self._draw_points_bgr(target_bgr.copy(), filtered_pos, filtered_neg)
        p_filtered = self._labeled_panel(
            p_filtered,
            f"Filtered Points | Img {idx + 1} | Pos {len(filtered_pos)} Neg {len(filtered_neg)}",
        )

        p_sam = target_bgr.copy()
        original_mask = debug_data.get('original_mask')
        if original_mask is not None:
            p_sam = self._blend_mask_bgr(p_sam, original_mask, (0, 140, 255), 0.45)
            sam_area = int(np.sum(original_mask))
            sam_q = float(stats.get('sam_quality', 0.0))
            p_sam = self._labeled_panel(
                p_sam, f"Original Mask (SAM) | Img {idx + 1} | Area {sam_area} Q {sam_q:.3f}"
            )
        else:
            p_sam = self._labeled_panel(p_sam, f"Original Mask (SAM) | Img {idx + 1} | N/A")

        p_ema = target_bgr.copy()
        if mask is not None:
            p_ema = self._blend_mask_bgr(p_ema, mask, (0, 255, 0), 0.45)
            ema_area = int(np.sum(mask))
            alpha = stats.get('fusion_alpha')
            sigma = stats.get('display_sigma')
            thr = stats.get('display_threshold')
            alpha_s = f"{alpha:.2f}" if alpha is not None else "—"
            meta = f"a={alpha_s}"
            if sigma is not None and thr is not None:
                meta += f" sigma={sigma} thr={thr}"
            p_ema = self._draw_centroid_cross_bgr(p_ema, mask)
            p_ema = self._labeled_panel(
                p_ema,
                f"Processed (Weights EMA) | Img {idx + 1} | Area {ema_area} | {meta}",
            )
        else:
            p_ema = self._labeled_panel(p_ema, f"Processed (Weights EMA) | Img {idx + 1} | Failed")

        grid = self._assemble_grid(
            [p_ref_seg, p_ref_pts, p_mapped, p_filtered, p_sam, p_ema],
            rows=2,
            cols=3,
        )
        self._save_cv2_debug_image(grid, image_paths[idx], output_dir, "merged_debug")

    def _resize_panel(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        max_w = self.DEBUG_CELL_MAX_WIDTH
        if w <= max_w:
            return image_bgr
        scale = max_w / float(w)
        return cv2.resize(
            image_bgr,
            (max_w, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _labeled_panel(self, image_bgr: np.ndarray, title: str) -> np.ndarray:
        img = self._resize_panel(image_bgr)
        bar = np.zeros((self.DEBUG_TITLE_HEIGHT, img.shape[1], 3), dtype=np.uint8)
        bar[:] = (32, 32, 32)
        cv2.putText(
            bar,
            title[:120],
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([bar, img])

    def _blend_mask_bgr(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        color_bgr: tuple,
        alpha: float,
    ) -> np.ndarray:
        out = image_bgr.astype(np.float32)
        m = mask.astype(bool)
        if not np.any(m):
            return image_bgr
        color = np.array(color_bgr, dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_points_bgr(
        self,
        image_bgr: np.ndarray,
        positive_points: List,
        negative_points: List,
    ) -> np.ndarray:
        r = self.DEBUG_POINT_RADIUS
        for x, y in positive_points:
            cv2.circle(image_bgr, (int(x), int(y)), r, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(image_bgr, (int(x), int(y)), r + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        for x, y in negative_points:
            cv2.circle(image_bgr, (int(x), int(y)), r, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(image_bgr, (int(x), int(y)), r + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        return image_bgr

    def _draw_centroid_cross_bgr(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            from .mask_utils import create_mask_analyzer
        except ImportError:
            from mask_utils import create_mask_analyzer
        analyzer = create_mask_analyzer()
        cx, cy = analyzer.calculate_mask_centroid(mask.astype(np.uint8))
        if cx == 0.0 and cy == 0.0:
            return image_bgr
        ix, iy = int(round(cx)), int(round(cy))
        s = 10
        cv2.line(image_bgr, (ix - s, iy), (ix + s, iy), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(image_bgr, (ix, iy - s), (ix, iy + s), (255, 255, 0), 2, cv2.LINE_AA)
        return image_bgr

    def _assemble_grid(self, panels: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        cell_h = max(p.shape[0] for p in panels)
        cell_w = max(p.shape[1] for p in panels)
        normalized = []
        for p in panels:
            canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            canvas[: p.shape[0], : p.shape[1]] = p
            normalized.append(canvas)
        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.hstack(normalized[r * cols : (r + 1) * cols]))
        return np.vstack(row_imgs)

    def _save_cv2_debug_image(self, grid_bgr: np.ndarray, image_path: str, output_dir: str, suffix: str):
        vis_dir = Path(output_dir) / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(image_path).stem
        debug_path = str(vis_dir / f"{base_name}_{suffix}.png")
        cv2.imwrite(debug_path, grid_bgr)
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
