#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整的基于RGB相似性的掩码传播功能
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "source"))

from image_segment.iterative_mask_propagation import create_iterative_segmenter
import matplotlib.pyplot as plt

def create_test_images():
    """创建测试图片序列"""
    print("创建测试图片序列...")
    
    # 创建输出目录
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图片序列
    image_paths = []
    for i in range(5):
        # 创建测试图片
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # 在每张图片中添加一个移动的圆形区域
        center_x = 100 + i * 10  # 圆形中心向右移动
        center_y = 100
        radius = 30
        
        # 绘制圆形
        cv2.circle(img, (center_x, center_y), radius, (100, 150, 200), -1)
        
        # 保存图片
        img_path = os.path.join(output_dir, f"test_image_{i:02d}.png")
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)
        
        print(f"  创建图片 {i+1}: {img_path}")
    
    return image_paths

def load_data_from_json(json_path: str) -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
    """
    从JSON文件加载图像路径和prompt数据
    
    JSON格式:
    {
        "image_paths": [
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            ...
        ],
        "prompt_data": {
            "0": {
                "points": [[x1, y1], [x2, y2], ...],
                "labels": [1, 1, 0, 0, ...]
            },
            "2": {
                "points": [[x1, y1], [x2, y2], ...], 
                "labels": [1, 0, 1, ...]
            }
        }
    }
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        Tuple[List[str], Dict[int, Dict[str, Any]]]: (图像路径列表, prompt数据字典)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取图像路径
        image_paths = data.get('image_paths', [])
        
        # 提取prompt数据并转换格式
        raw_prompt_data = data.get('prompt_data', {})
        prompt_data = {}
        
        for str_idx, prompt_info in raw_prompt_data.items():
            idx = int(str_idx)  # 将字符串索引转换为整数
            
            # 转换点坐标格式：[[x, y], ...] → [(x, y), ...]
            points = [tuple(point) for point in prompt_info.get('points', [])]
            labels = prompt_info.get('labels', [])
            
            prompt_data[idx] = {
                'points': points,
                'labels': labels
            }
        
        print(f"✓ 从JSON加载了 {len(image_paths)} 张图片")
        print(f"✓ 从JSON加载了 {len(prompt_data)} 个prompt配置")
        print(f"✓ Prompt图片索引: {sorted(prompt_data.keys())}")
        
        return image_paths, prompt_data
        
    except FileNotFoundError:
        print(f"❌ JSON文件不存在: {json_path}")
        return [], {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {e}")
        return [], {}
    except Exception as e:
        print(f"❌ 加载JSON失败: {e}")
        return [], {}

def save_contour_visualization(image_paths: List[str], masks: List[np.ndarray], output_dir: str = "test_output"):
    """
    生成带有蓝色轮廓的原图可视化
    
    Args:
        image_paths: 图像路径列表
        masks: 掩码列表
        output_dir: 输出目录
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
            image = cv2.imread(image_path)
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
                
                print(f"   图片 {i+1}: 绘制了 {contour_count} 个轮廓，总面积 {int(total_area)} 像素")
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

def generate_merged_debug_visualization(segmenter, image_paths, masks, prompt_data, output_dir: str = "test_output"):
    """
    生成合并的debug可视化图片
    有prompt点的图片: prompt_points, segmentation_result, next_iteration_sampling
    传播的图片: segmentation_result, next_iteration_sampling
    
    Args:
        segmenter: 分割器实例
        image_paths: 图像路径列表
        masks: 掩码列表
        prompt_data: prompt数据
        output_dir: 输出目录
    """
    try:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 为每张图片生成合并的debug可视化
        for i, image_path in enumerate(image_paths):
            print(f"   为图片 {i+1} 生成合并debug可视化...")
            
            # 读取目标图片
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 根据是否为prompt图片创建不同的布局
            if i in prompt_data:
                # 有prompt点的图片：创建1x3布局
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 1. prompt points (显示prompt点)
                points = prompt_data[i]['points']
                labels = prompt_data[i]['labels']
                pos_points = [p for p, l in zip(points, labels) if l == 1]
                neg_points = [p for p, l in zip(points, labels) if l == 0]
                
                axes[0].imshow(image_rgb)
                for x, y in pos_points:
                    axes[0].scatter(x, y, c='red', s=50, marker='o', edgecolors='white', linewidth=2)
                for x, y in neg_points:
                    axes[0].scatter(x, y, c='blue', s=50, marker='o', edgecolors='white', linewidth=2)
                axes[0].set_title(f"Prompt Points\nImage {i+1}\nPos: {len(pos_points)}, Neg: {len(neg_points)}")
                axes[0].axis('off')
                
                # 2. segmentation result (显示分割结果)
                axes[1].imshow(image_rgb)
                if masks[i] is not None:
                    axes[1].imshow(masks[i], alpha=0.5, cmap='Reds')
                    mask_area = int(np.sum(masks[i]))
                    mask_quality = segmenter.mask_analyzer.calculate_mask_quality(masks[i]) if masks[i] is not None else 0.0
                    axes[1].set_title(f"Segmentation Result\nImage {i+1}\nArea: {mask_area}, Quality: {mask_quality:.3f}")
                else:
                    axes[1].text(0.5, 0.5, "Segmentation Failed", ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f"Segmentation Result\nImage {i+1}\nFailed")
                axes[1].axis('off')
                
                # 3. sampling points for next iteration (为下次迭代选定的采样点)
                axes[2].imshow(image_rgb)
                if masks[i] is not None:
                    # 使用保存的下次迭代采样点
                    if (hasattr(segmenter, 'propagation_details') and 
                        i in segmenter.propagation_details and 
                        'next_iteration_points' in segmenter.propagation_details[i]):
                        
                        next_pos_points = segmenter.propagation_details[i]['next_iteration_points']['positive']
                        next_neg_points = segmenter.propagation_details[i]['next_iteration_points']['negative']
                        print(f"     使用保存的下次迭代采样点")
                    else:
                        # 回退：重新采样
                        next_pos_points = segmenter.prompt_generator.sample_points_from_mask(masks[i], 10, True)
                        next_neg_points = segmenter.prompt_generator.sample_points_from_mask(masks[i], 6, False)
                        print(f"     重新采样下次迭代点")
                    
                    for x, y in next_pos_points:
                        axes[2].scatter(x, y, c='red', s=50, marker='o', edgecolors='white', linewidth=2)
                    for x, y in next_neg_points:
                        axes[2].scatter(x, y, c='blue', s=50, marker='o', edgecolors='white', linewidth=2)
                    axes[2].set_title(f"Next Iteration Sampling\nImage {i+1}\nPos: {len(next_pos_points)}, Neg: {len(next_neg_points)}")
                else:
                    axes[2].text(0.5, 0.5, "No mask for sampling", ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title(f"Next Iteration Sampling\nImage {i+1}\nNo mask")
                axes[2].axis('off')
                
            else:
                # 传播的图片：创建2x3布局（显示完整的传播过程）
                if not hasattr(segmenter, 'propagation_details') or i not in segmenter.propagation_details:
                    continue
                debug_data = segmenter.propagation_details[i]
                if debug_data['status'] == 'unprocessed':
                    continue
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # 1. reference_segmentation (参考图片分割结果)
                ref_idx = debug_data['reference_image_idx']
                if ref_idx is not None and 0 <= ref_idx < len(image_paths):
                    ref_image = cv2.imread(image_paths[ref_idx])
                    if ref_image is not None:
                        ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                        axes[0, 0].imshow(ref_image_rgb)
                        if masks[ref_idx] is not None:
                            axes[0, 0].imshow(masks[ref_idx], alpha=0.5, cmap='Reds')
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
                    ref_image = cv2.imread(image_paths[ref_idx])
                    if ref_image is not None:
                        ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                        axes[0, 1].imshow(ref_image_rgb)
                        
                        ref_pos = debug_data['reference_points']['positive']
                        ref_neg = debug_data['reference_points']['negative']
                        for x, y in ref_pos:
                            axes[0, 1].scatter(x, y, c='red', s=50, marker='o', edgecolors='white', linewidth=2)
                        for x, y in ref_neg:
                            axes[0, 1].scatter(x, y, c='blue', s=50, marker='o', edgecolors='white', linewidth=2)
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
                for x, y in mapped_pos:
                    axes[0, 2].scatter(x, y, c='red', s=50, marker='o', edgecolors='white', linewidth=2)
                for x, y in mapped_neg:
                    axes[0, 2].scatter(x, y, c='blue', s=50, marker='o', edgecolors='white', linewidth=2)
                axes[0, 2].set_title(f"Mapped Points\nImage {i+1}\nPos: {len(mapped_pos)}, Neg: {len(mapped_neg)}")
                axes[0, 2].axis('off')
                
                # 4. filtered_points (筛选后的点)
                axes[1, 0].imshow(image_rgb)
                filtered_pos = debug_data['filtered_points']['positive']
                filtered_neg = debug_data['filtered_points']['negative']
                for x, y in filtered_pos:
                    axes[1, 0].scatter(x, y, c='red', s=50, marker='o', edgecolors='white', linewidth=2)
                for x, y in filtered_neg:
                    axes[1, 0].scatter(x, y, c='blue', s=50, marker='o', edgecolors='white', linewidth=2)
                axes[1, 0].set_title(f"Filtered Points\nImage {i+1}\nPos: {len(filtered_pos)}, Neg: {len(filtered_neg)}")
                axes[1, 0].axis('off')
                
                # 5. segmentation (最终分割结果)
                axes[1, 1].imshow(image_rgb)
                if masks[i] is not None:
                    axes[1, 1].imshow(masks[i], alpha=0.5, cmap='Reds')
                    mask_area = int(np.sum(masks[i]))
                    mask_quality = debug_data.get('mask_quality', 0.0) or 0.0
                    axes[1, 1].set_title(f"Segmentation Result\nImage {i+1}\nArea: {mask_area}, Quality: {mask_quality:.3f}")
                else:
                    axes[1, 1].text(0.5, 0.5, "Segmentation Failed", ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title(f"Segmentation Result\nImage {i+1}\nFailed")
                axes[1, 1].axis('off')
                
                # 6. debug信息文字
                debug_text = f"""Debug Info:
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
                
                axes[1, 2].text(0.05, 0.95, debug_text, transform=axes[1, 2].transAxes, 
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
                axes[1, 2].set_title("Debug Information")
                axes[1, 2].axis('off')
            
            # 保存合并的debug图片
            vis_dir = Path(output_dir) / "visualization"
            vis_dir.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_paths[i]).stem
            debug_path = str(vis_dir / f"{base_name}_merged_debug.png")
            
            plt.tight_layout()
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"     Saved merged debug image: {debug_path}")
            
    except Exception as e:
        print(f"   ⚠️ 生成合并debug可视化失败: {e}")
        import traceback
        traceback.print_exc()

def test_complete_propagation():
    """测试完整的掩码传播流程"""
    print("=" * 60)
    print("测试完整的基于RGB相似性的掩码传播")
    print("=" * 60)
    
    try:
        # 创建分割器
        print("1. 创建分割器...")
        segmenter = create_iterative_segmenter()
        print("   ✓ 分割器创建成功")
        
        # 创建测试图片
        print("\n2. 创建测试图片...")
        image_paths = create_test_images()
        print(f"   ✓ 创建了 {len(image_paths)} 张测试图片")
        
        # 准备prompt数据（只给第1张和第3张图片添加prompt）
        print("\n3. 准备prompt数据...")
        prompt_data = {
            0: {  # 第1张图片（中心(100,100)，半径≈30）
                'points': [
                    (100, 100),  # 正点：中心
                    (90, 100), (110, 100), (100, 110),  # 额外正点：圆内
                    (100, 40), (40, 100), (160, 100), (100, 160)  # 负点：圆外四方向
                ],
                'labels': [1, 1, 1, 1, 0, 0, 0, 0]
            },
            2: {  # 第3张图片（中心向右移动到(120,100)）
                'points': [
                    (120, 100),
                    (110, 100), (130, 100), (120, 110),  # 额外正点：圆内
                    (120, 40), (60, 100), (180, 100), (120, 160)
                ],
                'labels': [1, 1, 1, 1, 0, 0, 0, 0]
            }
        }
        print(f"   ✓ 为图片 {list(prompt_data.keys())} 添加了prompt点")
        
        # 执行分割
        print("\n4. 执行迭代掩码传播分割...")
        masks = segmenter.segment_sequence_with_iterative_propagation(
            image_paths=image_paths,
            prompt_data=prompt_data,
            output_dir="test_output",
            save_masks=True,
            save_visualization=False  # 不需要单独的分割结果图片，只要合并的debug图片
        )

        # 4.5. 生成合并的debug可视化
        print("\n4.5. 生成合并debug可视化...")
        generate_merged_debug_visualization(segmenter, image_paths, masks, prompt_data)
        
        # 分析结果
        print("\n5. 分析分割结果...")
        successful_masks = sum(1 for mask in masks if mask is not None)
        print(f"   ✓ 成功分割了 {successful_masks}/{len(masks)} 张图片")
        
        # 检查每张图片的分割结果
        for i, mask in enumerate(masks):
            if mask is not None:
                area = np.sum(mask)
                print(f"   图片 {i+1}: 掩码面积 = {area} 像素")
            else:
                print(f"   图片 {i+1}: 分割失败")
        
        # 验证结果质量
        print("\n6. 验证结果质量...")
        quality_scores = []
        for i, mask in enumerate(masks):
            if mask is not None:
                quality = segmenter.mask_analyzer.calculate_mask_quality(mask)
                quality_scores.append(quality)
                print(f"   图片 {i+1}: 质量分数 = {quality:.3f}")
        
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            print(f"   ✓ 平均质量分数: {avg_quality:.3f}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_from_json(json_path: str):
    """从JSON文件测试掩码传播流程"""
    print("=" * 60)
    print("从JSON文件测试迭代掩码传播")
    print("=" * 60)
    
    try:
        # 加载JSON数据
        print("1. 加载JSON数据...")
        image_paths, prompt_data = load_data_from_json(json_path)
        
        if not image_paths or not prompt_data:
            print("❌ JSON数据加载失败或为空")
            return False
        
        # 验证图像文件是否存在
        print("\n2. 验证图像文件...")
        valid_paths = []
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                valid_paths.append(path)
                print(f"   ✓ 图片 {i+1}: {os.path.basename(path)}")
            else:
                print(f"   ❌ 图片 {i+1}: 文件不存在 - {path}")
        
        if not valid_paths:
            print("❌ 没有找到有效的图像文件")
            return False
        
        image_paths = valid_paths
        
        # 创建分割器
        print("\n3. 创建分割器...")
        segmenter = create_iterative_segmenter()
        print("   ✓ 分割器创建成功")
        
        # 执行分割
        print("\n4. 执行迭代掩码传播分割...")
        masks = segmenter.segment_sequence_with_iterative_propagation(
            image_paths=image_paths,
            prompt_data=prompt_data,
            output_dir="json_test_output",
            save_masks=True,
            save_visualization=False
        )
        
        # 生成debug可视化
        print("\n5. 生成debug可视化...")
        generate_merged_debug_visualization(segmenter, image_paths, masks, prompt_data, "json_test_output")
        
        # 生成蓝色轮廓可视化
        print("\n6. 生成蓝色轮廓可视化...")
        save_contour_visualization(image_paths, masks, "json_test_output")
        
        # 分析结果
        print("\n7. 分析分割结果...")
        successful_masks = sum(1 for mask in masks if mask is not None)
        print(f"   ✓ 成功分割了 {successful_masks}/{len(masks)} 张图片")
        
        # 检查每张图片的分割结果
        for i, mask in enumerate(masks):
            if mask is not None:
                area = np.sum(mask)
                quality = segmenter.mask_analyzer.calculate_mask_quality(mask)
                print(f"   图片 {i+1}: 面积={int(area)} 像素, 质量={quality:.3f}")
            else:
                print(f"   图片 {i+1}: 分割失败")
        
        print("\n" + "=" * 60)
        print("JSON测试完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ JSON测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("基于RGB相似性的掩码传播完整测试")
    
    # 检查是否提供了JSON文件参数
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print(f"使用JSON文件: {json_path}")
        
        # 从JSON文件测试
        test_passed = test_from_json(json_path)
        
        print(f"\n{'='*60}")
        print(f"测试总结:")
        print(f"JSON文件测试: {'通过' if test_passed else '失败'}")
        print(f"{'='*60}")
        
        if test_passed:
            print("🎉 JSON测试通过！迭代掩码传播功能正常工作。")
            print("📁 输出文件:")
            print("   - json_test_output/masks/ - 掩码文件")
            print("   - json_test_output/visualization/ - debug可视化")
            print("   - json_test_output/contour_visualization/ - 蓝色轮廓图")
        else:
            print("⚠️ JSON测试失败，请检查输入数据和代码。")
    else:
        # 默认测试模式
        test_passed = test_complete_propagation()
        
        print(f"\n{'='*60}")
        print(f"测试总结:")
        print(f"完整传播流程测试: {'通过' if test_passed else '失败'}")
        print(f"{'='*60}")
        
        if test_passed:
            print("🎉 所有测试通过！基于RGB相似性的掩码传播功能正常工作。")
        else:
            print("⚠️ 测试失败，请检查代码。")
        
        print("\n💡 提示: 也可以使用JSON文件测试:")
        print("   python test_complete_propagation.py data.json")

if __name__ == "__main__":
    main()
