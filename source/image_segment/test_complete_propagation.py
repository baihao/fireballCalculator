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
from typing import List, Dict, Any, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "source"))

from image_segment.iterative_mask_propagation import create_iterative_segmenter
from image_segment.visualization_utils import create_visualizer

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

def load_data_from_json(json_path: str) -> Tuple[List[str], Dict[int, Dict[str, Any]], Optional[Tuple[float, float]]]:
    """
    从JSON文件加载图像路径和prompt数据
    
    支持两种JSON格式:
    1. 简单格式:
    {
        "image_paths": [...],
        "prompt_data": {...}
    }
    
    2. 火球序列格式 (fireball_sequence.json):
    {
        "image_sequence": {
            "image_paths": [...],
            "prompt_data": {...},
            "target_center": [x, y]
        }
    }
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        Tuple[List[str], Dict[int, Dict[str, Any]], Optional[Tuple[float, float]]]: 
        (图像路径列表, prompt数据字典, 目标中心点)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否为火球序列格式
        if 'image_sequence' in data:
            print("✓ 检测到火球序列格式")
            sequence_data = data['image_sequence']
            
            # 从image_sequence中提取数据
            image_paths = sequence_data.get('image_paths', [])
            raw_prompt_data = sequence_data.get('prompt_data', {})
            
            # 提取目标中心点
            target_center = None
            if 'target_center' in sequence_data:
                center_data = sequence_data['target_center']
                if isinstance(center_data, (list, tuple)) and len(center_data) >= 2:
                    target_center = (float(center_data[0]), float(center_data[1]))
                    print(f"✓ 读取到目标中心点: ({target_center[0]:.1f}, {target_center[1]:.1f})")
        else:
            print("✓ 检测到简单格式")
            # 简单格式：直接从根级别提取
            image_paths = data.get('image_paths', [])
            raw_prompt_data = data.get('prompt_data', {})
            target_center = None
        
        # 转换prompt数据格式
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
        
        return image_paths, prompt_data, target_center
        
    except FileNotFoundError:
        print(f"❌ JSON文件不存在: {json_path}")
        return [], {}, None
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {e}")
        return [], {}, None
    except Exception as e:
        print(f"❌ 加载JSON失败: {e}")
        return [], {}, None


def test_complete_propagation():
    """测试完整的掩码传播流程"""
    print("=" * 60)
    print("测试完整的基于RGB相似性的掩码传播")
    print("=" * 60)
    
    try:
        # 创建分割器（默认不启用后处理）
        print("1. 创建分割器...")
        segmenter = create_iterative_segmenter(enable_postprocessing=False)
        print("   ✓ 分割器创建成功（后处理已禁用）")
        
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
        masks, geometries = segmenter.segment_sequence_with_iterative_propagation(
            image_paths=image_paths,
            prompt_data=prompt_data,
            output_dir="test_output",
            save_masks=True,
            save_visualization=False  # 不需要单独的分割结果图片，只要合并的debug图片
        )

        # 4.5. 生成可视化
        print("\n4.5. 生成可视化...")
        visualizer = create_visualizer()
        
        # 生成debug可视化
        visualizer.generate_merged_debug_visualization(segmenter, image_paths, masks, prompt_data)
        
        # 生成轮廓可视化
        visualizer.save_contour_visualization(image_paths, masks, geometries=geometries)
        
        # 生成汇总可视化
        visualizer.create_summary_visualization(image_paths, masks)
        
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
        
        # 验证结果面积
        print("\n6. 验证结果面积...")
        areas = []
        for i, mask in enumerate(masks):
            if mask is not None:
                area = segmenter.mask_analyzer.calculate_mask_area(mask)
                areas.append(area)
                print(f"   图片 {i+1}: 掩码面积 = {area} 像素")
        
        if areas:
            avg_area = np.mean(areas)
            print(f"   ✓ 平均掩码面积: {avg_area:.0f} 像素")
        
        # 显示几何信息
        print("\n7. 几何信息分析...")
        for i, geometry in enumerate(geometries):
            if geometry is not None:
                centroid = geometry['centroid']
                max_radius = geometry['max_radius']
                area = geometry['area']
                print(f"   图片 {i+1}: 质心=({centroid[0]:.1f}, {centroid[1]:.1f}), 最大半径={max_radius:.1f}, 面积={area}")
        
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
        image_paths, prompt_data, target_center = load_data_from_json(json_path)
        
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
        
        # 创建分割器（默认不启用后处理）
        print("\n3. 创建分割器...")
        segmenter = create_iterative_segmenter(enable_postprocessing=False)
        print("   ✓ 分割器创建成功（后处理已禁用）")
        
        # 执行分割
        print("\n4. 执行迭代掩码传播分割...", flush=True)
        masks, geometries = segmenter.segment_sequence_with_iterative_propagation(
            image_paths=image_paths,
            prompt_data=prompt_data,
            output_dir="json_test_output",
            save_masks=True,
            save_visualization=False,
            target_centre=target_center
        )
        
        # 生成可视化
        print("\n5. 生成可视化...")
        visualizer = create_visualizer()
        
        # 生成debug可视化
        visualizer.generate_merged_debug_visualization(segmenter, image_paths, masks, prompt_data, "json_test_output")
        
        # 生成蓝色轮廓可视化
        visualizer.save_contour_visualization(image_paths, masks, "json_test_output", geometries)
        
        # 生成汇总可视化
        visualizer.create_summary_visualization(image_paths, masks, "json_test_output")
        
        # 分析结果
        print("\n7. 分析分割结果...")
        successful_masks = sum(1 for mask in masks if mask is not None)
        print(f"   ✓ 成功分割了 {successful_masks}/{len(masks)} 张图片")
        
        # 检查每张图片的分割结果
        for i, mask in enumerate(masks):
            if mask is not None:
                area = np.sum(mask)
                print(f"   图片 {i+1}: 面积={int(area)} 像素")
            else:
                print(f"   图片 {i+1}: 分割失败")
        
        # 显示几何信息
        print("\n8. 几何信息分析...")
        for i, geometry in enumerate(geometries):
            if geometry is not None:
                centroid = geometry['centroid']
                max_radius = geometry['max_radius']
                area = geometry['area']
                print(f"   图片 {i+1}: 质心=({centroid[0]:.1f}, {centroid[1]:.1f}), 最大半径={max_radius:.1f}, 面积={area}")
        
        # 导出分割结果到JSON文件
        print("\n9. 导出分割结果到JSON文件...")
        export_success = segmenter.output_manager.export_segmentation_results_to_json(
            json_path, image_paths, masks, geometries
        )
        
        if export_success:
            print("   ✅ 分割结果导出成功")
        else:
            print("   ❌ 分割结果导出失败")
        
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
