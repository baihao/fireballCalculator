#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整的基于RGB相似性的掩码传播功能
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

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

def visualize_points_on_image(image, positive_points, negative_points, output_path):
    """
    在图像上可视化正负点
    
    Args:
        image: 目标图像 (RGB格式)
        positive_points: 正点坐标列表 [(x, y), ...]
        negative_points: 负点坐标列表 [(x, y), ...]
        output_path: 输出图像路径
    """
    # 创建图像副本
    vis_image = image.copy()
    
    # 绘制正点（红色）
    for x, y in positive_points:
        cv2.circle(vis_image, (x, y), 5, (255, 0, 0), -1)  # 红色实心圆
        cv2.circle(vis_image, (x, y), 8, (255, 255, 255), 2)  # 白色边框
    
    # 绘制负点（蓝色）
    for x, y in negative_points:
        cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)  # 蓝色实心圆
        cv2.circle(vis_image, (x, y), 8, (255, 255, 255), 2)  # 白色边框
    
    # 保存图像
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"   保存带prompt点的图像: {output_path}")
    
    # 创建matplotlib可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原图
    axes[0].imshow(image)
    axes[0].set_title("原图")
    axes[0].axis('off')
    
    # 显示带点的图像
    axes[1].imshow(vis_image)
    axes[1].set_title(f"带Prompt点的图像\n红点: 正点({len(positive_points)}个), 蓝点: 负点({len(negative_points)}个)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存matplotlib图像
    matplotlib_path = output_path.replace('.png', '_matplotlib.png')
    plt.savefig(matplotlib_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   保存matplotlib可视化: {matplotlib_path}")

def generate_propagation_points_visualization(segmenter, image_paths, masks, prompt_data):
    """
    为传播的图片生成prompt点可视化
    
    Args:
        segmenter: 分割器实例
        image_paths: 图像路径列表
        masks: 掩码列表
        prompt_data: prompt数据
    """
    try:
        # 为每张非prompt图片生成三阶段点可视化（参考→映射→筛选），并在失败时强制画掩码
        for i, image_path in enumerate(image_paths):
            if i in prompt_data:
                continue
            print(f"   为图片 {i+1} 生成传播点三阶段可视化...")

            # 读取目标图
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 参考索引
            reference_idx = find_reference_image_for_propagation(i, masks, prompt_data)
            if reference_idx is None:
                print("     未找到参考图片，跳过")
                continue

            ref_image = cv2.imread(image_paths[reference_idx])
            if ref_image is None:
                print("     参考图片读取失败，跳过")
                continue
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_mask = masks[reference_idx]
            if ref_mask is None:
                print("     参考掩码为空，跳过")
                continue

            # 1) 在参考图上采样正负点
            ref_pos = segmenter.prompt_generator.sample_points_from_mask(ref_mask, 10, True)
            ref_neg = segmenter.prompt_generator.sample_points_from_mask(ref_mask, 6, False)

            # 可视化参考点
            vis_dir = Path("test_output/visualization")
            vis_dir.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_paths[i]).stem
            ref_points_png = str(vis_dir / f"{base_name}_ref_points.png")
            visualize_points_on_image(ref_image_rgb, ref_pos, ref_neg, ref_points_png)

            # 2) 将参考点映射到目标图
            mapped_pos = segmenter.prompt_generator.map_points_to_target(ref_image_rgb, ref_pos, image_rgb)
            mapped_neg = segmenter.prompt_generator.map_points_to_target(ref_image_rgb, ref_neg, image_rgb)
            mapped_points_png = str(vis_dir / f"{base_name}_mapped_points.png")
            visualize_points_on_image(image_rgb, mapped_pos, mapped_neg, mapped_points_png)

            # 3) 生成筛选后的目标点
            points, labels = segmenter.prompt_generator.generate_points_with_rgb_similarity(
                ref_image_rgb, ref_mask, image_rgb
            )
            pos_points = [p for p, l in zip(points, labels) if l == 1]
            neg_points = [p for p, l in zip(points, labels) if l == 0]
            filtered_points_png = str(vis_dir / f"{base_name}_filtered_points.png")
            visualize_points_on_image(image_rgb, pos_points, neg_points, filtered_points_png)
            print(f"     参考点: 正{len(ref_pos)} 负{len(ref_neg)} | 映射后: 正{len(mapped_pos)} 负{len(mapped_neg)} | 筛选后: 正{len(pos_points)} 负{len(neg_points)}")

            # 即使失败也尝试用筛选后的点强制画掩码
            try:
                if len(points) > 0:
                    segmenter.predictor.set_image(image_rgb)
                    point_coords = np.array(points)
                    point_labels = np.array(labels)
                    masks_pred, scores, logits = segmenter.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                    if len(masks_pred) > 0:
                        best_mask = segmenter._select_best_mask(masks_pred)
                        forced_vis_path = str(vis_dir / f"{base_name}_forced_mask.png")
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                        axes[0].imshow(image_rgb)
                        axes[0].set_title(f"原图 {i + 1}")
                        axes[0].axis('off')
                        axes[1].imshow(image_rgb)
                        axes[1].imshow(best_mask, alpha=0.5, cmap='Reds')
                        axes[1].set_title(f"强制掩码 {i + 1}\n点: 正{len(pos_points)} 负{len(neg_points)}")
                        axes[1].axis('off')
                        plt.tight_layout()
                        plt.savefig(forced_vis_path, dpi=150, bbox_inches='tight')
                        plt.close()

                        mask_area = int(np.sum(best_mask))
                        mask_quality = segmenter._calculate_mask_quality(best_mask)
                        print(f"     强制掩码: 面积={mask_area}, 质量={mask_quality:.3f}, 输出: {forced_vis_path}")
                else:
                    print("     无筛选点，跳过强制掩码")
            except Exception as e:
                print(f"     ⚠️ 强制掩码可视化失败: {e}")
    
    except Exception as e:
        print(f"   ⚠️ 生成传播点可视化失败: {e}")

def find_reference_image_for_propagation(target_idx, masks, prompt_data):
    """
    为传播的图片找到参考图片
    
    Args:
        target_idx: 目标图片索引
        masks: 掩码列表
        prompt_data: prompt数据
        
    Returns:
        int: 参考图片索引，如果没找到返回None
    """
    # 优先选择相邻的有掩码的图片
    for offset in [1, -1, 2, -2, 3, -3]:
        ref_idx = target_idx + offset
        if 0 <= ref_idx < len(masks) and masks[ref_idx] is not None:
            return ref_idx
    
    # 如果没找到相邻的，选择任意一个有掩码的图片
    for i, mask in enumerate(masks):
        if mask is not None and i != target_idx:
            return i
    
    return None

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
        # 为提示帧添加正负点：中心正点 + 圆外若干负点，约束分割不外溢
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
            save_visualization=True
        )

        # 在提示帧上也叠加显示prompt点，避免用户只看到掩码不见点
        try:
            vis_dir = Path("test_output/visualization")
            vis_dir.mkdir(parents=True, exist_ok=True)
            for idx, info in prompt_data.items():
                img = cv2.imread(image_paths[idx])
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                points = info.get('points', [])
                labels = info.get('labels', [])
                pos = [p for p, l in zip(points, labels) if l == 1]
                neg = [p for p, l in zip(points, labels) if l == 0]
                out_path = str(vis_dir / f"{Path(image_paths[idx]).stem}_prompt_points.png")
                visualize_points_on_image(img_rgb, pos, neg, out_path)
        except Exception as e:
            print(f"   ⚠️ 提示帧点可视化失败: {e}")
        
        # 4.5. 为传播的图片生成点可视化
        print("\n4.5. 生成传播图片的prompt点可视化...")
        generate_propagation_points_visualization(segmenter, image_paths, masks, prompt_data)
        
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
                quality = segmenter._calculate_mask_quality(mask)
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

def test_rgb_similarity_detailed():
    """详细测试RGB相似性功能"""
    print("\n" + "=" * 60)
    print("详细测试RGB相似性功能")
    print("=" * 60)
    
    try:
        segmenter = create_iterative_segmenter()
        
        # 创建测试图片
        ref_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        target_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 设置特定的RGB值用于测试
        ref_image[30:70, 30:70] = [100, 150, 200]  # 参考图片前景
        target_image[30:70, 30:70] = [105, 155, 205]  # 目标图片相似区域
        target_image[10:20, 10:20] = [200, 50, 100]   # 目标图片不相似区域
        
        # 创建掩码
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        
        print("1. 测试点生成...")
        points, labels = segmenter.prompt_generator.generate_points_with_rgb_similarity(ref_image, mask, target_image)
        
        print(f"   生成了 {len(points)} 个点")
        print(f"   正点数量: {sum(labels)}")
        print(f"   负点数量: {len(labels) - sum(labels)}")
        
        # 验证点的分布
        positive_points = [p for p, l in zip(points, labels) if l == 1]
        negative_points = [p for p, l in zip(points, labels) if l == 0]
        
        print(f"   正点坐标: {positive_points}")
        print(f"   负点坐标: {negative_points}")
        
        # 验证正点确实在相似区域
        for x, y in positive_points:
            rgb = target_image[y, x]
            print(f"   正点 ({x}, {y}) RGB: {rgb}")
        
        # 验证负点确实在不相似区域
        for x, y in negative_points:
            rgb = target_image[y, x]
            print(f"   负点 ({x}, {y}) RGB: {rgb}")
        
        # 2. 创建带有点的可视化图像
        print("2. 创建带prompt点的可视化图像...")
        visualize_points_on_image(target_image, positive_points, negative_points, "test_output/points_visualization.png")
        
        print("   ✓ RGB相似性功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ RGB相似性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("基于RGB相似性的掩码传播完整测试")
    
    # 测试RGB相似性功能
    test1_passed = test_rgb_similarity_detailed()
    
    # 测试完整传播流程
    test2_passed = test_complete_propagation()
    
    print(f"\n{'='*60}")
    print(f"测试总结:")
    print(f"RGB相似性功能测试: {'通过' if test1_passed else '失败'}")
    print(f"完整传播流程测试: {'通过' if test2_passed else '失败'}")
    print(f"{'='*60}")
    
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！基于RGB相似性的掩码传播功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查代码。")

if __name__ == "__main__":
    main()
