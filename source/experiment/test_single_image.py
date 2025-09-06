#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张图像火球分割测试
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_fireball_simple(image_path):
    """简化的火球分割函数"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 方法1: HSV颜色分割
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # 定义火球颜色范围
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到火球轮廓")
        return None
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算参数
    area = cv2.contourArea(largest_contour)
    diameter = 2 * np.sqrt(area / np.pi)
    
    # 计算中心
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x, center_y = x + w//2, y + h//2
    
    # 创建结果图像
    result_image = image_rgb.copy()
    
    # 绘制轮廓 (蓝色)
    cv2.drawContours(result_image, [largest_contour], -1, (0, 0, 255), 3)
    
    # 绘制中心点 (绿色)
    cv2.circle(result_image, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # 绘制直径圆 (黄色)
    radius = int(diameter / 2)
    cv2.circle(result_image, (center_x, center_y), radius, (255, 255, 0), 2)
    
    # 添加文字标注
    cv2.putText(result_image, f"Diameter: {diameter:.1f}px", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_image, f"Area: {area:.0f}px^2", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_image, f"Center: ({center_x}, {center_y})", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return {
        'original': image_rgb,
        'mask': mask,
        'result': result_image,
        'diameter': diameter,
        'area': area,
        'center': (center_x, center_y),
        'contour': largest_contour
    }

def visualize_result(result, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(result['original'])
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 分割掩码
    axes[1].imshow(result['mask'], cmap='gray')
    axes[1].set_title('分割掩码')
    axes[1].axis('off')
    
    # 结果图像
    axes[2].imshow(result['result'])
    axes[2].set_title(f'分割结果\n直径: {result["diameter"]:.1f} 像素')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 测试图像路径
    image_path = "../../images/fireball_sequence/1.jpg"
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    print("🔥 火球分割测试")
    print("=" * 30)
    print(f"处理图像: {image_path}")
    
    # 分割火球
    result = segment_fireball_simple(image_path)
    
    if result is None:
        print("分割失败")
        return
    
    # 显示结果
    print(f"直径: {result['diameter']:.1f} 像素")
    print(f"面积: {result['area']:.0f} 像素²")
    print(f"中心: {result['center']}")
    
    # 可视化
    visualize_result(result, "fireball_test_result.png")

if __name__ == "__main__":
    main()
