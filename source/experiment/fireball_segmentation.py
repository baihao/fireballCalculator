#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球分割实验 - 传统CV算法
使用多种方法分割火球并计算直径
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class FireballSegmenter:
    """火球分割器"""
    
    def __init__(self):
        self.results = []
        
    def segment_fireball(self, image_path, method='color_threshold'):
        """
        分割火球
        
        Args:
            image_path: 图像路径
            method: 分割方法 ('color_threshold', 'brightness', 'edge_detection', 'combined')
        
        Returns:
            dict: 分割结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        if method == 'color_threshold':
            result = self._segment_by_color(image_rgb)
        elif method == 'brightness':
            result = self._segment_by_brightness(image_rgb)
        elif method == 'edge_detection':
            result = self._segment_by_edge(image_rgb)
        elif method == 'combined':
            result = self._segment_combined(image_rgb)
        else:
            raise ValueError(f"未知的分割方法: {method}")
        
        # 计算几何参数
        result.update(self._calculate_geometry(result['mask'], image_rgb))
        result['original_image'] = image_rgb
        result['method'] = method
        result['image_path'] = image_path
        
        return result
    
    def _segment_by_color(self, image):
        """基于颜色阈值分割"""
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 定义火球颜色范围 (橙色到黄色)
        # 调整参数以适应不同光照条件
        lower_orange1 = np.array([0, 50, 50])    # 红色区域
        upper_orange1 = np.array([10, 255, 255])
        lower_orange2 = np.array([10, 50, 50])   # 橙色区域
        upper_orange2 = np.array([25, 255, 255])
        lower_yellow = np.array([25, 50, 50])    # 黄色区域
        upper_yellow = np.array([35, 255, 255])
        
        # 创建多个颜色掩码
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 合并掩码
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        
        # 形态学操作
        mask = self._morphology_operations(mask)
        
        return {'mask': mask, 'method_details': 'HSV颜色阈值分割'}
    
    def _segment_by_brightness(self, image):
        """基于亮度分割"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Otsu阈值
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 结合两种方法
        mask = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
        
        # 形态学操作
        mask = self._morphology_operations(mask)
        
        return {'mask': mask, 'method_details': '自适应+Otsu阈值分割'}
    
    def _segment_by_edge(self, image):
        """基于边缘检测分割"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建掩码
        mask = np.zeros_like(gray)
        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        return {'mask': mask, 'method_details': 'Canny边缘检测分割'}
    
    def _segment_combined(self, image):
        """组合方法分割"""
        # 颜色分割
        color_result = self._segment_by_color(image)
        color_mask = color_result['mask']
        
        # 亮度分割
        brightness_result = self._segment_by_brightness(image)
        brightness_mask = brightness_result['mask']
        
        # 边缘分割
        edge_result = self._segment_by_edge(image)
        edge_mask = edge_result['mask']
        
        # 组合掩码 (投票机制)
        combined_mask = np.zeros_like(color_mask)
        combined_mask[(color_mask > 0) & (brightness_mask > 0)] = 255
        combined_mask[(edge_mask > 0) & (brightness_mask > 0)] = 255
        
        # 形态学操作
        combined_mask = self._morphology_operations(combined_mask)
        
        return {'mask': combined_mask, 'method_details': '组合方法分割'}
    
    def _morphology_operations(self, mask):
        """形态学操作"""
        # 开运算去除小噪声
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 闭运算填充内部空洞
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 膨胀操作连接断裂区域
        kernel_dilate = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        return mask
    
    def _calculate_geometry(self, mask, original_image):
        """计算几何参数"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'diameter_pixels': 0,
                'area_pixels': 0,
                'center': (0, 0),
                'contours': [],
                'largest_contour': None
            }
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算面积
        area_pixels = cv2.contourArea(largest_contour)
        
        # 计算等效直径 (假设为圆形)
        diameter_pixels = 2 * np.sqrt(area_pixels / np.pi)
        
        # 计算中心点
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            # 使用边界框中心
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w//2, y + h//2
        
        return {
            'diameter_pixels': diameter_pixels,
            'area_pixels': area_pixels,
            'center': (center_x, center_y),
            'contours': contours,
            'largest_contour': largest_contour
        }
    
    def visualize_result(self, result, save_path=None):
        """可视化分割结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(result['original_image'])
        axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 分割掩码
        axes[0, 1].imshow(result['mask'], cmap='gray')
        axes[0, 1].set_title('分割掩码', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 叠加结果 - 用蓝色线标注轮廓
        overlay = result['original_image'].copy()
        if result['contours']:
            # 绘制所有轮廓
            cv2.drawContours(overlay, result['contours'], -1, (0, 0, 255), 2)  # 蓝色轮廓
            
            # 绘制最大轮廓
            if result['largest_contour'] is not None:
                cv2.drawContours(overlay, [result['largest_contour']], -1, (255, 0, 0), 3)  # 红色轮廓
            
            # 绘制中心点
            center = result['center']
            cv2.circle(overlay, center, 5, (0, 255, 0), -1)  # 绿色中心点
            
            # 绘制直径线
            radius = result['diameter_pixels'] / 2
            cv2.circle(overlay, center, int(radius), (255, 255, 0), 2)  # 黄色直径圆
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('分割结果 (蓝色轮廓)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 测量信息
        info_text = f"""
分割方法: {result['method_details']}
直径: {result['diameter_pixels']:.1f} 像素
面积: {result['area_pixels']:.0f} 像素²
中心: ({result['center'][0]}, {result['center'][1]})
轮廓数量: {len(result['contours'])}
        """
        
        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def process_sequence(self, image_folder, methods=['color_threshold', 'brightness', 'edge_detection', 'combined']):
        """处理整个图像序列"""
        # 获取所有图像文件
        image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
        image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
        image_files.sort()
        
        if not image_files:
            print(f"在 {image_folder} 中没有找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 为每个方法处理所有图像
        for method in methods:
            print(f"\n=== 使用 {method} 方法处理图像序列 ===")
            method_results = []
            
            for i, image_path in enumerate(image_files):
                print(f"处理图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                try:
                    result = self.segment_fireball(image_path, method)
                    method_results.append(result)
                    
                    # 保存单个结果
                    save_path = f"result_{method}_{i+1}.png"
                    self.visualize_result(result, save_path)
                    
                    print(f"  直径: {result['diameter_pixels']:.1f} 像素")
                    print(f"  面积: {result['area_pixels']:.0f} 像素²")
                    
                except Exception as e:
                    print(f"  处理失败: {e}")
            
            # 保存方法结果
            self.results.append({
                'method': method,
                'results': method_results
            })
    
    def generate_summary(self, save_path="segmentation_summary.png"):
        """生成分割结果总结"""
        if not self.results:
            print("没有结果可总结")
            return
        
        methods = [r['method'] for r in self.results]
        n_methods = len(methods)
        n_images = len(self.results[0]['results'])
        
        fig, axes = plt.subplots(n_methods, n_images, figsize=(4*n_images, 4*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        for method_idx, method_result in enumerate(self.results):
            method = method_result['method']
            results = method_result['results']
            
            for img_idx, result in enumerate(results):
                ax = axes[method_idx, img_idx]
                
                # 显示叠加结果
                overlay = result['original_image'].copy()
                if result['contours']:
                    cv2.drawContours(overlay, result['contours'], -1, (0, 0, 255), 2)
                    if result['largest_contour'] is not None:
                        cv2.drawContours(overlay, [result['largest_contour']], -1, (255, 0, 0), 3)
                    center = result['center']
                    cv2.circle(overlay, center, 5, (0, 255, 0), -1)
                    radius = result['diameter_pixels'] / 2
                    cv2.circle(overlay, center, int(radius), (255, 255, 0), 2)
                
                ax.imshow(overlay)
                ax.set_title(f"{method}\n直径: {result['diameter_pixels']:.1f}px", fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"总结图已保存到: {save_path}")
        plt.show()

def main():
    """主函数"""
    # 创建分割器
    segmenter = FireballSegmenter()
    
    # 图像文件夹路径
    image_folder = "images/fireball_sequence"
    
    # 检查文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"图像文件夹不存在: {image_folder}")
        return
    
    print("🔥 火球分割实验开始")
    print("=" * 50)
    
    # 定义分割方法
    methods = ['color_threshold', 'brightness', 'edge_detection', 'combined']
    
    # 处理图像序列
    segmenter.process_sequence(image_folder, methods)
    
    # 生成总结
    print("\n生成分割结果总结...")
    segmenter.generate_summary()
    
    print("\n🎉 火球分割实验完成！")

if __name__ == "__main__":
    main()
