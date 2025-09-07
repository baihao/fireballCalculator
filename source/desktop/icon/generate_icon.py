#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为爆炸火球分析系统生成应用图标
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os

def create_fireball_icon(size=512):
    """创建火球分析系统图标"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 背景渐变
    gradient = np.linspace(0, 1, 256).reshape(256, -1)
    ax.imshow(gradient, extent=[0, 1, 0, 1], cmap='Blues', alpha=0.3, aspect='auto')
    
    # 创建火球
    center_x, center_y = 0.5, 0.5
    
    # 外层火球（橙色到红色渐变）
    fireball_outer = Circle((center_x, center_y), 0.35, 
                          color='#FF6B35', alpha=0.8, zorder=3)
    ax.add_patch(fireball_outer)
    
    # 中层火球（黄色到橙色）
    fireball_middle = Circle((center_x, center_y), 0.28, 
                           color='#FFD23F', alpha=0.9, zorder=4)
    ax.add_patch(fireball_middle)
    
    # 内层火球（白色到黄色）
    fireball_inner = Circle((center_x, center_y), 0.2, 
                          color='#FFFFFF', alpha=0.95, zorder=5)
    ax.add_patch(fireball_inner)
    
    # 添加火焰效果
    flame_colors = ['#FF4500', '#FF6347', '#FF7F50', '#FFA500']
    for i, color in enumerate(flame_colors):
        flame = Circle((center_x, center_y), 0.4 + i*0.05, 
                      color=color, alpha=0.3 - i*0.05, zorder=2)
        ax.add_patch(flame)
    
    # 添加爆炸冲击波
    for i in range(3):
        wave = Circle((center_x, center_y), 0.45 + i*0.1, 
                     fill=False, edgecolor='#FFD700', 
                     linewidth=2, alpha=0.6 - i*0.15, zorder=1)
        ax.add_patch(wave)
    
    # 添加分析网格线（表示分析功能）
    grid_alpha = 0.2
    for i in range(5):
        # 水平线
        ax.axhline(y=0.2 + i*0.15, color='white', alpha=grid_alpha, linewidth=1)
        # 垂直线
        ax.axvline(x=0.2 + i*0.15, color='white', alpha=grid_alpha, linewidth=1)
    
    # 添加数据点（表示数据分析）
    data_points = [
        (0.3, 0.3), (0.4, 0.25), (0.6, 0.35), (0.7, 0.3),
        (0.25, 0.6), (0.35, 0.65), (0.65, 0.6), (0.75, 0.65)
    ]
    for x, y in data_points:
        point = Circle((x, y), 0.015, color='#00BFFF', alpha=0.8, zorder=6)
        ax.add_patch(point)
    
    # 添加连接线（表示数据关联）
    for i in range(0, len(data_points), 2):
        if i+1 < len(data_points):
            ax.plot([data_points[i][0], data_points[i+1][0]], 
                   [data_points[i][1], data_points[i+1][1]], 
                   color='#00BFFF', alpha=0.6, linewidth=1.5, zorder=5)
    
    # 设置背景
    fig.patch.set_facecolor('black')
    
    # 保存为PNG
    plt.tight_layout()
    plt.savefig('fireball_icon.png', dpi=size//8, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    return 'fireball_icon.png'

def create_modern_icon(size=512):
    """创建现代化图标设计"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a1a')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 背景圆形
    bg_circle = Circle((0.5, 0.5), 0.48, color='#2d2d2d', zorder=1)
    ax.add_patch(bg_circle)
    
    # 主火球（渐变效果）
    center_x, center_y = 0.5, 0.5
    
    # 创建渐变火球
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.25
    
    # 外层火焰
    for i in range(5):
        flame_r = r + 0.05 * i
        flame_alpha = 0.8 - i * 0.15
        flame_color = plt.cm.Reds(0.3 + i * 0.1)
        
        x_flame = center_x + flame_r * np.cos(theta)
        y_flame = center_y + flame_r * np.sin(theta)
        ax.fill(x_flame, y_flame, color=flame_color, alpha=flame_alpha, zorder=2+i)
    
    # 核心火球
    core = Circle((center_x, center_y), r, color='#FFD700', alpha=0.9, zorder=10)
    ax.add_patch(core)
    
    # 内层高光
    highlight = Circle((center_x-0.05, center_y+0.05), r*0.6, 
                      color='white', alpha=0.7, zorder=11)
    ax.add_patch(highlight)
    
    # 添加分析元素
    # 数据波形
    x_wave = np.linspace(0.1, 0.9, 50)
    y_wave = 0.15 + 0.05 * np.sin(8 * np.pi * x_wave) * np.exp(-2 * (x_wave - 0.5)**2)
    ax.plot(x_wave, y_wave, color='#00BFFF', linewidth=3, alpha=0.8, zorder=8)
    
    # 温度曲线
    x_temp = np.linspace(0.1, 0.9, 30)
    y_temp = 0.85 - 0.05 * np.sin(6 * np.pi * x_temp) * np.exp(-3 * (x_temp - 0.5)**2)
    ax.plot(x_temp, y_temp, color='#FF6B35', linewidth=3, alpha=0.8, zorder=8)
    
    # 添加网格点
    grid_points = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    for x, y in grid_points:
        point = Circle((x, y), 0.02, color='#00BFFF', alpha=0.6, zorder=9)
        ax.add_patch(point)
    
    # 设置背景
    fig.patch.set_facecolor('#1a1a1a')
    
    # 保存
    plt.tight_layout()
    plt.savefig('fireball_modern_icon.png', dpi=size//8, bbox_inches='tight', 
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    
    return 'fireball_modern_icon.png'

def create_minimal_icon(size=512):
    """创建简约风格图标"""
    
    # 使用PIL创建图标
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    
    # 背景圆形
    bg_radius = int(size * 0.45)
    draw.ellipse([center - bg_radius, center - bg_radius, 
                  center + bg_radius, center + bg_radius], 
                 fill=(45, 45, 45, 255))
    
    # 火球主体
    fireball_radius = int(size * 0.25)
    draw.ellipse([center - fireball_radius, center - fireball_radius, 
                  center + fireball_radius, center + fireball_radius], 
                 fill=(255, 215, 0, 255))  # 金色
    
    # 内层高光
    highlight_radius = int(size * 0.15)
    highlight_offset = int(size * 0.05)
    draw.ellipse([center - highlight_radius - highlight_offset, 
                  center - highlight_radius + highlight_offset, 
                  center + highlight_radius - highlight_offset, 
                  center + highlight_radius + highlight_offset], 
                 fill=(255, 255, 255, 180))
    
    # 添加火焰效果
    flame_colors = [(255, 69, 0, 200), (255, 99, 71, 150), (255, 165, 0, 100)]
    for i, color in enumerate(flame_colors):
        flame_radius = fireball_radius + int(size * 0.05 * (i + 1))
        draw.ellipse([center - flame_radius, center - flame_radius, 
                      center + flame_radius, center + flame_radius], 
                     fill=color)
    
    # 添加分析网格
    grid_size = int(size * 0.1)
    for i in range(3):
        for j in range(3):
            x = center - grid_size + i * grid_size
            y = center - grid_size + j * grid_size
            if (i + j) % 2 == 0:  # 只画部分网格点
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], 
                           fill=(0, 191, 255, 200))
    
    # 保存
    img.save('fireball_minimal_icon.png', 'PNG')
    return 'fireball_minimal_icon.png'

def convert_to_ico(png_file, ico_file):
    """将PNG转换为ICO格式"""
    try:
        img = Image.open(png_file)
        
        # 创建多个尺寸的ICO
        sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(ico_file, format='ICO', sizes=sizes)
        print(f"✅ 已创建ICO文件: {ico_file}")
        return True
    except Exception as e:
        print(f"❌ 创建ICO文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🎨 开始生成爆炸火球分析系统图标...")
    
    # 创建不同风格的图标
    print("📱 生成经典风格图标...")
    classic_icon = create_fireball_icon(512)
    
    print("🎯 生成现代风格图标...")
    modern_icon = create_modern_icon(512)
    
    print("✨ 生成简约风格图标...")
    minimal_icon = create_minimal_icon(512)
    
    # 转换为ICO格式
    print("🔄 转换为ICO格式...")
    convert_to_ico(classic_icon, 'fireball_icon.ico')
    convert_to_ico(modern_icon, 'fireball_modern_icon.ico')
    convert_to_ico(minimal_icon, 'fireball_minimal_icon.ico')
    
    print("\n🎉 图标生成完成！")
    print("📁 生成的文件:")
    print(f"   - {classic_icon} (经典风格)")
    print(f"   - {modern_icon} (现代风格)")
    print(f"   - {minimal_icon} (简约风格)")
    print(f"   - fireball_icon.ico (Windows图标)")
    print(f"   - fireball_modern_icon.ico (现代风格Windows图标)")
    print(f"   - fireball_minimal_icon.ico (简约风格Windows图标)")
    
    print("\n💡 使用建议:")
    print("   - 经典风格：适合传统应用界面")
    print("   - 现代风格：适合现代化UI设计")
    print("   - 简约风格：适合简洁的应用界面")

if __name__ == "__main__":
    main()
