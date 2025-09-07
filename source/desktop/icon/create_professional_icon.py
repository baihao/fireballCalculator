#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建专业的爆炸火球分析系统图标
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os

def create_professional_icon(size=512):
    """创建专业级图标"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a0a0a')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 背景渐变圆形
    bg_gradient = Circle((0.5, 0.5), 0.48, color='#1a1a2e', alpha=0.9, zorder=1)
    ax.add_patch(bg_gradient)
    
    # 外圈装饰
    outer_ring = Circle((0.5, 0.5), 0.45, fill=False, edgecolor='#16213e', 
                       linewidth=3, alpha=0.8, zorder=2)
    ax.add_patch(outer_ring)
    
    center_x, center_y = 0.5, 0.5
    
    # 创建多层火球效果
    # 外层火焰
    flame_layers = [
        (0.4, '#FF4500', 0.6),  # 深红
        (0.35, '#FF6347', 0.7), # 橙红
        (0.3, '#FF7F50', 0.8),  # 珊瑚
        (0.25, '#FFA500', 0.9), # 橙色
        (0.2, '#FFD700', 1.0),  # 金色
    ]
    
    for radius, color, alpha in flame_layers:
        flame = Circle((center_x, center_y), radius, color=color, alpha=alpha, zorder=3)
        ax.add_patch(flame)
    
    # 核心高光
    core_highlight = Circle((center_x-0.03, center_y+0.03), 0.12, 
                           color='white', alpha=0.9, zorder=4)
    ax.add_patch(core_highlight)
    
    # 添加分析元素 - 数据点
    data_points = [
        (0.25, 0.25), (0.35, 0.2), (0.45, 0.3), (0.55, 0.25),
        (0.65, 0.3), (0.75, 0.25), (0.25, 0.75), (0.35, 0.8),
        (0.45, 0.7), (0.55, 0.75), (0.65, 0.7), (0.75, 0.75)
    ]
    
    for x, y in data_points:
        # 数据点
        point = Circle((x, y), 0.015, color='#00BFFF', alpha=0.8, zorder=5)
        ax.add_patch(point)
        # 数据点外圈
        point_ring = Circle((x, y), 0.02, fill=False, edgecolor='#00BFFF', 
                           alpha=0.4, linewidth=1, zorder=5)
        ax.add_patch(point_ring)
    
    # 连接线表示数据分析
    connections = [
        ((0.25, 0.25), (0.35, 0.2)), ((0.45, 0.3), (0.55, 0.25)),
        ((0.65, 0.3), (0.75, 0.25)), ((0.25, 0.75), (0.35, 0.8)),
        ((0.45, 0.7), (0.55, 0.75)), ((0.65, 0.7), (0.75, 0.75))
    ]
    
    for (x1, y1), (x2, y2) in connections:
        ax.plot([x1, x2], [y1, y2], color='#00BFFF', alpha=0.6, 
                linewidth=1.5, zorder=4)
    
    # 添加温度曲线
    x_temp = np.linspace(0.15, 0.85, 30)
    y_temp = 0.1 + 0.05 * np.sin(6 * np.pi * x_temp) * np.exp(-2 * (x_temp - 0.5)**2)
    ax.plot(x_temp, y_temp, color='#FF6B35', linewidth=2.5, alpha=0.8, zorder=4)
    
    # 添加半径曲线
    x_radius = np.linspace(0.15, 0.85, 25)
    y_radius = 0.9 - 0.05 * np.sin(4 * np.pi * x_radius) * np.exp(-1.5 * (x_radius - 0.5)**2)
    ax.plot(x_radius, y_radius, color='#FFD700', linewidth=2.5, alpha=0.8, zorder=4)
    
    # 添加网格线
    for i in range(4):
        # 水平网格
        ax.axhline(y=0.2 + i*0.2, color='#00BFFF', alpha=0.2, linewidth=0.8)
        # 垂直网格
        ax.axvline(x=0.2 + i*0.2, color='#00BFFF', alpha=0.2, linewidth=0.8)
    
    # 设置背景
    fig.patch.set_facecolor('#0a0a0a')
    
    # 保存
    plt.tight_layout()
    plt.savefig('fireball_professional_icon.png', dpi=size//8, bbox_inches='tight', 
                facecolor='#0a0a0a', edgecolor='none')
    plt.close()
    
    return 'fireball_professional_icon.png'

def create_app_icon(size=512):
    """创建应用图标（推荐使用）"""
    
    # 使用PIL创建更精确的图标
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    
    # 背景圆形（深蓝色渐变效果）
    bg_radius = int(size * 0.48)
    for i in range(bg_radius, 0, -2):
        alpha = int(255 * (1 - i / bg_radius) * 0.8)
        color = (26, 26, 46, alpha)
        draw.ellipse([center - i, center - i, center + i, center + i], 
                    fill=color)
    
    # 外圈装饰
    draw.ellipse([center - bg_radius, center - bg_radius, 
                  center + bg_radius, center + bg_radius], 
                 outline=(22, 33, 62, 200), width=int(size * 0.006))
    
    # 火球主体（多层渐变）
    fireball_radius = int(size * 0.25)
    
    # 外层火焰
    flame_colors = [
        (255, 69, 0, 180),   # 深红
        (255, 99, 71, 200),  # 橙红
        (255, 127, 80, 220), # 珊瑚
        (255, 165, 0, 240),  # 橙色
        (255, 215, 0, 255),  # 金色
    ]
    
    for i, color in enumerate(flame_colors):
        radius = fireball_radius + int(size * 0.03 * (len(flame_colors) - i))
        draw.ellipse([center - radius, center - radius, 
                      center + radius, center + radius], 
                     fill=color)
    
    # 核心高光
    highlight_radius = int(size * 0.12)
    highlight_offset = int(size * 0.03)
    draw.ellipse([center - highlight_radius - highlight_offset, 
                  center - highlight_radius + highlight_offset, 
                  center + highlight_radius - highlight_offset, 
                  center + highlight_radius + highlight_offset], 
                 fill=(255, 255, 255, 220))
    
    # 添加分析元素
    # 数据点
    data_positions = [
        (0.25, 0.25), (0.35, 0.2), (0.45, 0.3), (0.55, 0.25),
        (0.65, 0.3), (0.75, 0.25), (0.25, 0.75), (0.35, 0.8),
        (0.45, 0.7), (0.55, 0.75), (0.65, 0.7), (0.75, 0.75)
    ]
    
    for x, y in data_positions:
        px = int(x * size)
        py = int(y * size)
        point_radius = int(size * 0.015)
        
        # 数据点
        draw.ellipse([px - point_radius, py - point_radius, 
                      px + point_radius, py + point_radius], 
                     fill=(0, 191, 255, 200))
        # 外圈
        draw.ellipse([px - point_radius*1.5, py - point_radius*1.5, 
                      px + point_radius*1.5, py + point_radius*1.5], 
                     outline=(0, 191, 255, 100), width=1)
    
    # 连接线
    connections = [
        ((0.25, 0.25), (0.35, 0.2)), ((0.45, 0.3), (0.55, 0.25)),
        ((0.65, 0.3), (0.75, 0.25)), ((0.25, 0.75), (0.35, 0.8)),
        ((0.45, 0.7), (0.55, 0.75)), ((0.65, 0.7), (0.75, 0.75))
    ]
    
    for (x1, y1), (x2, y2) in connections:
        px1, py1 = int(x1 * size), int(y1 * size)
        px2, py2 = int(x2 * size), int(y2 * size)
        draw.line([px1, py1, px2, py2], fill=(0, 191, 255, 150), width=2)
    
    # 添加温度曲线
    x_points = np.linspace(0.15, 0.85, 20)
    y_points = 0.1 + 0.05 * np.sin(6 * np.pi * x_points) * np.exp(-2 * (x_points - 0.5)**2)
    
    curve_points = []
    for x, y in zip(x_points, y_points):
        curve_points.append((int(x * size), int(y * size)))
    
    if len(curve_points) > 1:
        draw.line(curve_points, fill=(255, 99, 51, 200), width=3)
    
    # 添加半径曲线
    x_points = np.linspace(0.15, 0.85, 15)
    y_points = 0.9 - 0.05 * np.sin(4 * np.pi * x_points) * np.exp(-1.5 * (x_points - 0.5)**2)
    
    curve_points = []
    for x, y in zip(x_points, y_points):
        curve_points.append((int(x * size), int(y * size)))
    
    if len(curve_points) > 1:
        draw.line(curve_points, fill=(255, 215, 0, 200), width=3)
    
    # 保存
    img.save('fireball_app_icon.png', 'PNG')
    return 'fireball_app_icon.png'

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
    print("🎨 创建专业级爆炸火球分析系统图标...")
    
    # 创建专业图标
    print("🔥 生成专业风格图标...")
    professional_icon = create_professional_icon(512)
    
    print("📱 生成应用图标...")
    app_icon = create_app_icon(512)
    
    # 转换为ICO格式
    print("🔄 转换为ICO格式...")
    convert_to_ico(professional_icon, 'fireball_professional_icon.ico')
    convert_to_ico(app_icon, 'fireball_app_icon.ico')
    
    print("\n🎉 专业图标生成完成！")
    print("📁 生成的文件:")
    print(f"   - {professional_icon} (专业风格)")
    print(f"   - {app_icon} (应用图标 - 推荐)")
    print(f"   - fireball_professional_icon.ico (专业风格Windows图标)")
    print(f"   - fireball_app_icon.ico (应用图标Windows版本 - 推荐)")
    
    print("\n💡 推荐使用:")
    print("   - fireball_app_icon.ico 作为主要应用图标")
    print("   - 该图标包含了火球、数据分析元素和现代化设计")
    print("   - 适合在各种操作系统和界面中使用")

if __name__ == "__main__":
    main()
