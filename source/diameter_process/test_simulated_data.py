#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟数据测试模块

创建包含烟雾干扰的模拟数据来测试过滤算法。
"""

import numpy as np
import matplotlib.pyplot as plt
from data_filter import filter_smoke_interference_data, apply_data_filter, analyze_data_phases


def generate_simulated_fireball_data() -> tuple:
    """
    生成模拟的火球直径数据，包含烟雾干扰阶段
    
    Returns:
        tuple: (time_data, diameter_data)
    """
    # 时间范围：0-150ms
    t = np.linspace(0, 150, 100)
    
    # 拖曳函数参数
    K = 1200  # 最大直径
    B = 0.6   # 初始拖曳系数
    C = 0.0005  # 拖曳衰减系数
    
    # 生成理想的拖曳曲线
    D_ideal = K * (1 - B * np.exp(-C * t**2))
    
    # 添加噪声
    noise = np.random.normal(0, 10, len(t))  # 10m的标准差噪声
    D_noisy = D_ideal + noise
    
    # 模拟烟雾干扰：在80ms后开始下降
    smoke_start_idx = np.argmin(np.abs(t - 80))
    smoke_interference = np.zeros_like(t)
    
    for i in range(smoke_start_idx, len(t)):
        # 烟雾干扰：逐渐增加的不稳定下降
        time_since_smoke = t[i] - t[smoke_start_idx]
        smoke_factor = 1 + 0.1 * time_since_smoke  # 随时间增加的干扰
        smoke_interference[i] = -50 * smoke_factor * (1 + 0.5 * np.random.randn())
    
    # 应用烟雾干扰
    D_with_smoke = D_noisy + smoke_interference
    
    # 确保数据为正
    D_with_smoke = np.maximum(D_with_smoke, 100)
    
    return t.tolist(), D_with_smoke.tolist()


def test_simulated_data():
    """测试模拟数据"""
    print("=" * 60)
    print("模拟数据过滤测试")
    print("=" * 60)
    
    # 1. 生成模拟数据
    print("\n1. 生成模拟数据...")
    time_data, diameter_data = generate_simulated_fireball_data()
    print(f"✓ 生成 {len(time_data)} 个数据点")
    print(f"✓ 时间范围: {min(time_data):.1f} - {max(time_data):.1f} ms")
    print(f"✓ 直径范围: {min(diameter_data):.1f} - {max(diameter_data):.1f} m")
    
    # 2. 分析数据阶段
    print(f"\n2. 分析数据阶段...")
    phases = analyze_data_phases(time_data, diameter_data)
    if phases:
        print(f"✓ 最大直径: {phases['max_diameter']:.2f}m @ {phases['max_time']:.1f}ms")
        print(f"✓ 膨胀阶段: {phases['growth_phase']['duration']:.1f}ms")
        print(f"✓ 最大直径后: {phases['post_max_phase']['duration']:.1f}ms")
    
    # 3. 测试不同的过滤参数
    test_params = [
        (0.03, 10, "默认参数"),
        (0.05, 10, "更严格阈值"),
        (0.02, 15, "更大窗口"),
        (0.01, 5, "更敏感检测")
    ]
    
    results = []
    for drop_threshold, window_size, description in test_params:
        print(f"\n3. 测试 {description} (阈值={drop_threshold:.1%}, 窗口={window_size})...")
        cutoff_times = filter_smoke_interference_data(time_data, diameter_data, drop_threshold, window_size)
        
        if cutoff_times:
            print(f"✓ 检测到烟雾干扰，截断时间: {cutoff_times[0]:.1f}ms")
            filtered_time, filtered_diameter = apply_data_filter(time_data, diameter_data, drop_threshold, window_size)
            filtered_count = len(filtered_time)
        else:
            print("✓ 未检测到烟雾干扰")
            filtered_count = len(time_data)
        
        results.append({
            'description': description,
            'drop_threshold': drop_threshold,
            'window_size': window_size,
            'cutoff_times': cutoff_times,
            'filtered_count': filtered_count,
            'retention_rate': filtered_count / len(time_data)
        })
    
    # 4. 绘制结果
    print(f"\n4. 绘制结果...")
    plot_simulated_results(time_data, diameter_data, results, phases)
    
    # 5. 输出总结
    print(f"\n" + "=" * 60)
    print("测试结果总结:")
    for result in results:
        print(f"\n{result['description']}:")
        print(f"  参数: 阈值={result['drop_threshold']:.1%}, 窗口={result['window_size']}")
        if result['cutoff_times']:
            print(f"  截断时间: {result['cutoff_times'][0]:.1f}ms")
            print(f"  数据保留率: {result['retention_rate']:.1%}")
        else:
            print(f"  结果: 未检测到烟雾干扰")
            print(f"  数据保留率: {result['retention_rate']:.1%}")


def plot_simulated_results(time_data, diameter_data, results, phases):
    """绘制模拟数据测试结果"""
    try:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模拟数据过滤算法测试', fontsize=16, fontweight='bold')
        
        t = np.array(time_data)
        D = np.array(diameter_data)
        
        # 子图1：原始数据
        ax1 = axes[0, 0]
        ax1.plot(t, D, 'b-', linewidth=2, label='模拟数据')
        ax1.scatter(t, D, color='blue', s=20, alpha=0.6)
        
        # 标记最大直径点
        max_idx = np.argmax(D)
        max_time = t[max_idx]
        max_diameter = D[max_idx]
        ax1.plot(max_time, max_diameter, 'ro', markersize=10, label=f'最大直径 ({max_time:.1f}ms)')
        
        # 标记烟雾干扰开始点
        ax1.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='烟雾干扰开始 (80ms)')
        
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('火球直径 (m)')
        ax1.set_title('原始模拟数据')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2-4：不同参数的过滤结果
        colors = ['green', 'red', 'purple']
        for i, (result, color) in enumerate(zip(results[:3], colors)):
            ax = axes[0, 1] if i == 0 else axes[1, i-1]
            
            ax.plot(t, D, 'b-', linewidth=1, alpha=0.5, label='原始数据')
            
            if result['cutoff_times']:
                cutoff_time = result['cutoff_times'][0]
                filtered_mask = t <= cutoff_time
                ax.plot(t[filtered_mask], D[filtered_mask], color=color, linewidth=3, 
                       label=f'过滤后数据 ({result["retention_rate"]:.1%})')
                ax.axvline(x=cutoff_time, color=color, linestyle='--', linewidth=2, 
                          label=f'截断点 ({cutoff_time:.1f}ms)')
            else:
                ax.plot(t, D, color=color, linewidth=3, label='无过滤 (100%)')
            
            ax.set_xlabel('时间 (ms)')
            ax.set_ylabel('火球直径 (m)')
            ax.set_title(f'{result["description"]}\n(阈值={result["drop_threshold"]:.1%}, 窗口={result["window_size"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = 'simulated_data_filter_test.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 模拟数据测试图已保存: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 绘制失败: {e}")
        return False


if __name__ == "__main__":
    test_simulated_data()
