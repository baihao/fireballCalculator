#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径数据过滤测试模块

测试数据过滤算法，并生成可视化结果。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
source_path = os.path.join(project_root, 'source')
sys.path.insert(0, source_path)

from desktop.sequence_manager import SequenceManager
from desktop.segment_utils import build_time_diameter_series
from data_filter import filter_smoke_interference_data, apply_data_filter, analyze_data_phases


def load_and_extract_data(file_path: str) -> Tuple[List[float], List[float], str]:
    """
    加载分割序列文件并提取直径数据
    
    Args:
        file_path: 分割序列JSON文件路径
        
    Returns:
        Tuple[List[float], List[float], str]: (时间数据, 直径数据, 错误信息)
    """
    try:
        # 加载序列文件
        manager = SequenceManager()
        success, sequence_data, message = manager.load_sequence_file(file_path)
        
        if not success:
            return [], [], f"加载序列文件失败: {message}"
        
        # 检查是否包含分割结果
        if not manager.has_segmentation_results(sequence_data):
            return [], [], "序列文件中没有分割结果数据"
        
        # 获取分割结果
        segmentation_results = manager.get_segmentation_results_from_sequence(sequence_data)
        if not segmentation_results:
            return [], [], "无法获取分割结果"
        
        # 获取爆炸时长参数
        parameters = manager.get_parameters_from_sequence(sequence_data)
        explosion_duration_str = parameters.get('explosion_duration', '100')
        
        try:
            explosion_duration_ms = float(explosion_duration_str)
        except ValueError:
            explosion_duration_ms = 140
            print(f"⚠️ 无法解析爆炸时长 '{explosion_duration_str}'，使用默认值 {explosion_duration_ms} 毫秒")
        
        # 构建时间-直径序列
        time_diameter_series = build_time_diameter_series(segmentation_results, explosion_duration_ms)
        
        if not time_diameter_series:
            return [], [], "没有有效的直径数据点"
        
        # 分离时间和直径数据
        time_data = [point[0] for point in time_diameter_series]
        diameter_data = [point[1] for point in time_diameter_series]
        
        return time_data, diameter_data, "数据提取成功"
        
    except Exception as e:
        return [], [], f"数据提取失败: {str(e)}"


def plot_data_filtering_results(time_data: List[float], diameter_data: List[float],
                               cutoff_times: List[float], phases: dict,
                               save_path: str, drop_threshold: float = 0.02) -> bool:
    """
    绘制数据过滤结果图
    
    Args:
        time_data: 原始时间数据
        diameter_data: 原始直径数据
        cutoff_times: 截断时间点
        phases: 阶段分析结果
        save_path: 保存路径
        drop_threshold: 下降阈值
        
    Returns:
        bool: 是否成功绘制
    """
    try:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        t = np.array(time_data)
        D = np.array(diameter_data)
        
        # 主图：直径随时间变化
        ax1.plot(t, D, 'b-', linewidth=2, label='原始数据', alpha=0.7)
        ax1.scatter(t, D, color='blue', s=30, alpha=0.6, zorder=3)
        
        # 标记最大直径点
        max_idx = np.argmax(D)
        max_time = t[max_idx]
        max_diameter = D[max_idx]
        ax1.plot(max_time, max_diameter, 'ro', markersize=12, label=f'最大直径 ({max_time:.1f}ms, {max_diameter:.1f}m)', zorder=4)
        
        # 标记截断点
        if cutoff_times:
            cutoff_time = cutoff_times[0]
            cutoff_idx = np.argmin(np.abs(t - cutoff_time))
            cutoff_diameter = D[cutoff_idx]
            ax1.axvline(x=cutoff_time, color='red', linestyle='--', linewidth=2, 
                       label=f'截断点 ({cutoff_time:.1f}ms)', zorder=2)
            ax1.plot(cutoff_time, cutoff_diameter, 'rs', markersize=10, zorder=4)
            
            # 标记过滤后的数据
            filtered_mask = t <= cutoff_time
            ax1.plot(t[filtered_mask], D[filtered_mask], 'g-', linewidth=3, 
                    label='过滤后数据', alpha=0.8, zorder=1)
        
        # 添加阶段标注
        if phases:
            growth_phase = phases.get('growth_phase', {})
            post_max_phase = phases.get('post_max_phase', {})
            
            # 膨胀阶段
            ax1.axvspan(growth_phase.get('start_time', 0), growth_phase.get('end_time', 0), 
                       alpha=0.2, color='green', label='膨胀阶段')
            
            # 最大直径后阶段
            if cutoff_times:
                ax1.axvspan(post_max_phase.get('start_time', 0), cutoff_times[0], 
                           alpha=0.2, color='yellow', label='稳定阶段')
                ax1.axvspan(cutoff_times[0], post_max_phase.get('end_time', 0), 
                           alpha=0.2, color='red', label='烟雾干扰阶段')
            else:
                ax1.axvspan(post_max_phase.get('start_time', 0), post_max_phase.get('end_time', 0), 
                           alpha=0.2, color='yellow', label='最大直径后阶段')
        
        ax1.set_xlabel('时间 (ms)', fontsize=12)
        ax1.set_ylabel('火球直径 (m)', fontsize=12)
        ax1.set_title('火球直径数据过滤结果', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'''数据统计:
• 总数据点: {len(time_data)}
• 时间范围: {t[0]:.1f} - {t[-1]:.1f} ms
• 直径范围: {D.min():.1f} - {D.max():.1f} m
• 最大直径: {max_diameter:.1f} m @ {max_time:.1f} ms'''
        
        if cutoff_times:
            filtered_count = np.sum(t <= cutoff_times[0])
            stats_text += f'''
• 截断时间: {cutoff_times[0]:.1f} ms
• 保留数据点: {filtered_count}'''
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 子图：滑动平均分析
        if cutoff_times:
            # 计算滑动平均
            window_size = 10
            max_idx = np.argmax(D)
            t_after = t[max_idx:]
            D_after = D[max_idx:]
            
            # 计算滑动平均
            smoothed_D = []
            for i in range(len(D_after)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(D_after), i + window_size // 2 + 1)
                smoothed_D.append(np.mean(D_after[start_idx:end_idx]))
            
            smoothed_D = np.array(smoothed_D)
            
            # 计算相对下降
            max_diameter = D[max_idx]
            relative_drops = (max_diameter - smoothed_D) / max_diameter
            
            ax2.plot(t_after, relative_drops * 100, 'g-', linewidth=2, label='相对下降 (%)')
            ax2.axhline(y=drop_threshold * 100, color='red', linestyle='--', linewidth=2, label=f'{drop_threshold*100:.0f}% 阈值')
            ax2.axvline(x=cutoff_times[0], color='red', linestyle='--', linewidth=2, 
                       label=f'截断点 ({cutoff_times[0]:.1f}ms)')
            
            ax2.set_xlabel('时间 (ms)', fontsize=12)
            ax2.set_ylabel('相对下降 (%)', fontsize=12)
            ax2.set_title('滑动平均相对下降分析', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(relative_drops) * 100 * 1.1)
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 数据过滤结果图已保存: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 绘制失败: {e}")
        return False


def test_data_filter(file_path: str, output_dir: Optional[str] = None,
                    drop_threshold: float = 0.02, window_size: int = 10) -> Dict[str, Any]:
    """
    测试数据过滤算法
    
    Args:
        file_path: 分割序列JSON文件路径
        output_dir: 输出目录（可选）
        drop_threshold: 下降阈值
        window_size: 滑动窗口大小
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    try:
        print("=" * 60)
        print("火球直径数据过滤测试")
        print("=" * 60)
        
        # 1. 加载数据
        print(f"\n1. 加载数据: {file_path}")
        time_data, diameter_data, message = load_and_extract_data(file_path)
        if not time_data or not diameter_data:
            return {
                'success': False,
                'error': f"数据加载失败: {message}",
                'file_path': file_path
            }
        
        print(f"✓ {message}")
        print(f"✓ 数据点: {len(time_data)}, 时间范围: {min(time_data):.1f}-{max(time_data):.1f}ms")
        
        # 2. 分析数据阶段
        print(f"\n2. 分析数据阶段...")
        phases = analyze_data_phases(time_data, diameter_data)
        if phases:
            print(f"✓ 最大直径: {phases['max_diameter']:.2f}m @ {phases['max_time']:.1f}ms")
            print(f"✓ 膨胀阶段: {phases['growth_phase']['duration']:.1f}ms")
            print(f"✓ 最大直径后: {phases['post_max_phase']['duration']:.1f}ms")
        
        # 3. 执行数据过滤
        print(f"\n3. 执行数据过滤...")
        print(f"   参数: 下降阈值={drop_threshold:.1%}, 窗口大小={window_size}")
        cutoff_times = filter_smoke_interference_data(time_data, diameter_data, drop_threshold, window_size)
        
        if cutoff_times:
            print(f"✓ 检测到烟雾干扰，截断时间: {cutoff_times[0]:.1f}ms")
            
            # 应用过滤
            filtered_time, filtered_diameter = apply_data_filter(time_data, diameter_data, drop_threshold, window_size)
            print(f"✓ 过滤后数据点: {len(filtered_time)}")
        else:
            print("✓ 未检测到烟雾干扰，保留所有数据")
            filtered_time, filtered_diameter = time_data, diameter_data
        
        # 4. 绘制结果
        print(f"\n4. 绘制结果...")
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = os.path.join(output_dir, f"{base_name}_data_filter_test.png")
        
        plot_success = plot_data_filtering_results(time_data, diameter_data, cutoff_times, phases, plot_path, drop_threshold)
        
        if plot_success:
            print(f"✓ 结果图已保存: {plot_path}")
        else:
            print("⚠️ 保存结果图失败")
        
        # 5. 返回结果
        result = {
            'success': True,
            'file_path': file_path,
            'plot_path': plot_path if plot_success else None,
            'original_data': {
                'time_data': time_data,
                'diameter_data': diameter_data,
                'data_points': len(time_data)
            },
            'filtered_data': {
                'time_data': filtered_time,
                'diameter_data': filtered_diameter,
                'data_points': len(filtered_time)
            },
            'filtering_info': {
                'cutoff_times': cutoff_times,
                'drop_threshold': drop_threshold,
                'window_size': window_size,
                'filtered': len(cutoff_times) > 0
            },
            'phases': phases
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"测试过程异常: {str(e)}",
            'file_path': file_path
        }


def main():
    """主函数：命令行接口"""
    if len(sys.argv) < 2:
        print("用法: python test_data_filter.py <segmented_sequence.json> [output_dir] [--threshold=0.03] [--window=10]")
        print("")
        print("参数:")
        print("  segmented_sequence.json: 分割序列JSON文件路径")
        print("  output_dir: 输出目录（可选，默认为输入文件所在目录）")
        print("  --threshold=0.03: 下降阈值（可选，默认3%）")
        print("  --window=10: 滑动窗口大小（可选，默认10）")
        print("")
        print("示例:")
        print("  python test_data_filter.py test_data/fireball_sequence_segmented.json")
        print("  python test_data_filter.py test_data/fireball_sequence_segmented.json ./output --threshold=0.05 --window=15")
        return
    
    # 解析命令行参数
    file_path = sys.argv[1]
    output_dir = None
    drop_threshold = 0.02
    window_size = 10
    
    for arg in sys.argv[2:]:
        if arg.startswith('--threshold='):
            drop_threshold = float(arg.split('=')[1])
        elif arg.startswith('--window='):
            window_size = int(arg.split('=')[1])
        elif not arg.startswith('--'):
            output_dir = arg
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 执行测试
    result = test_data_filter(file_path, output_dir, drop_threshold, window_size)
    
    # 输出结果
    print("\n" + "=" * 60)
    if result['success']:
        print("✅ 数据过滤测试完成！")
        print(f"📁 输入文件: {result['file_path']}")
        if result.get('plot_path'):
            print(f"📊 结果图表: {result['plot_path']}")
        
        original = result['original_data']
        filtered = result['filtered_data']
        filtering = result['filtering_info']
        
        print(f"\n📈 数据统计:")
        print(f"   原始数据点: {original['data_points']}")
        print(f"   过滤后数据点: {filtered['data_points']}")
        print(f"   数据保留率: {filtered['data_points']/original['data_points']:.1%}")
        
        if filtering['filtered']:
            print(f"\n🔍 过滤信息:")
            print(f"   截断时间: {filtering['cutoff_times'][0]:.1f} ms")
            print(f"   下降阈值: {filtering['drop_threshold']:.1%}")
            print(f"   窗口大小: {filtering['window_size']}")
        else:
            print(f"\n🔍 过滤信息: 未检测到烟雾干扰")
        
        phases = result.get('phases', {})
        if phases:
            print(f"\n📊 阶段分析:")
            print(f"   最大直径: {phases['max_diameter']:.2f} m @ {phases['max_time']:.1f} ms")
            print(f"   膨胀阶段: {phases['growth_phase']['duration']:.1f} ms")
            print(f"   最大直径后: {phases['post_max_phase']['duration']:.1f} ms")
        
    else:
        print("❌ 数据过滤测试失败！")
        print(f"📁 输入文件: {result['file_path']}")
        print(f"💥 错误信息: {result['error']}")


if __name__ == "__main__":
    main()
