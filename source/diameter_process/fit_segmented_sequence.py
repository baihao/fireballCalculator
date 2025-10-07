#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于分割序列JSON的拖曳曲线拟合模块

使用 sequence_manager.py 读取分割序列JSON文件，
提取各时间点的最大半径值，然后使用 diameter_drag_fitting.py 
拟合拖曳曲线参数并绘制结果。
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
source_path = os.path.join(project_root, 'source')
sys.path.insert(0, source_path)

from desktop.sequence_manager import SequenceManager
from desktop.segment_utils import build_time_diameter_series
from diameter_drag_fitting import DiameterDragFitter
from drag_fit_plotter import DragFitPlotter


def load_segmented_sequence(file_path: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    加载分割序列文件
    
    Args:
        file_path: 分割序列JSON文件路径
        
    Returns:
        Tuple[bool, Dict[str, Any], str]: (是否成功, 序列数据, 错误信息)
    """
    try:
        manager = SequenceManager()
        success, sequence_data, message = manager.load_sequence_file(file_path)
        
        if not success:
            return False, {}, f"加载序列文件失败: {message}"
        
        # 检查是否包含分割结果
        if not manager.has_segmentation_results(sequence_data):
            return False, {}, "序列文件中没有分割结果数据"
        
        return True, sequence_data, "加载成功"
        
    except Exception as e:
        return False, {}, f"加载序列文件异常: {str(e)}"


def extract_diameter_data(sequence_data: Dict[str, Any]) -> Tuple[List[float], List[float], str]:
    """
    从序列数据中提取直径数据
    
    Args:
        sequence_data: 序列数据字典
        
    Returns:
        Tuple[List[float], List[float], str]: (时间数据, 直径数据, 错误信息)
    """
    try:
        manager = SequenceManager()
        
        # 获取分割结果
        segmentation_results = manager.get_segmentation_results_from_sequence(sequence_data)
        if not segmentation_results:
            return [], [], "无法获取分割结果"
        
        # 获取爆炸时长参数
        parameters = manager.get_parameters_from_sequence(sequence_data)
        explosion_duration_str = parameters.get('explosion_duration', '100')
        
        try:
            # 爆炸时长参数已经是毫秒单位，不需要转换
            explosion_duration_ms = float(explosion_duration_str)
        except ValueError:
            explosion_duration_ms = 140  # 默认140毫秒
            print(f"⚠️ 无法解析爆炸时长 '{explosion_duration_str}'，使用默认值 {explosion_duration_ms} 毫秒")
        
        # 构建时间-直径序列
        time_diameter_series = build_time_diameter_series(segmentation_results, explosion_duration_ms)
        
        if not time_diameter_series:
            return [], [], "没有有效的直径数据点"
        
        # 分离时间和直径数据
        time_data = [point[0] for point in time_diameter_series]  # 保持毫秒单位
        diameter_data = [point[1] for point in time_diameter_series]  # 保持米为单位
        
        print(f"✓ 提取到 {len(time_data)} 个有效数据点")
        print(f"✓ 时间范围: {min(time_data):.1f} - {max(time_data):.1f} 毫秒")
        print(f"✓ 直径范围: {min(diameter_data):.3f} - {max(diameter_data):.3f} 米")
        
        return time_data, diameter_data, "提取成功"
        
    except Exception as e:
        return [], [], f"提取直径数据失败: {str(e)}"


def fit_drag_curve_from_sequence(file_path: str, output_dir: Optional[str] = None, 
                                use_robust_fitting: bool = True) -> Dict[str, Any]:
    """
    从分割序列文件拟合拖曳曲线
    
    Args:
        file_path: 分割序列JSON文件路径
        output_dir: 输出目录（可选）
        use_robust_fitting: 是否使用鲁棒拟合
        
    Returns:
        Dict[str, Any]: 拟合结果
    """
    try:
        print("=" * 60)
        print("火球直径拖曳曲线拟合")
        print("=" * 60)
        
        # 1. 加载序列文件
        print(f"\n1. 加载序列文件: {file_path}")
        success, sequence_data, message = load_segmented_sequence(file_path)
        if not success:
            return {
                'success': False,
                'error': f"加载序列文件失败: {message}",
                'file_path': file_path
            }
        
        print(f"✓ {message}")
        
        # 2. 提取直径数据
        print(f"\n2. 提取直径数据...")
        time_data, diameter_data, extract_message = extract_diameter_data(sequence_data)
        if not time_data or not diameter_data:
            return {
                'success': False,
                'error': f"提取直径数据失败: {extract_message}",
                'file_path': file_path
            }
        
        print(f"✓ {extract_message}")
        
        # 3. 执行拟合（启用数据过滤）
        print(f"\n3. 执行拖曳曲线拟合...")
        fitter = DiameterDragFitter()
        fit_result = fitter.fit_drag_curve(time_data, diameter_data, use_robust_fitting, time_unit='ms',
                                         enable_data_filtering=True, drop_threshold=0.02, window_size=10)
        
        if not fit_result.get('success', False):
            return {
                'success': False,
                'error': f"拟合失败: {fit_result.get('error', '未知错误')}",
                'file_path': file_path,
                'time_data': time_data,
                'diameter_data': diameter_data
            }
        
        print(f"✓ 拟合成功")
        print(f"  参数: K={fit_result['K']:.4f}, B={fit_result['B']:.4f}, C={fit_result['C']:.4f}")
        print(f"  质量: R²={fit_result.get('r_squared', 0):.4f}, RMSE={fit_result.get('rmse', 0):.4f}")
        
        # 4. 绘制结果
        print(f"\n4. 绘制拟合结果...")
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = os.path.join(output_dir, f"{base_name}_drag_fit.png")
        
        plotter = DragFitPlotter()
        plot_success = plotter.plot_fit_results(time_data, diameter_data, fit_result, plot_path, time_unit='ms')
        
        if plot_success:
            print(f"✓ 拟合结果图已保存: {plot_path}")
        else:
            print(f"⚠️ 保存拟合结果图失败")
        
        # 5. 返回完整结果
        result = {
            'success': True,
            'file_path': file_path,
            'plot_path': plot_path if plot_success else None,
            'fit_result': fit_result,
            'data_summary': {
                'data_points': len(time_data),
                'time_range': [min(time_data), max(time_data)],
                'diameter_range': [min(diameter_data), max(diameter_data)],
                'explosion_duration': max(time_data) - min(time_data)
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"拟合过程异常: {str(e)}",
            'file_path': file_path
        }


def main():
    """主函数：命令行接口"""
    if len(sys.argv) < 2:
        print("用法: python fit_segmented_sequence.py <segmented_sequence.json> [output_dir] [--robust]")
        print("")
        print("参数:")
        print("  segmented_sequence.json: 分割序列JSON文件路径")
        print("  output_dir: 输出目录（可选，默认为输入文件所在目录）")
        print("  --robust: 使用鲁棒拟合（可选，默认为True）")
        print("")
        print("示例:")
        print("  python fit_segmented_sequence.py test_data/fireball_sequence_segmented.json")
        print("  python fit_segmented_sequence.py test_data/fireball_sequence_segmented.json ./output --robust")
        return
    
    # 解析命令行参数
    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    use_robust = '--robust' in sys.argv or '--no-robust' not in sys.argv  # 默认使用鲁棒拟合
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 执行拟合
    result = fit_drag_curve_from_sequence(file_path, output_dir, use_robust)
    
    # 输出结果
    print("\n" + "=" * 60)
    if result['success']:
        print("✅ 拟合完成！")
        print(f"📁 输入文件: {result['file_path']}")
        if result.get('plot_path'):
            print(f"📊 结果图表: {result['plot_path']}")
        
        fit_result = result['fit_result']
        print(f"\n📈 拟合参数:")
        print(f"   K (最大直径): {fit_result['K']:.4f} 米")
        print(f"   B (拖曳系数): {fit_result['B']:.4f}")
        print(f"   C (衰减系数): {fit_result['C']:.4e} 毫秒⁻²")
        
        print(f"\n📊 拟合质量:")
        print(f"   R² 决定系数: {fit_result.get('r_squared', 0):.4f}")
        print(f"   均方根误差: {fit_result.get('rmse', 0):.4f} 米")
        print(f"   平均绝对误差: {fit_result.get('mae', 0):.4f} 米")
        
        data_summary = result['data_summary']
        print(f"\n📋 数据摘要:")
        print(f"   数据点数: {data_summary['data_points']}")
        print(f"   时间范围: {data_summary['time_range'][0]:.1f} - {data_summary['time_range'][1]:.1f} 毫秒")
        print(f"   直径范围: {data_summary['diameter_range'][0]:.3f} - {data_summary['diameter_range'][1]:.3f} 米")
        print(f"   爆炸时长: {data_summary['explosion_duration']:.1f} 毫秒")
        
    else:
        print("❌ 拟合失败！")
        print(f"📁 输入文件: {result['file_path']}")
        print(f"💥 错误信息: {result['error']}")


if __name__ == "__main__":
    main()
