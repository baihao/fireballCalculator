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
import csv
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
source_path = os.path.join(project_root, 'source')
sys.path.insert(0, source_path)
# 确保 desktop 目录也在路径中，以便能找到 framework 模块
desktop_path = os.path.join(source_path, 'desktop')
if desktop_path not in sys.path:
    sys.path.insert(0, desktop_path)

from diameter_drag_fitting import DiameterDragFitter
from drag_fit_plotter import DragFitPlotter
from csv_data_loader import load_csv_data
from farthest_point_extractor import extract_farthest_point_data


def load_segmented_sequence(file_path: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    加载分割序列文件
    
    Args:
        file_path: 分割序列JSON文件路径
        
    Returns:
        Tuple[bool, Dict[str, Any], str]: (是否成功, 序列数据, 错误信息)
    """
    # 延迟导入，只在需要时导入
    from desktop.extract_tab.utils.sequence_manager import SequenceManager
    
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
    # 延迟导入，只在需要时导入
    from desktop.extract_tab.utils.sequence_manager import SequenceManager
    from desktop.extract_tab.utils.segment_utils import build_time_diameter_series
    
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


def _perform_drag_fitting(time_data: List[float], diameter_data: List[float],
                          file_path: str, output_dir: Optional[str],
                          use_robust_fitting: bool, is_csv_input: bool = False) -> Dict[str, Any]:
    """
    执行拖曳曲线拟合和绘制结果的共同方法
    
    Args:
        time_data: 时间数据列表（毫秒）
        diameter_data: 直径数据列表（米）
        file_path: 输入文件路径（用于生成输出文件名）
        output_dir: 输出目录（可选）
        use_robust_fitting: 是否使用鲁棒拟合
        is_csv_input: 是否为CSV输入（如果是，则输出拟合后的CSV文件）
        
    Returns:
        Dict[str, Any]: 拟合结果
    """
    # 执行拟合（启用数据过滤）
    print(f"\n执行拖曳曲线拟合...")
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
    
    # 如果是CSV输入，计算拟合值并保存CSV
    csv_output_path = None
    if is_csv_input:
        print(f"\n生成拟合后的CSV文件...")
        if output_dir is None:
            output_dir = os.path.dirname(file_path) or '.'
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_output_path = os.path.join(output_dir, f"{base_name}_fitted.csv")
        
        # 计算拟合值
        t_array = np.array(time_data)
        K = fit_result['K']
        B = fit_result['B']
        C = fit_result['C']
        fitted_diameter = DiameterDragFitter.drag_function(t_array, K, B, C)
        
        # 保存CSV文件（只包含拟合数据）
        try:
            with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['时间(ms)', '拟合直径(m)'])
                for i in range(len(time_data)):
                    writer.writerow([
                        f'{time_data[i]:.6f}',
                        f'{fitted_diameter[i]:.6f}'
                    ])
            print(f"✓ 拟合后的CSV文件已保存: {csv_output_path}")
        except Exception as e:
            print(f"⚠️ 保存CSV文件失败: {str(e)}")
            csv_output_path = None
    
    # 绘制结果
    print(f"\n绘制拟合结果...")
    if output_dir is None:
        output_dir = os.path.dirname(file_path) or '.'
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_path = os.path.join(output_dir, f"{base_name}_drag_fit.png")
    
    plotter = DragFitPlotter()
    plot_success = plotter.plot_fit_results(time_data, diameter_data, fit_result, plot_path, time_unit='ms')
    
    if plot_success:
        print(f"✓ 拟合结果图已保存: {plot_path}")
    else:
        print(f"⚠️ 保存拟合结果图失败")
    
    # 返回完整结果
    result = {
        'success': True,
        'file_path': file_path,
        'plot_path': plot_path if plot_success else None,
        'csv_output_path': csv_output_path,
        'fit_result': fit_result,
        'data_summary': {
            'data_points': len(time_data),
            'time_range': [min(time_data), max(time_data)],
            'diameter_range': [min(diameter_data), max(diameter_data)],
            'explosion_duration': max(time_data) - min(time_data)
        }
    }
    
    return result


def fit_drag_curve_from_sequence(file_path: str, output_dir: Optional[str] = None, 
                                use_robust_fitting: bool = True,
                                output_farthest_point: bool = False) -> Dict[str, Any]:
    """
    从分割序列文件拟合拖曳曲线
    
    Args:
        file_path: 分割序列JSON文件路径
        output_dir: 输出目录（可选）
        use_robust_fitting: 是否使用鲁棒拟合
        output_farthest_point: 是否输出最远点坐标CSV文件
        
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
        
        # 2. 如果指定了输出最远点，先提取最远点数据
        farthest_point_csv_path = None
        if output_farthest_point:
            print(f"\n2. 提取最远点数据...")
            farthest_point_csv_path = extract_farthest_point_data(sequence_data, output_dir, file_path)
        
        # 3. 提取直径数据
        step_num = 3 if output_farthest_point else 2
        print(f"\n{step_num}. 提取直径数据...")
        time_data, diameter_data, extract_message = extract_diameter_data(sequence_data)
        if not time_data or not diameter_data:
            return {
                'success': False,
                'error': f"提取直径数据失败: {extract_message}",
                'file_path': file_path
            }
        
        print(f"✓ {extract_message}")
        
        # 4. 执行拟合和绘制（调用共同方法）
        result = _perform_drag_fitting(time_data, diameter_data, file_path, output_dir, use_robust_fitting, is_csv_input=False)
        
        # 添加最远点CSV路径到结果中
        if farthest_point_csv_path:
            result['farthest_point_csv_path'] = farthest_point_csv_path
        
        return result
        
    except Exception as e:
            return {
                'success': False,
            'error': f"拟合过程异常: {str(e)}",
            'file_path': file_path
        }


def fit_drag_curve_from_csv(file_path: str, output_dir: Optional[str] = None,
                            use_robust_fitting: bool = True) -> Dict[str, Any]:
    """
    从CSV文件拟合拖曳曲线
    
    Args:
        file_path: CSV文件路径（格式：时间(ms),直径(m)）
        output_dir: 输出目录（可选）
        use_robust_fitting: 是否使用鲁棒拟合
        
    Returns:
        Dict[str, Any]: 拟合结果
    """
    try:
        print("=" * 60)
        print("火球直径拖曳曲线拟合（CSV数据）")
        print("=" * 60)
        
        # 1. 加载CSV文件
        print(f"\n1. 加载CSV文件: {file_path}")
        time_data, diameter_data, load_message = load_csv_data(file_path)
        if not time_data or not diameter_data:
            return {
                'success': False,
                'error': f"加载CSV文件失败: {load_message}",
                'file_path': file_path
            }
        
        print(f"✓ {load_message}")
        
        # 2. 执行拟合和绘制（调用共同方法，标记为CSV输入）
        result = _perform_drag_fitting(time_data, diameter_data, file_path, output_dir, use_robust_fitting, is_csv_input=True)
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
        print("用法: python fit_segmented_sequence.py <input_file> [output_dir] [--robust] [-outputFarthestPoint]")
        print("")
        print("参数:")
        print("  input_file: 输入文件路径（支持JSON或CSV格式）")
        print("    - JSON格式: 分割序列JSON文件")
        print("    - CSV格式: 包含时间(ms)和直径(m)列的CSV文件")
        print("  output_dir: 输出目录（可选，默认为输入文件所在目录）")
        print("  --robust: 使用鲁棒拟合（可选，默认为True）")
        print("  -outputFarthestPoint: 输出每张图片的最远点坐标和像素半径CSV（仅适用于JSON格式）")
        print("")
        print("示例:")
        print("  python fit_segmented_sequence.py test_data/fireball_sequence_segmented.json")
        print("  python fit_segmented_sequence.py experiment/diameter.csv ./output")
        print("  python fit_segmented_sequence.py test_data/fireball_sequence_segmented.json ./output --robust")
        print("  python fit_segmented_sequence.py test_data/fireball_sequence_segmented.json ./output -outputFarthestPoint")
        return
    
    # 解析命令行参数
    file_path = sys.argv[1]
    output_dir = None
    use_robust = True
    output_farthest_point = False
    
    # 解析参数
    for arg in sys.argv[2:]:
        if arg.startswith('--'):
            if arg == '--robust':
                use_robust = True
            elif arg == '--no-robust':
                use_robust = False
        elif arg == '-outputFarthestPoint':
            output_farthest_point = True
        elif not arg.startswith('-'):
            # 非选项参数，作为输出目录
            output_dir = arg
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 根据文件扩展名选择处理方式
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # CSV文件：使用CSV拟合函数
        if output_farthest_point:
            print("⚠️ 警告: -outputFarthestPoint 参数仅适用于JSON格式文件，已忽略")
        result = fit_drag_curve_from_csv(file_path, output_dir, use_robust)
    elif file_ext == '.json':
        # JSON文件：使用序列拟合函数
        result = fit_drag_curve_from_sequence(file_path, output_dir, use_robust, output_farthest_point)
    else:
        # 尝试自动检测：先尝试CSV，如果失败再尝试JSON
        print(f"⚠️ 未识别的文件扩展名 '{file_ext}'，尝试自动检测...")
        time_data, diameter_data, _ = load_csv_data(file_path)
        if time_data and diameter_data:
            print("✓ 检测为CSV格式")
            if output_farthest_point:
                print("⚠️ 警告: -outputFarthestPoint 参数仅适用于JSON格式文件，已忽略")
            result = fit_drag_curve_from_csv(file_path, output_dir, use_robust)
        else:
            print("✓ 检测为JSON格式")
            result = fit_drag_curve_from_sequence(file_path, output_dir, use_robust, output_farthest_point)
    
    # 输出结果
    print("\n" + "=" * 60)
    if result['success']:
        print("✅ 拟合完成！")
        print(f"📁 输入文件: {result['file_path']}")
        if result.get('plot_path'):
            print(f"📊 结果图表: {result['plot_path']}")
        if result.get('csv_output_path'):
            print(f"📄 拟合CSV文件: {result['csv_output_path']}")
        if result.get('farthest_point_csv_path'):
            print(f"📍 最远点CSV文件: {result['farthest_point_csv_path']}")
        
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
