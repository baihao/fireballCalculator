#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径数据过滤模块

基于物理约束过滤烟雾干扰数据，保留物理合理的膨胀阶段。
"""

import numpy as np
from typing import List, Tuple, Optional


def filter_smoke_interference_data(time_data: List[float], diameter_data: List[float], 
                                 drop_threshold: float = 0.02, window_size: int = 10) -> List[float]:
    """
    过滤烟雾干扰数据，返回需要截断的时间点
    
    策略：
    1. 找到全局最大直径点
    2. 使用滑动窗口平滑数据
    3. 检测最大直径后的显著下降（默认2%）
    4. 返回截断时间点
    
    Args:
        time_data: 时间数据列表（毫秒）
        diameter_data: 直径数据列表（米）
        drop_threshold: 下降阈值（默认2%）
        window_size: 滑动窗口大小（默认10）
        
    Returns:
        List[float]: 截断时间点列表，如果不需要截断则返回空列表
    """
    try:
        if len(time_data) != len(diameter_data):
            raise ValueError("时间和直径数据长度不匹配")
        
        if len(time_data) < window_size + 5:
            print("⚠️ 数据点太少，无法进行有效过滤")
            return []
        
        t = np.array(time_data)
        D = np.array(diameter_data)
        
        # 1. 找到全局最大直径点
        max_idx = np.argmax(D)
        max_diameter = D[max_idx]
        max_time = t[max_idx]
        
        print(f"全局最大直径: {max_diameter:.2f}m 在时间 {max_time:.1f}ms")
        
        # 2. 如果最大值在数据末尾附近，不需要过滤
        if max_idx >= len(t) - window_size:
            print("最大直径在数据末尾，无需过滤")
            return []
        
        # 3. 获取最大直径后的数据
        t_after = t[max_idx:]
        D_after = D[max_idx:]
        
        # 4. 计算滑动平均
        smoothed_D = _calculate_sliding_average(D_after, window_size)
        
        # 5. 检测显著下降
        cutoff_times = []
        for i in range(len(smoothed_D)):
            relative_drop = (max_diameter - smoothed_D[i]) / max_diameter
            if relative_drop > drop_threshold:
                # 找到第一个显著下降点
                cutoff_time = t_after[i]
                cutoff_times.append(cutoff_time)
                print(f"检测到烟雾干扰: 时间 {cutoff_time:.1f}ms, 下降 {relative_drop:.1%}")
                break
        
        return cutoff_times
        
    except Exception as e:
        print(f"⚠️ 数据过滤失败: {e}")
        return []


def _calculate_sliding_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    计算滑动平均
    
    Args:
        data: 输入数据
        window_size: 窗口大小
        
    Returns:
        np.ndarray: 滑动平均结果
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return np.array(smoothed)


def apply_data_filter(time_data: List[float], diameter_data: List[float],
                     drop_threshold: float = 0.02, window_size: int = 10) -> Tuple[List[float], List[float]]:
    """
    应用数据过滤，返回过滤后的数据
    
    Args:
        time_data: 时间数据列表（毫秒）
        diameter_data: 直径数据列表（米）
        drop_threshold: 下降阈值（默认2%）
        window_size: 滑动窗口大小（默认10）
        
    Returns:
        Tuple[List[float], List[float]]: (过滤后的时间数据, 过滤后的直径数据)
    """
    cutoff_times = filter_smoke_interference_data(time_data, diameter_data, drop_threshold, window_size)
    
    if not cutoff_times:
        print("未检测到烟雾干扰，保留所有数据")
        return time_data, diameter_data
    
    # 使用第一个截断点
    cutoff_time = cutoff_times[0]
    
    # 过滤数据
    filtered_time = []
    filtered_diameter = []
    
    for t, d in zip(time_data, diameter_data):
        if t <= cutoff_time:
            filtered_time.append(t)
            filtered_diameter.append(d)
        else:
            break
    
    print(f"数据过滤: 保留 {len(filtered_time)} 个数据点（截断时间: {cutoff_time:.1f}ms）")
    
    return filtered_time, filtered_diameter


def analyze_data_phases(time_data: List[float], diameter_data: List[float]) -> dict:
    """
    分析数据的不同阶段
    
    Args:
        time_data: 时间数据列表（毫秒）
        diameter_data: 直径数据列表（米）
        
    Returns:
        dict: 阶段分析结果
    """
    try:
        t = np.array(time_data)
        D = np.array(diameter_data)
        
        # 找到最大直径点
        max_idx = np.argmax(D)
        max_diameter = D[max_idx]
        max_time = t[max_idx]
        
        # 分析阶段
        phases = {
            'total_duration': t[-1] - t[0],
            'max_diameter': max_diameter,
            'max_time': max_time,
            'max_index': max_idx,
            'initial_diameter': D[0],
            'final_diameter': D[-1],
            'diameter_range': max_diameter - D[0],
            'growth_phase': {
                'start_time': t[0],
                'end_time': max_time,
                'duration': max_time - t[0],
                'diameter_change': max_diameter - D[0]
            },
            'post_max_phase': {
                'start_time': max_time,
                'end_time': t[-1],
                'duration': t[-1] - max_time,
                'diameter_change': D[-1] - max_diameter
            }
        }
        
        return phases
        
    except Exception as e:
        print(f"⚠️ 阶段分析失败: {e}")
        return {}


if __name__ == "__main__":
    print("火球直径数据过滤模块")
    print("使用方法:")
    print("  from data_filter import filter_smoke_interference_data, apply_data_filter")
    print("  cutoff_times = filter_smoke_interference_data(time_data, diameter_data)")
    print("  filtered_time, filtered_diameter = apply_data_filter(time_data, diameter_data)")
