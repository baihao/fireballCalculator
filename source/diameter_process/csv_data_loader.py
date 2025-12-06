#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据加载模块

提供从CSV文件加载时间和直径数据的功能
"""

import csv
from typing import List, Tuple


def load_csv_data(file_path: str) -> Tuple[List[float], List[float], str]:
    """
    从CSV文件加载时间和直径数据
    
    Args:
        file_path: CSV文件路径，格式应为：时间(ms),直径(m)
        
    Returns:
        Tuple[List[float], List[float], str]: (时间数据, 直径数据, 错误信息)
    """
    try:
        time_data = []
        diameter_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # 读取表头
            header = next(reader, None)
            if header is None:
                return [], [], "CSV文件为空"
            
            # 检查表头格式（支持中英文）
            header_str = ','.join(header).lower()
            if '时间' not in header_str and 'time' not in header_str:
                return [], [], "CSV文件缺少时间列"
            if '直径' not in header_str and 'diameter' not in header_str:
                return [], [], "CSV文件缺少直径列"
            
            # 读取数据行
            for row_num, row in enumerate(reader, start=2):
                if len(row) < 2:
                    continue  # 跳过空行或格式不正确的行
                
                try:
                    time_val = float(row[0].strip())
                    diameter_val = float(row[1].strip())
                    
                    # 验证数据有效性
                    if time_val < 0:
                        print(f"⚠️ 警告: 第{row_num}行时间值为负数，已跳过")
                        continue
                    if diameter_val <= 0:
                        print(f"⚠️ 警告: 第{row_num}行直径值无效，已跳过")
                        continue
                    
                    time_data.append(time_val)
                    diameter_data.append(diameter_val)
                    
                except ValueError as e:
                    print(f"⚠️ 警告: 第{row_num}行数据格式错误，已跳过: {row}")
                    continue
        
        if not time_data or not diameter_data:
            return [], [], "CSV文件中没有有效的数据点"
        
        print(f"✓ 从CSV文件读取到 {len(time_data)} 个有效数据点")
        print(f"✓ 时间范围: {min(time_data):.1f} - {max(time_data):.1f} 毫秒")
        print(f"✓ 直径范围: {min(diameter_data):.3f} - {max(diameter_data):.3f} 米")
        
        return time_data, diameter_data, "加载成功"
        
    except FileNotFoundError:
        return [], [], f"CSV文件不存在: {file_path}"
    except Exception as e:
        return [], [], f"读取CSV文件失败: {str(e)}"

