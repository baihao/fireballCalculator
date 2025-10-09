#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分割结果数据工具

将 image_sequence_segmentation 的结果转换为用于绘图的时间-直径序列。
仅保留分割成功且半径有效的点；对失败的图片不做插值，不输出数据点。
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import os
import sys
import subprocess


def _extract_radius_value(max_radius: Any) -> Optional[float]:
    """
    从max_radius字段中提取半径数值。
    支持两种格式：
    - 数字: 直接返回
    - 字典: {"value": float, "endpoint": {"x":..., "y":...}}
    返回None表示无效。
    """
    if max_radius is None:
        return None
    try:
        if isinstance(max_radius, dict):
            value = float(max_radius.get("value", 0))
        else:
            value = float(max_radius)
        if value > 0:
            return value
        return None
    except (TypeError, ValueError):
        return None


def build_time_diameter_series(
    segmentation_results: List[Dict[str, Any]],
    explosion_duration_ms: float,
) -> List[Tuple[float, float]]:
    """
    根据分割结果构建用于绘图的(时间ms, 直径m)序列。

    - 仅包含 success=True 且 max_radius 有效的结果
    - 不对失败或无效结果进行插值

    Args:
        segmentation_results: image_sequence_segmentation 数组
        explosion_duration_ms: 爆炸时长（毫秒）

    Returns:
        List[Tuple[float, float]]: 有效的数据点列表
    """
    series: List[Tuple[float, float]] = []
    total = len(segmentation_results)
    if total == 0:
        return series

    for i, result in enumerate(segmentation_results):
        # 时间轴（线性映射到时长）
        time_ms = (i / (total - 1) * explosion_duration_ms) if total > 1 else 0.0

        if not result or not result.get("success", False):
            continue

        radius_value = _extract_radius_value(result.get("max_radius"))
        if radius_value is None:
            continue

        diameter = 2.0 * radius_value
        series.append((float(time_ms), float(diameter)))

    return series



def run_segmentation_script(
    sequence_file_path: str,
    on_output_line: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    运行 image_segment/test_complete_propagation.py，并实时回调输出。

    - 合并stdout/stderr到同一流，逐行回调
    - 设置PYTHONPATH=项目source目录，cwd为image_segment目录

    Args:
        sequence_file_path: 序列JSON路径
        on_output_line: 接收实时输出的回调，可为None

    Returns:
        bool: 脚本是否成功退出（returncode==0）
    """
    try:
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'image_segment'))
        script_path = os.path.join(script_dir, 'test_complete_propagation.py')
        if not os.path.exists(script_path):
            if on_output_line:
                on_output_line(f"❌ 分割脚本不存在: {script_path}\n")
            return False

        # 环境变量
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(os.path.join(project_root, 'source'))

        # 启动子进程，合并stderr到stdout以避免读阻塞
        # 设置环境变量强制Python输出无缓冲
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(
            [sys.executable, '-u', script_path, sequence_file_path, '--no-viz'],  # -u 参数强制无缓冲输出，禁止可视化输出
            cwd=script_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # 无缓冲
            universal_newlines=True,
        )

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                if on_output_line:
                    on_output_line(line)
        process.wait()

        return process.returncode == 0
    except Exception as e:
        if on_output_line:
            on_output_line(f"❌ 执行分割脚本异常: {e}\n")
        return False

