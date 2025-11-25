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
from contextlib import redirect_stdout

# 项目根目录常量（从当前文件位置向上4级：utils -> extract_tab -> desktop -> source -> 项目根）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 添加 source 目录到路径，以便导入 image_segment 模块
SOURCE_DIR = os.path.join(PROJECT_ROOT, 'source')
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)


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
    pixel_length: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    根据分割结果构建用于绘图的(时间ms, 直径m)序列。

    - 仅包含 success=True 且 max_radius 有效的结果
    - 不对失败或无效结果进行插值

    Args:
        segmentation_results: image_sequence_segmentation 数组
        explosion_duration_ms: 爆炸时长（毫秒）
        pixel_length: 每像素代表的长度（米），默认为1.0

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

        # 半径需要乘以每像素代表的长度，转换为实际物理单位（米）
        radius_in_meters = radius_value * pixel_length
        diameter = 2.0 * radius_in_meters
        series.append((float(time_ms), float(diameter)))

    return series



class LogCaptureStream:
    """自定义流类，用于实时捕获 print 输出并调用回调函数"""
    def __init__(self, callback: Optional[Callable[[str], None]], original_stdout):
        self.callback = callback
        self.original_stdout = original_stdout
        self._buffer = ''  # 用于缓冲不完整的行
    
    def write(self, text: str) -> int:
        """写入文本时，同时写入原始流和调用回调"""
        # 写入原始流（保持正常输出）
        if self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()
        
        # 调用回调函数实时传递日志
        if self.callback and text:
            # 将新文本添加到缓冲区
            self._buffer += text
            
            # 按行分割并逐行回调
            while '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                if self.callback:
                    self.callback(line + '\n')
            
            # 如果缓冲区还有内容但没有换行符，可能是未完成的行，暂时不发送
            # 等待更多内容或 flush 时再发送
        
        return len(text)
    
    def flush(self):
        """刷新流，发送缓冲区中的剩余内容"""
        if self.original_stdout:
            self.original_stdout.flush()
        
        # 发送缓冲区中剩余的内容（如果有）
        if self._buffer and self.callback:
            # 如果缓冲区内容不以换行符结尾，添加换行符以保持一致性
            output = self._buffer if self._buffer.endswith('\n') else self._buffer + '\n'
            self.callback(output)
            self._buffer = ''


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
        # 使用项目根目录常量计算脚本路径
        script_dir = os.path.join(PROJECT_ROOT, 'source', 'image_segment')
        script_path = os.path.join(script_dir, 'test_complete_propagation.py')
        if not os.path.exists(script_path):
            if on_output_line:
                on_output_line(f"❌ 分割脚本不存在: {script_path}\n")
            return False

        # 环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(PROJECT_ROOT, 'source')

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


def run_segmentation_direct(
    sequence_file_path: str,
    on_output_line: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    直接调用 test_from_json 函数进行分割，并实时回调输出。

    Args:
        sequence_file_path: 序列JSON路径
        on_output_line: 接收实时输出的回调，可为None

    Returns:
        bool: 分割是否成功
    """
    try:
        # 导入 test_from_json 函数
        from image_segment.test_complete_propagation import test_from_json
        
        # 验证文件是否存在
        if not os.path.exists(sequence_file_path):
            if on_output_line:
                on_output_line(f"❌ 序列文件不存在: {sequence_file_path}\n")
            return False
        
        # 创建日志捕获流
        original_stdout = sys.stdout
        log_stream = LogCaptureStream(on_output_line, original_stdout)
        
        # 重定向 stdout 以捕获所有 print 输出
        with redirect_stdout(log_stream):
            # 直接调用 test_from_json（禁用可视化，启用快速模式）
            success = test_from_json(
                json_path=sequence_file_path,
                generate_visualization=False,  # 桌面应用不需要可视化
                output_dir="json_test_output",
                fast_mode=True  # 默认启用快速模式
            )
        
        return success
        
    except ImportError as e:
        error_msg = f"❌ 导入分割模块失败: {e}\n"
        if on_output_line:
            on_output_line(error_msg)
        print(error_msg, file=sys.stderr)
        return False
    except Exception as e:
        error_msg = f"❌ 执行分割异常: {e}\n"
        if on_output_line:
            on_output_line(error_msg)
        import traceback
        traceback.print_exc()
        return False

