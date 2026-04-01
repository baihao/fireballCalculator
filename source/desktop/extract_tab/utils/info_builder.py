#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息描述文案生成器

将 UI 中的说明文字拼装逻辑集中管理，降低上层组件的复杂度。
"""
from typing import Dict, Optional, Tuple, List


def build_prompt_info_text(prompt_data: Dict[int, Dict[str, List[List[int]]]],
                           ignition_point: Optional[Tuple[int, int]]) -> str:
    """
    生成参考点信息面板的文本内容。

    Args:
        prompt_data: {image_index: {"points": [[x,y], ...], "labels": [1/0, ...]}}
        ignition_point: 起爆点 (x, y) 或 None

    Returns:
        str: 多行说明文本
    """
    if not prompt_data and not ignition_point:
        return "暂无选择的参考点\n\n提示：\n1. 先加载图像序列\n2. 点击「开始选择参考点」\n3. 在图像上点击选择正负点\n4. 点击「参考点选择完成」保存到序列文件"

    lines: List[str] = []

    # 起爆点信息
    if ignition_point:
        lines.append("🎯 起爆点信息：")
        lines.append(f"  - 坐标：({ignition_point[0]}, {ignition_point[1]})")
        lines.append("")

    # 各图像参考点
    if prompt_data:
        for image_idx in sorted(prompt_data.keys()):
            points = prompt_data[image_idx].get("points", [])
            labels = prompt_data[image_idx].get("labels", [])

            positive_points: List[str] = []
            negative_points: List[str] = []
            for point, label in zip(points, labels):
                if label == 1:
                    positive_points.append(f"({point[0]}, {point[1]})")
                else:
                    negative_points.append(f"({point[0]}, {point[1]})")

            lines.append(f"第 {image_idx + 1} 张图片参考点：")
            if positive_points:
                lines.append(f"  - 正点坐标：{', '.join(positive_points)}")
            if negative_points:
                lines.append(f"  - 负点坐标：{', '.join(negative_points)}")
            lines.append("")

    # 统计信息
    total_images_with_prompts = len(prompt_data) if prompt_data else 0
    total_points = sum(len(d.get("points", [])) for d in prompt_data.values()) if prompt_data else 0
    total_positive = sum(sum(1 for lb in d.get("labels", []) if lb == 1) for d in prompt_data.values()) if prompt_data else 0
    total_negative = total_points - total_positive

    lines.append("=" * 30)
    lines.append("统计信息：")
    if total_images_with_prompts > 0:
        lines.append(f"  - 有参考点的图像：{total_images_with_prompts} 张")
        lines.append(f"  - 总点数：{total_points} 个")
        lines.append(f"  - 正点：{total_positive} 个")
        lines.append(f"  - 负点：{total_negative} 个")
    if ignition_point:
        lines.append("  - 起爆点：1 个")

    return "\n".join(lines)


def build_segmentation_info_text(successful_count: int, total_count: int) -> str:
    """
    生成分割结果信息面板的文本内容。

    Args:
        successful_count: 成功分割数量
        total_count: 总图像数量

    Returns:
        str: 多行说明文本
    """
    lines: List[str] = []
    lines.append("🎯 分割结果信息")
    lines.append("=" * 30)
    lines.append(f"总图片数: {total_count}")
    lines.append(f"成功分割: {successful_count}")
    lines.append(f"失败分割: {total_count - successful_count}")
    ratio = (successful_count / total_count * 100.0) if total_count else 0.0
    lines.append(f"成功率: {ratio:.1f}%")
    lines.append("")
    lines.append("💡 当前显示分割结果（轮廓+质心+半径）")
    lines.append("   - 蓝色轮廓: 火球边界")
    lines.append("   - 绿色质心: 爆心位置")
    lines.append("   - 绿色箭头: 最大半径方向")
    return "\n".join(lines)


