#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像读写工具（兼容中文/非 ASCII 路径）

说明：
- Windows 下部分 OpenCV 版本对包含中文/特殊字符的路径使用 cv2.imread 会返回 None。
- 这里统一使用 np.fromfile + cv2.imdecode 的方式读取，供 image_segment 下其他模块复用。
"""

import cv2
import numpy as np
from typing import Optional


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    兼容中文/非 ASCII 路径的图像读取。

    Args:
        path: 图像文件路径
        flags: OpenCV 读取标志，例如 cv2.IMREAD_COLOR / cv2.IMREAD_GRAYSCALE

    Returns:
        np.ndarray 或 None：成功返回图像数组，失败返回 None
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


