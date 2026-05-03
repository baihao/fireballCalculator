#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像几何信息（宽度等），供视场 → pixel_length 换算。"""

from __future__ import annotations

from typing import Optional


def get_image_width_pixels(path: str) -> Optional[int]:
    """
    读取图像文件宽度（像素）。失败返回 None。
    """
    if not path:
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, _ = im.size
            return int(w) if w > 0 else None
    except Exception:
        pass
    try:
        from PySide6.QtGui import QImage

        img = QImage(path)
        if not img.isNull():
            w = img.width()
            return int(w) if w > 0 else None
    except Exception:
        pass
    return None
