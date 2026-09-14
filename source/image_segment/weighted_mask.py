#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""浮点权重掩码：传播融合、高斯平滑与展示用二值/轮廓导出。"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

DEFAULT_MASK_FUSION_ALPHA = 0.15
DISPLAY_GAUSSIAN_SIGMA = 5.0
DISPLAY_WEIGHT_THRESHOLD = 0.5


def _gaussian_ksize(sigma: float) -> int:
    k = int(max(3, 2 * np.ceil(3.0 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    return k


def resize_mask_weights(
    weights: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """将参考帧权重图缩放到目标 (H, W)。"""
    th, tw = target_shape
    if weights.shape[0] == th and weights.shape[1] == tw:
        return weights.astype(np.float32, copy=False)
    return cv2.resize(
        weights.astype(np.float32),
        (tw, th),
        interpolation=cv2.INTER_LINEAR,
    )


def fuse_sam_with_reference_weights(
    sam_mask: np.ndarray,
    reference_weights: np.ndarray,
    alpha: float = DEFAULT_MASK_FUSION_ALPHA,
) -> np.ndarray:
    """
    p_t = alpha * p_sam + (1 - alpha) * p_o
    sam_mask: 当前帧 SAM 二值掩码；reference_weights: 已对齐到目标尺寸的 p_o。
    """
    p_sam = sam_mask.astype(np.float32)
    if p_sam.max() > 1.0:
        p_sam = (p_sam > 0).astype(np.float32)
    p_o = np.clip(reference_weights.astype(np.float32), 0.0, 1.0)
    fused = alpha * p_sam + (1.0 - alpha) * p_o
    return np.clip(fused, 0.0, 1.0)


def sam_mask_to_weights(sam_mask: np.ndarray) -> np.ndarray:
    w = sam_mask.astype(np.float32)
    if w.max() > 1.0:
        w = (w > 0).astype(np.float32)
    return np.clip(w, 0.0, 1.0)


def _largest_foreground_component(binary: np.ndarray) -> np.ndarray:
    """保留面积最大的前景连通域。"""
    bin_u8 = (binary > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_u8, connectivity=8
    )
    if num_labels <= 1:
        return bin_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8)


def weights_to_display_mask_and_contour(
    weights: np.ndarray,
    sigma: float = DISPLAY_GAUSSIAN_SIGMA,
    threshold: float = DISPLAY_WEIGHT_THRESHOLD,
    keep_largest_component: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    权重图 → 高斯平滑 → 阈值 → 可选最大连通域 → 展示二值 mask + 轮廓信息。
    """
    if weights is None or weights.size == 0:
        empty = {
            "display_mask": None,
            "blurred_weights": None,
            "contour": None,
            "area": 0.0,
            "centroid": (0.0, 0.0),
        }
        return None, empty

    w = np.clip(weights.astype(np.float32), 0.0, 1.0)
    ksize = _gaussian_ksize(sigma)
    blurred = cv2.GaussianBlur(w, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    binary = (blurred >= threshold).astype(np.uint8)
    if keep_largest_component and np.any(binary):
        binary = _largest_foreground_component(binary)

    area = float(np.sum(binary))
    contour = None
    centroid = (0.0, 0.0)
    if area > 0:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(binary)
            if moments["m00"] > 0:
                centroid = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )

    display_mask = binary.astype(bool)
    details = {
        "display_mask": display_mask,
        "blurred_weights": blurred,
        "contour": contour,
        "area": area,
        "centroid": centroid,
        "threshold": threshold,
        "sigma": sigma,
    }
    return display_mask, details


def create_weighted_mask_processor(
    alpha: float = DEFAULT_MASK_FUSION_ALPHA,
    sigma: float = DISPLAY_GAUSSIAN_SIGMA,
    threshold: float = DISPLAY_WEIGHT_THRESHOLD,
):
    """便于在 segmenter 中注入默认参数。"""

    _alpha, _sigma, _threshold = alpha, sigma, threshold

    class _Processor:
        alpha = _alpha
        sigma = _sigma
        threshold = _threshold

        resize_mask_weights = staticmethod(resize_mask_weights)
        fuse_sam_with_reference_weights = staticmethod(fuse_sam_with_reference_weights)
        sam_mask_to_weights = staticmethod(sam_mask_to_weights)
        weights_to_display_mask_and_contour = staticmethod(
            weights_to_display_mask_and_contour
        )

    return _Processor()
