#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球序列 / 拟合结果 JSON 的统一文件命名。

与训练数据 ``fireball_diameter_fit_{当量}_{含铝}``.json 风格一致，例如：
- ``fireball_diameter_fit_160_30.json``
- ``myseq_fireball_sequence_160_30_segmented.json``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]


def eq_al_filename_token(equivalent: Any, al_percent: Any) -> str:
    """当量、含铝量文件名 token，如 ``160_30``、``1_40``。"""
    eq = float(equivalent)
    al = float(al_percent)
    return f"{eq:g}_{al:g}"


def _parameters_from_data(sequence_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not sequence_data:
        return {}
    params = sequence_data.get("parameters")
    return params if isinstance(params, dict) else {}


def _eq_al_from_parameters(params: Dict[str, Any]) -> Optional[tuple[float, float]]:
    try:
        return float(params["equivalent"]), float(params["al_percent"])
    except (KeyError, TypeError, ValueError):
        return None


def _normalize_sequence_stem(stem: str) -> str:
    """去掉 ``_segmented`` 及尾部 ``_{当量}_{含铝}``，得到原始序列 stem。"""
    if stem.endswith("_segmented"):
        stem = stem[: -len("_segmented")]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        base, eq_part, al_part = parts
        try:
            float(eq_part)
            float(al_part)
            return base
        except ValueError:
            pass
    return stem


def fireball_diameter_fit_filename(
    equivalent: Any,
    al_percent: Any,
    *,
    directory: Optional[PathLike] = None,
) -> str:
    """拖曳拟合结果默认文件名。"""
    name = f"fireball_diameter_fit_{eq_al_filename_token(equivalent, al_percent)}.json"
    if directory is None:
        return name
    return str(Path(directory).expanduser() / name)


def segmented_sequence_filename(
    original_json_path: PathLike,
    equivalent: Any,
    al_percent: Any,
) -> Path:
    """分割结果 JSON 路径（与源序列同目录）。"""
    p = Path(original_json_path).expanduser()
    stem = _normalize_sequence_stem(p.stem)
    token = eq_al_filename_token(equivalent, al_percent)
    return p.with_name(f"{stem}_{token}_segmented{p.suffix or '.json'}")


def segmented_sequence_filename_from_data(
    original_json_path: PathLike,
    sequence_data: Dict[str, Any],
) -> Path:
    """从序列 ``parameters`` 解析当量/含铝并生成分割结果路径。"""
    params = _parameters_from_data(sequence_data)
    pair = _eq_al_from_parameters(params)
    if pair is None:
        p = Path(original_json_path).expanduser()
        return p.with_name(f"{p.stem}_segmented{p.suffix or '.json'}")
    eq, al = pair
    return segmented_sequence_filename(original_json_path, eq, al)


def legacy_segmented_sequence_filename(original_json_path: PathLike) -> Path:
    """旧版命名：``{stem}_segmented.json``。"""
    p = Path(original_json_path).expanduser()
    stem = _normalize_sequence_stem(p.stem)
    return p.with_name(f"{stem}_segmented{p.suffix or '.json'}")


def candidate_segmented_sequence_paths(
    original_json_path: PathLike,
    sequence_data: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """按优先级返回可能的分割结果路径（新命名 → 旧命名）。"""
    p = Path(original_json_path).expanduser()
    candidates: List[Path] = []
    if sequence_data is not None:
        candidates.append(segmented_sequence_filename_from_data(p, sequence_data))
    candidates.append(legacy_segmented_sequence_filename(p))
    seen = set()
    ordered: List[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.is_absolute() else str(c)
        if key not in seen:
            seen.add(key)
            ordered.append(c)
    return ordered


def find_existing_segmented_sequence_path(
    original_json_path: PathLike,
    sequence_data: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """查找已存在的分割结果 JSON（兼容新旧命名）。"""
    for path in candidate_segmented_sequence_paths(original_json_path, sequence_data):
        if path.is_file():
            return path
    return None


def preferred_segmented_sequence_path(
    original_json_path: PathLike,
    sequence_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """写入或提示时期望的分割结果路径（不存在时也返回新命名）。"""
    if sequence_data is not None:
        params = _parameters_from_data(sequence_data)
        if _eq_al_from_parameters(params) is not None:
            return segmented_sequence_filename_from_data(original_json_path, sequence_data)
    return legacy_segmented_sequence_filename(original_json_path)
