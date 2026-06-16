#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据目录导入：扫描 `*.json`，解析 ``parameters`` / ``drag_fit``，填充 ``TrainingExperimentRecord``。

与同仓库 ``gp_model.data_input`` 对齐；跳过 ``trained_hyperparams.json``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from gp_model.config import DEFAULT_TRAINED_FILENAME

from ..training_dataset_model import TrainingExperimentRecord


@dataclass
class TrainingFolderImportResult:
    ok: bool
    records: List[TrainingExperimentRecord] = field(default_factory=list)
    folder_resolved: str = ""
    error_message: str = ""
    diagnostics: List[str] = field(default_factory=list)


def _read_json_file(path: Path) -> dict:
    """读取 JSON；Windows 上部分文件为 GBK/ANSI，需多编码尝试。"""
    raw: str | None = None
    last_decode_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError as e:
            last_decode_error = e
    if raw is None:
        raise last_decode_error or UnicodeDecodeError("unknown", b"", 0, 1, "decode failed")
    return json.loads(raw)


def _parse_one_drag_fit_sample(
    path: Path,
    *,
    strict_drag_fit_success: bool,
) -> tuple[TrainingExperimentRecord | None, str | None]:
    """
    返回 (record, skip_reason)。
    skip_reason 非空表示跳过该文件；strict 时为致命错误语义由调用方转为 error_message。
    """
    try:
        data = _read_json_file(path)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        reason = f"{path.name}: JSON 读取失败 ({e})"
        if strict_drag_fit_success:
            raise RuntimeError(reason) from e
        return None, reason

    params = data.get("parameters") or {}
    drag = data.get("drag_fit") or {}

    if not isinstance(params, dict) or not isinstance(drag, dict):
        reason = f"{path.name}: 缺少 parameters 或 drag_fit"
        if strict_drag_fit_success:
            raise RuntimeError(reason)
        return None, reason

    if drag.get("success") is not True:
        reason = f"{path.name}: drag_fit.success 不为 true"
        if strict_drag_fit_success:
            raise RuntimeError(reason)
        return None, reason

    try:
        eq = float(params.get("equivalent"))
        al = float(params.get("al_percent"))
    except (TypeError, ValueError):
        reason = f"{path.name}: equivalent / al_percent 无效"
        if strict_drag_fit_success:
            raise RuntimeError(reason) from None
        return None, reason

    try:
        K = float(drag["K"])
        B = float(drag["B"])
        C = float(drag["C"])
    except (KeyError, TypeError, ValueError):
        reason = f"{path.name}: K,B,C 缺失或无效"
        if strict_drag_fit_success:
            raise RuntimeError(reason) from None
        return None, reason

    return (
        TrainingExperimentRecord(
            source_path=str(path.resolve()),
            equivalent_kg_tnt=eq,
            al_percent=al,
            K=K,
            B=B,
            C=C,
        ),
        None,
    )


def _iter_json_files(root: Path, *, recursive: bool) -> Sequence[Path]:
    if recursive:
        files = sorted(root.rglob("*.json"))
    else:
        files = sorted(root.glob("*.json"))
    out: List[Path] = []
    for fp in files:
        if fp.name == DEFAULT_TRAINED_FILENAME:
            continue
        out.append(fp)
    return out


def import_training_folder(
    folder: str | Path,
    *,
    recursive: bool = True,
    strict_drag_fit_success: bool = False,
) -> TrainingFolderImportResult:
    """
    从目录加载多组 ``fireball_diameter_fit*.json`` 风格样本。

    Args:
        folder: 用户选择的文件夹。
        recursive: 是否递归子目录（默认 True，适配「多组数据」分_subdirectory 存放）。
        strict_drag_fit_success: True 时在首个不合规文件上抛语义错误（暂未对 UI 开放）。
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return TrainingFolderImportResult(
            ok=False,
            folder_resolved=str(root),
            error_message=f"不是有效目录: {root}",
        )

    records: List[TrainingExperimentRecord] = []
    diag: List[str] = []

    try:
        for fp in _iter_json_files(root, recursive=recursive):
            rec, skip = _parse_one_drag_fit_sample(
                fp, strict_drag_fit_success=strict_drag_fit_success
            )
            if rec is not None:
                records.append(rec)
            elif skip:
                diag.append(skip)
    except RuntimeError as e:
        return TrainingFolderImportResult(
            ok=False,
            folder_resolved=str(root),
            error_message=str(e),
            diagnostics=diag,
        )

    if not records:
        return TrainingFolderImportResult(
            ok=False,
            folder_resolved=str(root),
            diagnostics=diag,
            error_message=(
                f"目录中未解析到有效训练样本（需要 drag_fit.success=true 且含 K,B,C；"
                f"共扫描 {len(diag)} 个跳过说明）。"
            ),
        )

    return TrainingFolderImportResult(
        ok=True,
        records=records,
        folder_resolved=str(root),
        diagnostics=diag,
    )


def export_training_dataset(_model, _path: str) -> None:
    raise NotImplementedError
