#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据集会话状态：单次实验字段 + 聚合统计，供侧栏概要、图表与训练桥接读取。
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class TrainingExperimentRecord:
    """目录下单个 JSON 样本（与 `gp_model.data_input` 解析约定一致）。"""

    source_path: str
    equivalent_kg_tnt: float
    al_percent: float
    K: float
    B: float
    C: float


class TrainingDatasetModel:
    """UI 与图表共用的轻量模型。"""

    def __init__(self) -> None:
        self.records: List[TrainingExperimentRecord] = []
        self.data_folder: Optional[str] = None
        self.b_mean: Optional[float] = None
        self.total_samples: int = 0
        self.split_strategy: str = "loocv"
        self.last_krr_artifact_root: Optional[str] = None

    def set_split_strategy(self, key: str) -> None:
        """当前仅支持留一交叉验证（loocv）。"""
        _ = key
        self.split_strategy = "loocv"

    def set_loaded_training_folder(self, folder: str, records: Sequence[TrainingExperimentRecord]) -> None:
        """由 `dataset_io` 载入后写入；清空或覆盖旧数据。"""
        self.data_folder = folder
        self.records = list(records)
        self.total_samples = len(self.records)
        self.last_krr_artifact_root = None
        if self.records:
            self.b_mean = float(statistics.mean(r.B for r in self.records))
        else:
            self.b_mean = None

    def clear_training_data(self) -> None:
        self.records = []
        self.data_folder = None
        self.b_mean = None
        self.total_samples = 0
        self.last_krr_artifact_root = None

    def summary_text(self) -> str:
        strat_label = "留一交叉验证（LOOCV）" if self.split_strategy == "loocv" else self.split_strategy
        lines = [
            "训练模型：核岭回归（Kernel Ridge）",
            f"数据目录：{self.data_folder or '（未选择）'}",
        ]
        if self.last_krr_artifact_root:
            lines.append(f"最新核回归 artefact：{self.last_krr_artifact_root}")
        lines.extend(
            [
            f"数据集总样本数：{self.total_samples}",
            f"划分策略：{strat_label}",
            "（LOOCV：每折留出 1 条样本作验证，其余参与该折训练；共 n 折，n 为样本数）",
            ]
        )
        if self.total_samples > 0:
            if self.total_samples <= 5:
                lines.append("提示：样本数 ≤5 时，难以获得较好的训练效果，建议扩充数据。")
        if self.total_samples > 0 and self.b_mean is not None:
            lines.append(f"拟合参数 B 的样本均值（辅助信息）：{self.b_mean:g}")
            lines.append("纵轴约定：最大直径 ← K；初始状态常数 ← B（拖曳拟合）；时间常数 ← C。")
            lines.append("样本明细（截取前 20 行）：")
            for i, r in enumerate(self.records[:20], start=1):
                base = os.path.basename(r.source_path)
                lines.append(
                    f"  {i}. {base} | 当量 {r.equivalent_kg_tnt:g} kg, 含铝 {r.al_percent:g}% | "
                    f"K={r.K:g}, B={r.B:g}, C={r.C:g}"
                )
            if len(self.records) > 20:
                lines.append(f"  … 共 {len(self.records)} 组，其余省略。")
        return "\n".join(lines)
