#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SequenceModel

负责缓存火球序列相关的所有业务数据，并提供统一的读写接口，供
ExtractTab / PromptController 等上层组件使用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sequence_manager import SequenceManager


class SequenceModel:
    """管理火球序列的内存状态与数据操作。"""

    def __init__(self, sequence_manager: SequenceManager) -> None:
        self._manager = sequence_manager
        self.reset()

    # ------------------------------------------------------------------ #
    # 基础状态
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """清空当前序列的所有缓存状态。"""
        self._current_path: Optional[str] = None
        self._sequence_data: Dict[str, Any] = {}
        self._image_paths: List[str] = []
        self._parameters: Dict[str, Any] = {}
        self._pixel_length: float = 1.0
        self._explosion_duration_ms: float = 140.0

        self._prompt_data: Dict[int, Dict[str, Any]] = {}
        self._ignition_point: Optional[Tuple[int, int]] = None
        self._annotated_indices: Set[int] = set()
        self._group_count: int = 1

        self._segmentation_results: List[Dict[str, Any]] = []
        self._has_segmentation_data: bool = False

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    @property
    def sequence_data(self) -> Dict[str, Any]:
        return self._sequence_data

    @property
    def image_paths(self) -> List[str]:
        return list(self._image_paths)

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(self._parameters)

    @property
    def pixel_length(self) -> float:
        return float(self._pixel_length)

    @property
    def explosion_duration_ms(self) -> float:
        return float(self._explosion_duration_ms)

    @property
    def group_count(self) -> int:
        return self._group_count

    # ------------------------------------------------------------------ #
    # 加载 & 应用
    # ------------------------------------------------------------------ #
    def load_from_file(self, file_path: str) -> Tuple[bool, str]:
        """从 JSON 文件加载序列数据并刷新内部状态。"""
        success, sequence_data, message = self._manager.load_sequence_file(file_path)
        if not success:
            return False, message

        self.apply_sequence_dict(sequence_data, file_path)
        return True, message

    def apply_sequence_dict(
        self,
        sequence_data: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> None:
        """应用已有的序列数据（无需重新读取文件）。"""
        self._sequence_data = sequence_data or {}
        if file_path:
            self._current_path = file_path

        self._image_paths = self._manager.get_image_paths_from_sequence(self._sequence_data)
        self._parameters = self._manager.get_parameters_from_sequence(self._sequence_data) or {}
        self._pixel_length = float(self._parameters.get("pixel_length", 1.0))
        self._explosion_duration_ms = float(self._parameters.get("explosion_duration", 140.0))
        self._update_group_count()

        self._load_prompt_artifacts_from_sequence()
        self._load_segmentation_from_sequence()

    # ------------------------------------------------------------------ #
    # Prompt / Ignition
    # ------------------------------------------------------------------ #
    def get_prompt_data(self) -> Dict[int, Dict[str, Any]]:
        return {idx: dict(data) for idx, data in self._prompt_data.items()}

    def get_prompt_points(self, image_index: int) -> Dict[str, Any]:
        data = self._prompt_data.get(image_index, {"points": [], "labels": []})
        return {"points": list(data.get("points", [])), "labels": list(data.get("labels", []))}

    def set_prompt_data(self, prompt_data: Dict[int, Dict[str, Any]]) -> None:
        self._prompt_data = {
            int(idx): {
                "points": [list(pt) for pt in value.get("points", [])],
                "labels": list(value.get("labels", [])),
            }
            for idx, value in (prompt_data or {}).items()
        }
        self._refresh_annotated_indices()
        self._apply_prompt_to_sequence_data()

    def add_prompt_point(self, image_index: int, point: Tuple[int, int], is_positive: bool) -> None:
        entry = self._prompt_data.setdefault(image_index, {"points": [], "labels": []})
        entry["points"].append([int(point[0]), int(point[1])])
        entry["labels"].append(1 if is_positive else 0)
        self._annotated_indices.add(image_index)
        self._apply_prompt_to_sequence_data()

    def remove_prompt_points(self, image_index: int) -> None:
        if image_index in self._prompt_data:
            del self._prompt_data[image_index]
        self._annotated_indices.discard(image_index)
        self._apply_prompt_to_sequence_data()

    def clear_prompt_data(self) -> None:
        self._prompt_data.clear()
        self._annotated_indices.clear()
        self._apply_prompt_to_sequence_data()

    def get_annotated_indices(self) -> Set[int]:
        return set(self._annotated_indices)

    def set_ignition_point(self, point: Optional[Tuple[int, int]]) -> None:
        self._ignition_point = tuple(point) if point is not None else None
        self._apply_prompt_to_sequence_data()

    def get_ignition_point(self) -> Optional[Tuple[int, int]]:
        return self._ignition_point

    def save_prompt_artifacts(self) -> Tuple[bool, str]:
        if not self._current_path:
            return False, "未指定序列文件路径，无法保存参考点数据"
        success, message = self._manager.save_prompt_and_ignition_data_to_sequence(
            self._current_path,
            self.get_prompt_data(),
            self._ignition_point,
        )
        if success:
            # 同步更新 sequence_data 内的内容，避免重复读取
            self._apply_prompt_to_sequence_data()
        return success, message

    def clear_prompt_artifacts(self) -> Tuple[bool, str]:
        if not self._current_path:
            return False, "未指定序列文件路径，无法清除参考点数据"
        prompt_ok, prompt_msg = self._manager.clear_prompt_data_from_sequence(self._current_path)
        ignition_ok, ignition_msg = self._manager.clear_ignition_point_from_sequence(self._current_path)

        if prompt_ok:
            self.clear_prompt_data()
        if ignition_ok:
            self._ignition_point = None
        if prompt_ok or ignition_ok:
            self._apply_prompt_to_sequence_data()

        messages = [msg for ok, msg in ((prompt_ok, prompt_msg), (ignition_ok, ignition_msg)) if msg]
        status_msg = "；".join(messages) if messages else ""
        return prompt_ok and ignition_ok, status_msg

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #
    def update_segmentation_results(self, results: Sequence[Dict[str, Any]]) -> None:
        self._segmentation_results = list(results or [])
        self._has_segmentation_data = len(self._segmentation_results) > 0

    def get_segmentation_results(self) -> List[Dict[str, Any]]:
        return list(self._segmentation_results)

    def has_segmentation_data(self) -> bool:
        return self._has_segmentation_data

    def clear_segmentation_results(self) -> Tuple[bool, str]:
        if not self._current_path:
            return False, "未指定序列文件路径，无法清除分割结果"
        success, message = self._manager.clear_segmentation_results_from_sequence(self._current_path)
        if success:
            self._segmentation_results = []
            self._has_segmentation_data = False
            if isinstance(self._sequence_data, dict) and 'image_sequence_segmentation' in self._sequence_data:
                del self._sequence_data['image_sequence_segmentation']
        return success, message

    def get_segmentation_summary(self) -> Dict[str, Any]:
        total = len(self._segmentation_results)
        success_count = sum(1 for item in self._segmentation_results if item.get("success"))
        return {
            "total": total,
            "success": success_count,
            "has_data": self._has_segmentation_data,
        }

    # ------------------------------------------------------------------ #
    # 温度 / 导出等辅助
    # ------------------------------------------------------------------ #
    def get_temperature_series(self) -> Tuple[List[float], List[float]]:
        return self._manager.get_temperature_data_from_sequence(self._sequence_data)

    def build_export_payload(
        self,
        diameter_series: Sequence[Tuple[float, float]],
        drag_fit_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """构建保存分析结果所需的字典。"""
        return {
            "diameter_over_time": [
                {"time_ms": float(t), "diameter_m": float(d)}
                for t, d in (diameter_series or [])
            ],
            "parameters": {
                "material_type": self._parameters.get("material_type"),
                "equivalent": self._parameters.get("equivalent"),
                "al_percent": self._parameters.get("al_percent"),
                "explosion_duration": self._parameters.get("explosion_duration"),
                "pixel_length": self._parameters.get("pixel_length"),
            },
            "drag_fit": drag_fit_result or {
                "success": False,
                "K": None,
                "B": None,
                "C": None,
                "expression": "D(t) = K * (1 - B * exp(-C * t^2))",
            },
        }

    def get_sequence_summary(self) -> Dict[str, Any]:
        """返回用于 UI 展示的序列摘要信息。"""
        prompt_points_total = sum(len(data.get("points", [])) for data in self._prompt_data.values())
        temperature_time, temperature_data = self.get_temperature_series()
        summary = {
            "image_count": len(self._image_paths),
            "has_temperature_data": len(temperature_data) > 0,
            "temperature_points": len(temperature_data),
            "has_prompt_data": len(self._prompt_data) > 0,
            "total_prompt_points": prompt_points_total,
            "has_ignition_point": self._ignition_point is not None,
            "ignition_point": self._ignition_point,
            "explosion_duration": self._explosion_duration_ms,
        }
        return summary

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #
    def _update_group_count(self) -> None:
        total = len(self._image_paths)
        if total <= 0:
            self._group_count = 1
            return
        max_groups_by_size = (total + 249) // 250
        group_count = max(2, max_groups_by_size)
        self._group_count = min(group_count, total)

    def _load_prompt_artifacts_from_sequence(self) -> None:
        # 先读取起爆点，避免后续写入 prompt 时误删
        ignition_point = self._manager.get_ignition_point_from_sequence(self._sequence_data)
        try:
            print(f"[SequenceModel] ignition_point(from manager) = {ignition_point}")
        except Exception:
            pass
        self._ignition_point = ignition_point

        # 再读取 prompt 数据（内部可能会同步到内存 image_sequence，但不应覆盖/删除已有起爆点）
        prompt_data = self._manager.get_prompt_data_from_sequence(self._sequence_data)
        self.set_prompt_data(prompt_data or {})

    def _apply_prompt_to_sequence_data(self) -> None:
        """在内存中同步 prompt / ignition 数据，避免重复读取文件。"""
        if not isinstance(self._sequence_data, dict):
            return
        image_seq = self._sequence_data.setdefault("image_sequence", {})
        if self._prompt_data:
            image_seq["prompt_data"] = {str(idx): data for idx, data in self._prompt_data.items()}
        elif "prompt_data" in image_seq:
            del image_seq["prompt_data"]
        if self._ignition_point is not None:
            image_seq["target_center"] = list(self._ignition_point)
        elif "target_center" in image_seq:
            del image_seq["target_center"]

    def _load_segmentation_from_sequence(self) -> None:
        results = self._manager.get_segmentation_results_from_sequence(self._sequence_data)
        if results:
            self._segmentation_results = list(results)
            self._has_segmentation_data = True
        else:
            self._segmentation_results = []
            self._has_segmentation_data = False

    def _refresh_annotated_indices(self) -> None:
        self._annotated_indices = {
            idx for idx, data in self._prompt_data.items()
            if isinstance(data, dict) and len(data.get("points", [])) > 0
        }


