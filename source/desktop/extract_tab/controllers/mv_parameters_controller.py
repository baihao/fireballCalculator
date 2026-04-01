#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆炸信息 / 炸药参数面板控制器

绑定 UI 与 SequenceModel，不向外部暴露 QWidget；ExtractTab 仅调用公开方法。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..sequence_model import SequenceModel
from ..utils.image_geometry import get_image_width_pixels


def _parse_positive_float(s: str, default: float) -> float:
    try:
        v = float(str(s).strip())
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _format_param_float(x: float) -> str:
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _derive_explosion_duration_ms(fps: float, n_images: int) -> float:
    if n_images <= 1 or fps <= 0:
        return 0.0
    return (n_images - 1) / fps * 1000.0


def _derive_pixel_length_m(fov_m: float, width_px: Optional[int]) -> float:
    if not width_px or width_px <= 0 or fov_m <= 0:
        return 0.01
    return fov_m / float(width_px)


class MvParametersController:
    """管理「图片参数」「炸药参数」四个控件与序列模型的同步。"""

    _DEFAULT_FPS = 1000.0
    _DEFAULT_FOV_M = 60.0

    def __init__(
        self,
        parent_tab,
        sequence_model: SequenceModel,
        ui_components: Dict[str, Any],
    ) -> None:
        self._parent = parent_tab
        self._model = sequence_model
        self._frame_rate_fps = ui_components.get("mv_frame_rate_fps")
        self._field_of_view_m = ui_components.get("mv_field_of_view_m")
        self._equivalent = ui_components.get("mv_equivalent")
        self._al_percent = ui_components.get("mv_al_percent")
        self._connect_signals()

    def _connect_signals(self) -> None:
        for w in (
            self._frame_rate_fps,
            self._field_of_view_m,
            self._equivalent,
            self._al_percent,
        ):
            if w is not None:
                w.textChanged.connect(self._on_text_changed)

    def _ui_fps_fov_strings(self) -> Tuple[str, str]:
        fps_s = self._frame_rate_fps.text().strip() if self._frame_rate_fps else ""
        fov_s = self._field_of_view_m.text().strip() if self._field_of_view_m else ""
        persist_fps = fps_s if fps_s else "1000"
        persist_fov = fov_s if fov_s else "60"
        return persist_fps, persist_fov

    def _infer_fps_from_legacy_params(self) -> str:
        p = self._model.parameters
        if p.get("frame_rate_fps"):
            return str(p["frame_rate_fps"]).strip()
        n = len(self._model.image_paths)
        try:
            t_ms = float(p.get("explosion_duration", 0))
            if n > 1 and t_ms > 0:
                return _format_param_float((n - 1) * 1000.0 / t_ms)
        except (TypeError, ValueError):
            pass
        return "1000"

    def _infer_fov_from_legacy_params(self) -> str:
        p = self._model.parameters
        if p.get("field_of_view_m"):
            return str(p["field_of_view_m"]).strip()
        paths = self._model.image_paths
        try:
            pl = float(p.get("pixel_length", 0.01))
            if paths:
                w = get_image_width_pixels(paths[0])
                if w and w > 0:
                    return _format_param_float(pl * float(w))
        except (TypeError, ValueError):
            pass
        return "60"

    def sync_model_from_ui(self) -> None:
        if not self._model.current_path:
            return
        fps_s, fov_s = self._ui_fps_fov_strings()
        fps = _parse_positive_float(fps_s, self._DEFAULT_FPS)
        fov = _parse_positive_float(fov_s, self._DEFAULT_FOV_M)
        n = len(self._model.image_paths)
        dur_ms = _derive_explosion_duration_ms(fps, n)
        first = self._model.image_paths[0] if n else ""
        wpx = get_image_width_pixels(first) if first else None
        pl_m = _derive_pixel_length_m(fov, wpx)
        self._model.apply_parameters_from_ui(
            self._equivalent.text().strip() if self._equivalent else "1",
            self._al_percent.text().strip() if self._al_percent else "30",
            fps_s,
            fov_s,
            _format_param_float(dur_ms),
            _format_param_float(pl_m),
        )

    def sync_ui_from_model(self) -> None:
        try:
            p = self._model.parameters
            if not p:
                return
            line_widgets = (
                self._frame_rate_fps,
                self._field_of_view_m,
                self._equivalent,
                self._al_percent,
            )
            for w in line_widgets:
                if w is not None:
                    w.blockSignals(True)
            if self._frame_rate_fps is not None:
                self._frame_rate_fps.setText(self._infer_fps_from_legacy_params())
            if self._field_of_view_m is not None:
                self._field_of_view_m.setText(self._infer_fov_from_legacy_params())
            if self._equivalent is not None:
                self._equivalent.setText(str(p.get("equivalent", "1")))
            if self._al_percent is not None:
                self._al_percent.setText(str(p.get("al_percent", "30")))
            for w in line_widgets:
                if w is not None:
                    w.blockSignals(False)
        except Exception as e:
            print(f"sync_ui_from_model: {e}")

    def parameter_values_for_sequence_creation(self, folder_path: str) -> Tuple[str, str, str, str, str, str, str]:
        """
        根据所选文件夹内图像张数、首张宽度与当前 UI 帧率/视场推导爆炸时长与 pixel_length。
        返回 (material_type, equivalent, al_percent, explosion_duration, pixel_length,
        frame_rate_fps, field_of_view_m)，与 create_work_sequence_from_image_folder 一致。
        """
        sm = self._parent.sequence_manager
        paths = sm.collect_image_paths_in_folder(folder_path)
        fps_s, fov_s = self._ui_fps_fov_strings()
        fps = _parse_positive_float(fps_s, self._DEFAULT_FPS)
        fov = _parse_positive_float(fov_s, self._DEFAULT_FOV_M)
        n = len(paths)
        dur_ms = _derive_explosion_duration_ms(fps, n)
        wpx = get_image_width_pixels(paths[0]) if paths else None
        pl_m = _derive_pixel_length_m(fov, wpx)
        return (
            SequenceModel.DEFAULT_MATERIAL_TYPE,
            self._equivalent.text().strip() if self._equivalent else "1",
            self._al_percent.text().strip() if self._al_percent else "30",
            _format_param_float(dur_ms),
            _format_param_float(pl_m),
            fps_s,
            fov_s,
        )

    def set_enabled(self, enabled: bool) -> None:
        for w in (
            self._frame_rate_fps,
            self._field_of_view_m,
            self._equivalent,
            self._al_percent,
        ):
            if w is not None:
                w.setEnabled(enabled)

    def _on_text_changed(self, *args: Any) -> None:
        if not self._model.current_path:
            return
        try:
            self.sync_model_from_ui()
        except Exception:
            pass
