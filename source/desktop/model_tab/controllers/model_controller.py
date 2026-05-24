#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程计算 — 模型导入状态、概要文本、仿真前 K/B/C 解析（核岭回归预测或计算器缩放）。
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PySide6.QtWidgets import QMessageBox

from fireball_radius_calculator import FireballCalculator

# 与 kernel_regression.train_kbc_kernel_ridge.MODEL_ARTIFACT_FILENAMES 一致
_KRR_JOBLIB_NAMES = ("kbc_krr_K.joblib", "kbc_krr_B.joblib", "kbc_krr_C.joblib")


def _ensure_kernel_regression_path() -> None:
    desktop = Path(__file__).resolve().parent.parent.parent
    source = desktop.parent
    for p in (desktop, source):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class ModelFolderLoadResult:
    summary_text: str
    can_run_simulation: bool
    applied_json_count: int


class ModelController:
    """管理导入目录、JSON/KRR 元数据，以及在仿真开始时解析 K、B、C。"""

    def __init__(self, calculator: FireballCalculator) -> None:
        self._calc = calculator
        self.model_folder_path: Optional[str] = None
        self.training_files: List[str] = []
        self.training_temperature_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.training_K_value: Optional[float] = None
        self.training_equivalent: Optional[float] = None
        self.krr_artifact_root: Optional[Path] = None
        self.last_sim_kbc: Optional[Tuple[float, float, float]] = None
        self.last_kbc_source: str = ""
        self.last_first_params: Optional[dict] = None

    def get_material_by_al_content(self, al_content: float) -> str:
        if al_content <= 30:
            return "30%Al/Rubber"
        if al_content <= 40:
            return "40%Al/Rubber"
        if al_content <= 50:
            return "50%Al/Rubber"
        if al_content <= 60:
            return "60%Al/Rubber"
        return "40%Al/Rubber"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            cleaned = cleaned.replace("%", "")
            cleaned = re.sub(r"[^\d\.\-eE+]", "", cleaned)
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _krr_manifest_path(self, root: Path) -> Optional[Path]:
        mf = root / "manifest.json"
        return mf if mf.is_file() else None

    def _krr_artifact_complete(self, root: Path) -> bool:
        """当前目录是否包含完整核岭回归 artefact（manifest + 三套 joblib）。"""
        if self._krr_manifest_path(root) is None:
            return False
        return all((root / n).is_file() for n in _KRR_JOBLIB_NAMES)

    def _detect_krr_artifact(self, root: Path) -> None:
        """仅在 manifest 与模型文件齐全时视为可用 KRR 目录（供 predict 使用）。"""
        self.krr_artifact_root = root if self._krr_artifact_complete(root) else None

    def _summarize_krr_manifests(self, root: Path) -> List[str]:
        """概要：仅针对当前目录下的 manifest（不再扫描子目录或挑选多版本）。"""
        mf = self._krr_manifest_path(root)
        if mf is None:
            return []
        try:
            with open(mf, encoding="utf-8") as fp:
                m = json.load(fp)
        except Exception as e:
            return [f"  manifest.json 读取失败（{e}）"]

        out: List[str] = []
        out.append("    模型类型：核岭回归 RBF / KernelRidge（K、B、C 三目标）")
        out.append(f"    训练样本数：{m.get('n_samples', '—')}")
        out.append(f"    正则 α（KernelRidge）：{m.get('alpha', '—')}")
        out.append(f"    核：{m.get('kernel', '—')}")
        tg = m.get("targets") or {}
        for name in ("K", "B", "C"):
            info = tg.get(name) or {}
            if not info:
                continue
            sig = info.get("best_sigma", "—")
            tr = info.get("best_loocv_train_mse", "—")
            te = info.get("best_loocv_test_mse", "—")
            out.append(f"    目标 {name}：σ={sig}，LOOCV train_MSE={tr}，test_MSE={te}")
        return out

    def load_folder(self, folder: str, parent: Optional[Any] = None) -> ModelFolderLoadResult:
        root = Path(folder).expanduser().resolve()
        self.model_folder_path = str(root)
        self.training_temperature_data = None
        self.training_K_value = None
        self.training_equivalent = None
        self.training_files = []
        self.last_sim_kbc = None
        self.last_kbc_source = ""
        self._detect_krr_artifact(root)

        lines: List[str] = []
        lines.append(f"模型目录：{self.model_folder_path}")
        lines.append("")

        krr_lines = self._summarize_krr_manifests(root)
        if krr_lines:
            lines.append("【核岭回归 artefact】")
            lines.append("  （当前目录即为一次训练输出目录）")
            lines.extend(krr_lines)
            lines.append("")

        json_files = sorted(root.glob("*.json"))
        self.training_files = [str(p) for p in json_files]
        applied = 0
        first_params = None
        self.last_first_params = None
        if self.training_files:
            lines.append("【火球实验 JSON】")
            for p in self.training_files:
                lines.append(f"  · {os.path.basename(p)}")
            lines.append("")
            applied, first_params = self._apply_training_parameters_from_files(self.training_files)
            lines.append(f"已解析并用于计算器初值：{applied} / {len(self.training_files)} 个文件")
            if first_params:
                eq = first_params.get("equivalent")
                al = first_params.get("al_percent")
                dur = first_params.get("duration")
                lines.append(
                    f"（用于填充仿真默认值）当量 {eq} kg TNT；含铝 {al} %；爆炸时长 {dur}"
                )
            if self.training_equivalent is not None:
                lines.append(f"标准当量（缩放基准）：{self.training_equivalent:g} kg TNT")
            if self.training_K_value is not None:
                lines.append(f"拖曳直径 K（来自 JSON）：{self.training_K_value:g} m")
            if self.training_temperature_data is not None:
                tms, _tk = self.training_temperature_data
                lines.append(f"温度序列点数：{len(tms)}")
        else:
            lines.append("【火球实验 JSON】")
            if self.krr_artifact_root is not None:
                lines.append(
                    "  （目录下无 *.json；仿真将依据核岭回归模型及侧栏当量、含铝量等参数进行预测。）"
                )
            else:
                lines.append(
                    "  （当前目录下无 *.json；需完整核岭回归 artefact 或可解析 JSON 后方可仿真。）"
                )
            lines.append("")

        krr_ok = self.krr_artifact_root is not None
        can_run = applied > 0 or krr_ok
        if applied == 0 and self.training_files and not krr_ok:
            lines.append("")
            lines.append("提示：未能从 JSON 解析出当量/含铝等参数，请检查文件内容。")
        if not self.training_files and not krr_lines:
            lines.append("")
            lines.append("提示：当前目录无 manifest.json 且无火球实验 JSON。")

        self.last_first_params = first_params

        if parent is not None and applied == 0 and not krr_ok:
            QMessageBox.warning(
                parent,
                "导入失败",
                "当前目录缺少完整的核岭回归模型文件（需要 manifest.json 以及\n"
                "kbc_krr_K.joblib、kbc_krr_B.joblib、kbc_krr_C.joblib），\n"
                "且未发现可解析的火球实验 JSON。\n\n"
                "请确认已选择正确的训练输出目录，或放置有效的 *.json。",
            )

        return ModelFolderLoadResult(
            summary_text="\n".join(lines),
            can_run_simulation=can_run,
            applied_json_count=applied,
        )

    def resolve_kbc_for_simulation(
        self,
        equivalent_kg_tnt: float,
        al_percent: float,
        material_name: str,
    ) -> Tuple[str, Optional[Tuple[float, float, float]], bool]:
        """
        仿真开始时解析本轮使用的 K、B、C。

        Returns:
            source_tag: ``krr`` | ``calculator``
            kbc: 若核岭回归成功则为 (K,B,C)，否则为 None（走计算器缩放直径）
            use_explicit_kbc: 直径是否用显式拖曳式（仅 krr 路径为 True）
        """
        self.last_sim_kbc = None
        self.last_kbc_source = ""

        if self.krr_artifact_root is not None:
            _ensure_kernel_regression_path()
            try:
                from kernel_regression.train_kbc_kernel_ridge import predict_kernel_regression_kbc
            except ImportError as e:
                print(f"⚠️ 无法导入核岭回归预测：{e}")
            else:
                root = self.krr_artifact_root.resolve()
                try:
                    k, b, c = predict_kernel_regression_kbc(root, float(equivalent_kg_tnt), float(al_percent))
                except Exception as e:
                    print(f"⚠️ KRR 预测失败，回退计算器：{e}")
                else:
                    self.last_sim_kbc = (float(k), float(b), float(c))
                    self.last_kbc_source = "krr"
                    print(
                        f"✓ 核岭回归 K,B,C @ 当量={equivalent_kg_tnt:g} kg, 含铝={al_percent:g} % → "
                        f"K={k:g}, B={b:g}, C={c:g}"
                    )
                    return "krr", self.last_sim_kbc, True

        p = self._calc.get_standard_parameters(material_name)
        std_eq = float(self.training_equivalent) if self.training_equivalent is not None else float(
            p["standard_equivalent"]
        )
        m = float(equivalent_kg_tnt) / std_eq if std_eq > 0 else 1.0
        K_rad_std = float(p["K"])
        B_std = float(p["B"])
        C_std = float(p["C"])

        K_rad_eff = math.sqrt(m) * K_rad_std
        K_d_eff = 2.0 * K_rad_eff
        B_eff = B_std
        C_eff = C_std / m if m > 0 else C_std
        self.last_sim_kbc = (K_d_eff, B_eff, C_eff)
        self.last_kbc_source = "calculator"
        print(
            f"✓ 计算器缩放 K,B,C（M={m:.4g}）→ K_d={K_d_eff:g} m, B={B_eff:g}, C={C_eff:g}"
        )
        return "calculator", self.last_sim_kbc, False

    def apply_first_params_to_widgets(self, params: Optional[dict], tab: Any) -> None:
        """将首选 JSON 中的参数写入仿真侧栏 LineEdit（由 ModelTab 传入自身）。"""
        if not params:
            return
        if hasattr(tab, "p_eq") and params.get("equivalent") is not None:
            tab.p_eq.setText(f"{params['equivalent']:.6g}")
        if hasattr(tab, "p_al") and params.get("al_percent") is not None:
            tab.p_al.setText(f"{params['al_percent']:.6g}")
        if hasattr(tab, "p_duration") and params.get("duration") is not None:
            tab.p_duration.setText(f"{params['duration']:.6g}")

    def _apply_training_parameters_from_files(self, files: List[str]) -> Tuple[int, Optional[dict]]:
        if not files:
            return 0, None
        applied = 0
        first_params = None
        for path in files:
            params = self._apply_training_parameters_from_file(path)
            if params:
                applied += 1
                if first_params is None:
                    first_params = params
        if applied:
            print(f"🔧 已从 {applied} 个训练文件更新标准当量与 K/B/C 参数")
        return applied, first_params

    def _apply_training_parameters_from_file(self, file_path: str) -> Optional[dict]:
        if not file_path.lower().endswith(".json"):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"⚠️ 无法读取训练文件 {file_path}: {e}")
            return None

        params = data.get("parameters") or {}
        drag_fit = data.get("drag_fit") or {}

        equivalent = self._safe_float(params.get("equivalent"))
        al_percent = self._safe_float(params.get("al_percent"))
        duration = self._safe_float(params.get("explosion_duration"))
        k_value = self._safe_float(drag_fit.get("K"))
        b_value = self._safe_float(drag_fit.get("B"))
        c_value = self._safe_float(drag_fit.get("C"))

        if equivalent is None or al_percent is None:
            return None

        temperature_data = data.get("temperature", [])
        if temperature_data and len(temperature_data) > 0:
            try:
                time_data = []
                temp_data = []
                for time_temp_pair in temperature_data:
                    if len(time_temp_pair) >= 2:
                        time_data.append(float(time_temp_pair[0]))
                        temp_data.append(float(time_temp_pair[1]))
                if len(time_data) > 0 and len(temp_data) > 0:
                    self.training_temperature_data = (
                        np.array(time_data),
                        np.array(temp_data),
                    )
            except Exception as e:
                print(f"⚠️ 提取训练文件温度数据失败: {e}")
                self.training_temperature_data = None
        else:
            self.training_temperature_data = None

        material_name = self.get_material_by_al_content(al_percent)
        kwargs: dict = {"standard_equivalent": equivalent}
        self.training_equivalent = equivalent
        if k_value is not None:
            k_radius = k_value / 2.0
            kwargs["K"] = k_radius
            self.training_K_value = k_value
        else:
            self.training_K_value = None
        if b_value is not None:
            kwargs["B"] = b_value
        if c_value is not None:
            kwargs["C"] = c_value

        try:
            self._calc.set_standard_parameters(material_name, **kwargs)
            return {
                "equivalent": equivalent,
                "al_percent": al_percent,
                "duration": duration,
            }
        except Exception as exc:
            print(f"⚠️ 更新材料 {material_name} 参数失败: {exc}")
            return None
