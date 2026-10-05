#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工程计算结果 HTML 排版（公式集中罗列 + 代入 + 按式逐步代入）。"""

from __future__ import annotations

import html as html_lib
from typing import List

from . import symbols as sym
from .image_formula_engineering import (
    AL_REFERENCE_FRACTION,
    H_TNT_J_PER_KG,
    EngineeringEstimateInputs,
    EngineeringEstimateResult,
    _al_exp_shift,
)

_RESULT_STYLE = """
body {
  background-color: #0b1220;
  color: #e5e7eb;
  font-family: "Menlo", "Consolas", "Courier New", "SimSun", "Songti SC", monospace;
  font-size: 12px;
  line-height: 1.5;
  margin: 8px;
}
h2 { color: #38bdf8; font-size: 13px; font-weight: bold; margin: 14px 0 6px 0; }
.formula { color: #cbd5e1; margin: 2px 0 2px 12px; }
.subst { color: #94a3b8; margin: 4px 0 4px 12px; }
.calc-head { color: #7dd3fc; margin: 10px 0 2px 12px; font-weight: bold; }
.calc-step { color: #e5e7eb; margin: 2px 0 2px 24px; }
.calc-step b { color: #fbbf24; font-weight: normal; }
"""


def _g(v: float, digits: int = 6) -> str:
    return html_lib.escape(f"{v:.{digits}g}")


def _line(text: str) -> str:
    return f"<div>{text}</div>"


def _calc_block(heading: str, steps: List[str]) -> str:
    chunks = [_line(f"<div class='calc-head'>{heading}</div>")]
    for step in steps:
        chunks.append(_line(f"<div class='calc-step'>{step}</div>"))
    return "".join(chunks)


def format_engineering_result_html(
    inputs: EngineeringEstimateInputs,
    result: EngineeringEstimateResult,
) -> str:
    w = float(inputs.equivalent_kg)
    x = float(inputs.al_fraction)
    dx = _al_exp_shift(x)
    t_amb = float(inputs.t_amb_k)

    ref = AL_REFERENCE_FRACTION
    x_minus_ref = f"(x−{ref})"

    parts: List[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<style>{_RESULT_STYLE}</style></head><body>",
        _line("<h2>工程计算</h2>"),
        _line("<h2>一、公式</h2>"),
        _line(
            "<div class='formula'>"
            f"{sym.R_MAX} = 2.1405 · W<sup>0.371</sup> · exp[0.552·(x−{ref})] &nbsp; [m]"
            "</div>"
        ),
        _line(
            "<div class='formula'>"
            f"v = 196.72 · W<sup>−0.089</sup> · exp[−1.78·(x−{ref})] &nbsp; [m/s]"
            "</div>"
        ),
        _line(
            "<div class='formula'>"
            f"{sym.T_M} = 0.0057 · W<sup>0.426</sup> · exp[0.630·(x−{ref})] &nbsp; [s]"
            "</div>"
        ),
        _line(f"<div class='formula'>{sym.D_MAX} = 2 · {sym.R_MAX} &nbsp; [m]</div>"),
        _line(
            f"<div class='formula'>{sym.T_D} ≈ 0.30 · W<sup>1/3</sup> &nbsp; [s]</div>"
        ),
        _line(
            "<div class='formula'>"
            f"{sym.T_EQ} = [ {sym.T_AMB}<sup>4</sup> + "
            f"{sym.CHI_R}·W·{sym.H_TNT} / ({sym.EPSILON}·{sym.SIGMA}·{sym.KAPPA_A}·4π·"
            f"{sym.R_MAX}<sup>2</sup>·{sym.T_D}) ]<sup>1/4</sup> &nbsp; [K]"
            "</div>"
        ),
        _line("<h2>二、代入参数</h2>"),
        _line(
            f"<div class='subst'>W = {_g(w)} kg TNT，"
            f"x = {_g(x * 100, 4)} % → {_g(x, 6)}，"
            f"{sym.T_AMB} = {_g(t_amb, 6)} K（{_g(t_amb - 273.15, 4)} °C）</div>"
        ),
        _line(
            f"<div class='subst'>"
            f"{sym.CHI_R} = {_g(inputs.chi_r)}，{sym.EPSILON} = {_g(inputs.epsilon)}，"
            f"{sym.KAPPA_A} = {_g(inputs.kappa_a)}，{sym.SIGMA} = {_g(inputs.sigma)} "
            f"{sym.UNIT_SIGMA}，{sym.H_TNT} = {_g(H_TNT_J_PER_KG)} J/kg</div>"
        ),
        _line(
            f"<div class='subst'>(x − {ref}) = {_g(x, 6)} − {ref} = <b>{_g(dx, 6)}</b></div>"
        ),
        _line("<h2>三、按式计算</h2>"),
        _calc_block(
            "最大半径",
            [
                f"{sym.R_MAX} = 2.1405 · W<sup>0.371</sup> · exp[0.552·{x_minus_ref}]",
                (
                    f"= 2.1405 · ({_g(w)})<sup>0.371</sup> · "
                    f"exp[0.552·({_g(x, 6)}−{ref})]"
                ),
                f"= <b>{_g(result.r_max_m, 6)} m</b>",
            ],
        ),
        _calc_block(
            "最大膨胀速度",
            [
                f"v = 196.72 · W<sup>−0.089</sup> · exp[−1.78·{x_minus_ref}]",
                (
                    f"= 196.72 · ({_g(w)})<sup>−0.089</sup> · "
                    f"exp[−1.78·({_g(x, 6)}−{ref})]"
                ),
                f"= <b>{_g(result.v_max_m_s, 6)} m/s</b>",
            ],
        ),
        _calc_block(
            "达到最大尺寸时间",
            [
                f"{sym.T_M} = 0.0057 · W<sup>0.426</sup> · exp[0.630·{x_minus_ref}]",
                (
                    f"= 0.0057 · ({_g(w)})<sup>0.426</sup> · "
                    f"exp[0.630·({_g(x, 6)}−{ref})]"
                ),
                f"= <b>{_g(result.t_m_s, 6)} s</b>（{_g(result.t_m_s * 1000, 6)} ms）",
            ],
        ),
        _calc_block(
            "最大直径",
            [
                f"{sym.D_MAX} = 2 · {sym.R_MAX}",
                f"= 2 · {_g(result.r_max_m, 6)}",
                f"= <b>{_g(result.d_max_m, 6)} m</b>",
            ],
        ),
        _calc_block(
            "火球总持续时间",
            [
                f"{sym.T_D} ≈ 0.30 · W<sup>1/3</sup>",
                f"= 0.30 · ({_g(w)})<sup>1/3</sup>",
                f"= <b>{_g(result.t_d_s, 6)} s</b>（{_g(result.t_d_s * 1000, 6)} ms）",
            ],
        ),
        _calc_block(
            "火球温度",
            [
                (
                    f"{sym.T_EQ} = [ {sym.T_AMB}<sup>4</sup> + "
                    f"{sym.CHI_R}·W·{sym.H_TNT} / ({sym.EPSILON}·{sym.SIGMA}·{sym.KAPPA_A}·4π·"
                    f"{sym.R_MAX}<sup>2</sup>·{sym.T_D}) ]<sup>1/4</sup>"
                ),
                (
                    f"= [ ({_g(t_amb, 6)})<sup>4</sup> + "
                    f"{_g(inputs.chi_r)}·{_g(w)}·{_g(H_TNT_J_PER_KG)} / "
                    f"({_g(inputs.epsilon)}·{_g(inputs.sigma)}·{_g(inputs.kappa_a)}·4π·"
                    f"{_g(result.r_max_m, 6)}<sup>2</sup>·{_g(result.t_d_s, 6)}) ]<sup>1/4</sup>"
                ),
                f"= <b>{_g(result.t_eq_k, 6)} K</b>（{_g(result.t_eq_c, 6)} °C）",
            ],
        ),
        "</body></html>",
    ]
    return "".join(parts)
