#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工程计算结果 HTML 排版（公式集中罗列 + 代入 + 按式逐步代入）。"""

from __future__ import annotations

import html as html_lib
from typing import List

from . import symbols as sym
from .image_formula_engineering import (
    AL_REFERENCE_FRACTION,
    T_BASE_K,
    T_SCALE_AL_BASE,
    T_SCALE_AL_COEF,
    T_SCALE_COEF,
    T_SCALE_M_EXP,
    T_TRANSIENT_AL,
    T_TRANSIENT_BASE,
    U_COEF,
    EngineeringEstimateInputs,
    EngineeringEstimateResult,
    _al_exp_shift,
    _temperature_scale_c,
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
    a_coef, c_coef = _temperature_scale_c(w, x)

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
            f"u = {_g(U_COEF)} · t / M<sup>1/3</sup>"
            "</div>"
        ),
        _line(
            "<div class='formula'>"
            f"T(t; M, x) = {_g(T_BASE_K, 4)} + ({_g(T_TRANSIENT_BASE, 4)} + "
            f"{_g(T_TRANSIENT_AL, 4)}x)·e<sup>−2u</sup> + "
            f"{_g(T_SCALE_COEF)}·M<sup>{_g(T_SCALE_M_EXP, 4)}</sup>·"
            f"({_g(T_SCALE_AL_BASE, 4)} + {_g(T_SCALE_AL_COEF, 4)}x)·"
            f"(e<sup>−u</sup> − 14/13·e<sup>−2u</sup> + 1/13·e<sup>−15u</sup>) &nbsp; [K]"
            "</div>"
        ),
        _line(
            f"<div class='formula'>{sym.T_MAX} = max<sub>t≥0</sub> T(t; M, x) &nbsp; [K]</div>"
        ),
        _line("<h2>二、代入参数</h2>"),
        _line(
            f"<div class='subst'>M = W = {_g(w)} kg，"
            f"x = {_g(x * 100, 4)} % → {_g(x, 6)}</div>"
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
            "最大温度",
            [
                (
                    f"A = {_g(T_TRANSIENT_BASE, 4)} + {_g(T_TRANSIENT_AL, 4)}x = "
                    f"{_g(T_TRANSIENT_BASE, 4)} + {_g(T_TRANSIENT_AL, 4)}·{_g(x, 6)} = "
                    f"<b>{_g(a_coef, 6)}</b>"
                ),
                (
                    f"C = {_g(T_SCALE_COEF)}·M<sup>{_g(T_SCALE_M_EXP, 4)}</sup>·"
                    f"({_g(T_SCALE_AL_BASE, 4)} + {_g(T_SCALE_AL_COEF, 4)}x) = "
                    f"<b>{_g(c_coef, 6)}</b>"
                ),
                (
                    f"{sym.T_MAX} = max<sub>t</sub> [ {_g(T_BASE_K, 4)} + A·e<sup>−2u</sup> + "
                    f"C·(e<sup>−u</sup> − 14/13·e<sup>−2u</sup> + 1/13·e<sup>−15u</sup>) ]，"
                    f"u = {_g(U_COEF)}·t/M<sup>1/3</sup>"
                ),
                (
                    f"= <b>{_g(result.t_max_k, 6)} K</b>（{_g(result.t_max_c, 6)} °C），"
                    f"对应 t = {_g(result.t_at_tmax_s, 6)} s"
                    f"（{_g(result.t_at_tmax_s * 1000, 6)} ms）"
                ),
            ],
        ),
        "</body></html>",
    ]
    return "".join(parts)
