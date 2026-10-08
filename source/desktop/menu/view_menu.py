#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「视图」菜单：主标签切换、侧边栏、全屏。

逻辑与「文件」菜单分离，便于维护。
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMenu

# 与 FireballAnalysisApp 中 tab_widget 顺序一致
TAB_INDEX_MACHINE_VISION = 0
TAB_INDEX_MACHINE_LEARNING = 1
TAB_INDEX_PARAMETER_PREDICTION = 2  # ModelTab「参数预测」
TAB_INDEX_ENGINEERING_HUB = 3  # EngineeringHubTab「工程计算」大标签

# 工程计算大标签内子页
SUB_INDEX_ENGINEERING = 0
SUB_INDEX_PARAMETER_SIM = 1


def setup_view_menu(main_window: QMainWindow, view_menu: QMenu) -> None:
    """在已创建的「视图」菜单上添加条目并绑定主窗口 tab_widget。"""
    tab_widget = main_window.tab_widget
    engineering_hub = main_window.engineering_hub

    group = QActionGroup(main_window)
    group.setExclusive(True)

    act_mv = QAction("机器视觉", main_window)
    act_mv.setCheckable(True)
    group.addAction(act_mv)
    view_menu.addAction(act_mv)

    act_ml = QAction("机器学习", main_window)
    act_ml.setCheckable(True)
    group.addAction(act_ml)
    view_menu.addAction(act_ml)

    act_pred = QAction("参数预测", main_window)
    act_pred.setCheckable(True)
    group.addAction(act_pred)
    view_menu.addAction(act_pred)

    act_eng_calc = QAction("工程计算", main_window)
    act_eng_calc.setCheckable(True)
    group.addAction(act_eng_calc)
    view_menu.addAction(act_eng_calc)

    act_param_sim = QAction("参数仿真", main_window)
    act_param_sim.setCheckable(True)
    group.addAction(act_param_sim)
    view_menu.addAction(act_param_sim)

    def on_mv_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_MACHINE_VISION)

    def on_ml_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_MACHINE_LEARNING)

    def on_pred_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_PARAMETER_PREDICTION)

    def on_eng_calc_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_ENGINEERING_HUB)
            engineering_hub.set_sub_tab(SUB_INDEX_ENGINEERING)

    def on_param_sim_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_ENGINEERING_HUB)
            engineering_hub.set_sub_tab(SUB_INDEX_PARAMETER_SIM)

    act_mv.toggled.connect(on_mv_toggled)
    act_ml.toggled.connect(on_ml_toggled)
    act_pred.toggled.connect(on_pred_toggled)
    act_eng_calc.toggled.connect(on_eng_calc_toggled)
    act_param_sim.toggled.connect(on_param_sim_toggled)

    def _sync_check_state() -> None:
        index = tab_widget.currentIndex()
        sub = engineering_hub.current_sub_index()
        act_mv.blockSignals(True)
        act_ml.blockSignals(True)
        act_pred.blockSignals(True)
        act_eng_calc.blockSignals(True)
        act_param_sim.blockSignals(True)
        act_mv.setChecked(index == TAB_INDEX_MACHINE_VISION)
        act_ml.setChecked(index == TAB_INDEX_MACHINE_LEARNING)
        act_pred.setChecked(index == TAB_INDEX_PARAMETER_PREDICTION)
        act_eng_calc.setChecked(
            index == TAB_INDEX_ENGINEERING_HUB and sub == SUB_INDEX_ENGINEERING
        )
        act_param_sim.setChecked(
            index == TAB_INDEX_ENGINEERING_HUB and sub == SUB_INDEX_PARAMETER_SIM
        )
        act_mv.blockSignals(False)
        act_ml.blockSignals(False)
        act_pred.blockSignals(False)
        act_eng_calc.blockSignals(False)
        act_param_sim.blockSignals(False)

    def sync_tabs_from_menu(index: int) -> None:
        _sync_check_state()

    tab_widget.currentChanged.connect(sync_tabs_from_menu)
    engineering_hub.sub_tab_changed.connect(lambda _i: _sync_check_state())
    _sync_check_state()

    view_menu.addSeparator()

    sidebar = getattr(main_window, "sidebar", None)
    if sidebar is not None:
        sidebar_action = QAction("显示侧边栏", main_window)
        sidebar_action.setCheckable(True)
        sidebar_action.setChecked(sidebar.isVisible())
        sidebar_action.toggled.connect(sidebar.setVisible)
        view_menu.addAction(sidebar_action)

    fullscreen_action = QAction("切换全屏", main_window)
    fullscreen_action.setShortcut(QKeySequence(QKeySequence.StandardKey.FullScreen))

    def _toggle_fullscreen() -> None:
        if main_window.isFullScreen():
            main_window.showNormal()
        else:
            main_window.showFullScreen()

    fullscreen_action.triggered.connect(_toggle_fullscreen)
    view_menu.addAction(fullscreen_action)
