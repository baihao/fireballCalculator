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
TAB_INDEX_MACHINE_LEARNING = 1  # TrainingTab（原「模型训练」）
TAB_INDEX_ENGINEERING = 2  # ModelTab（原「机器学习」）


def setup_view_menu(main_window: QMainWindow, view_menu: QMenu) -> None:
    """在已创建的「视图」菜单上添加条目并绑定主窗口 tab_widget。"""
    tab_widget = main_window.tab_widget

    group = QActionGroup(main_window)
    group.setExclusive(True)

    act_mv = QAction("机器视觉", main_window)
    act_mv.setCheckable(True)
    act_mv.setChecked(tab_widget.currentIndex() == TAB_INDEX_MACHINE_VISION)
    group.addAction(act_mv)
    view_menu.addAction(act_mv)

    act_ml = QAction("机器学习", main_window)
    act_ml.setCheckable(True)
    act_ml.setChecked(tab_widget.currentIndex() == TAB_INDEX_MACHINE_LEARNING)
    group.addAction(act_ml)
    view_menu.addAction(act_ml)

    act_eng = QAction("参数预测", main_window)
    act_eng.setCheckable(True)
    act_eng.setChecked(tab_widget.currentIndex() == TAB_INDEX_ENGINEERING)
    group.addAction(act_eng)
    view_menu.addAction(act_eng)

    def on_mv_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_MACHINE_VISION)

    def on_ml_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_MACHINE_LEARNING)

    def on_eng_toggled(checked: bool) -> None:
        if checked:
            tab_widget.setCurrentIndex(TAB_INDEX_ENGINEERING)

    act_mv.toggled.connect(on_mv_toggled)
    act_ml.toggled.connect(on_ml_toggled)
    act_eng.toggled.connect(on_eng_toggled)

    def sync_tabs_from_menu(index: int) -> None:
        act_mv.blockSignals(True)
        act_ml.blockSignals(True)
        act_eng.blockSignals(True)
        act_mv.setChecked(index == TAB_INDEX_MACHINE_VISION)
        act_ml.setChecked(index == TAB_INDEX_MACHINE_LEARNING)
        act_eng.setChecked(index == TAB_INDEX_ENGINEERING)
        act_mv.blockSignals(False)
        act_ml.blockSignals(False)
        act_eng.blockSignals(False)

    tab_widget.currentChanged.connect(sync_tabs_from_menu)
    sync_tabs_from_menu(tab_widget.currentIndex())

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
