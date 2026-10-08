#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工程计算大标签：内嵌「工程计算」「参数仿真」两个子标签。"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ui_fonts import song_family_qss

from .engineering_tab import EngineeringTab
from parameter_sim_tab import ParameterSimTab

SUB_INDEX_ENGINEERING = 0
SUB_INDEX_PARAMETER_SIM = 1


class EngineeringHubTab(QWidget):
    """顶层「工程计算」页：子标签切换时通知主窗口刷新侧栏。"""

    sub_tab_changed = Signal(int)

    def __init__(
        self,
        engineering_tab: Optional[EngineeringTab] = None,
        parameter_sim_tab: Optional[ParameterSimTab] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engineering_tab = engineering_tab or EngineeringTab()
        self.parameter_sim_tab = parameter_sim_tab or ParameterSimTab()

        self._inner = QTabWidget()
        _fq = song_family_qss()
        self._inner.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid #1f2937;
                background-color: #111827;
                font-family: {_fq};
            }}
            QTabBar::tab {{
                background-color: #0b1220;
                color: #e5e7eb;
                padding: 6px 12px;
                margin-right: 2px;
                border: 1px solid #1f2937;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-family: {_fq};
            }}
            QTabBar::tab:selected {{
                background-color: #111827;
                border-color: #38bdf8;
            }}
            QTabBar::tab:hover {{
                background-color: #1f2937;
            }}
            """
        )
        self._inner.addTab(self.engineering_tab, "工程计算")
        self._inner.addTab(self.parameter_sim_tab, "参数仿真")
        self._inner.currentChanged.connect(self._on_sub_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._inner)

    def _on_sub_tab_changed(self, index: int) -> None:
        self.sub_tab_changed.emit(index)

    def current_sub_index(self) -> int:
        return self._inner.currentIndex()

    def set_sub_tab(self, index: int) -> None:
        if 0 <= index < self._inner.count():
            self._inner.setCurrentIndex(index)

    def get_sidebar_widget(self) -> QWidget:
        page = self._inner.currentWidget()
        getter = getattr(page, "get_sidebar_widget", None)
        if callable(getter):
            return getter()
        return QWidget()

    def refresh_layout_after_show(self) -> None:
        page = self._inner.currentWidget()
        refresh = getattr(page, "refresh_layout_after_show", None)
        if callable(refresh):
            refresh()
