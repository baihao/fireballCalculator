#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用菜单栏装配：创建「文件」「视图」并委托给独立模块。

- 「文件」：`file_menu.setup_file_menu`
- 「视图」：`view_menu.setup_view_menu`
"""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow

from .file_menu import setup_file_menu
from .view_menu import setup_view_menu


def setup_application_menu(main_window: QMainWindow) -> None:
    """
    在主窗口上创建菜单栏。

    要求 main_window 已具备：tab_widget、extract_tab、training_tab、model_tab、sidebar。
    「退出」仅出现在「文件」菜单中（NoRole，不并入 macOS 应用菜单）。
    """
    bar = main_window.menuBar()

    file_menu = bar.addMenu("文件")
    setup_file_menu(main_window, file_menu)

    view_menu = bar.addMenu("视图")
    setup_view_menu(main_window, view_menu)
