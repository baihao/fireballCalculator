#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「帮助」菜单：用户手册等。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox


def resolve_user_manual_pdf() -> Path:
    """
    用户手册 PDF 路径。

    - 打包后：与可执行文件同目录下的 ``manual/manual.pdf``
    - 开发时：``source/manual/manual.pdf``（相对本模块 ``desktop/menu``）
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        # …/source/desktop/menu/help_menu.py → …/source
        base = Path(__file__).resolve().parent.parent.parent
    return base / "manual" / "manual.pdf"


def open_user_manual(parent: QMainWindow | None = None) -> None:
    """用系统默认 PDF 阅读器打开用户手册。"""
    path = resolve_user_manual_pdf()
    if not path.is_file():
        QMessageBox.warning(
            parent,
            "用户手册",
            f"未找到用户手册文件：\n{path}",
        )
        return
    ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
    if not ok:
        QMessageBox.warning(
            parent,
            "用户手册",
            f"无法打开用户手册：\n{path}\n\n请确认已安装 PDF 阅读器。",
        )


def setup_help_menu(main_window: QMainWindow, help_menu: QMenu) -> None:
    """在已创建的「帮助」菜单上添加条目。"""
    act_manual = QAction("用户手册", main_window)
    act_manual.triggered.connect(lambda: open_user_manual(main_window))
    help_menu.addAction(act_manual)
