#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主窗口菜单栏（文件、视图等）。"""

from .app_menu_bar import setup_application_menu
from .file_menu import setup_file_menu
from .view_menu import setup_view_menu

__all__ = ["setup_application_menu", "setup_file_menu", "setup_view_menu"]
