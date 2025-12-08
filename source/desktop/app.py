#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆炸火球分析系统 - 桌面应用主程序
"""

import sys
import os
from log import setup_logging
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from framework import FireballAnalysisApp


def main():
    setup_logging()
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("爆炸火球分析系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Fireball Analysis")
    
    # 设置应用程序图标
    icon_path = os.path.join(os.path.dirname(__file__), 'icon', 'fireball_app_icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # 创建主窗口
    window = FireballAnalysisApp()
    
    # 设置主窗口图标
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
