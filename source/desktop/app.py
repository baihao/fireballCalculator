#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆炸火球分析系统 - 桌面应用主程序
"""

import sys
from PySide6.QtWidgets import QApplication
from framework import FireballAnalysisApp


def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("爆炸火球分析系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Fireball Analysis")
    
    # 创建主窗口
    window = FireballAnalysisApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
