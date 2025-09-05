#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆炸火球分析系统 - 桌面应用启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from main import main
    main()
