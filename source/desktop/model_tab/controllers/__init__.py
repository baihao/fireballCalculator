#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_tab.controllers package

用于暴露模型标签页相关的控制器。
"""

from .chart_controller import ModelTabChartController
from .model_controller import ModelController

__all__ = ["ModelTabChartController", "ModelController"]
