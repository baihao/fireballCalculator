#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「文件」菜单：按当前视图显示不同导入项、退出。

逻辑与「视图」菜单分离，便于维护。
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMenu

TAB_INDEX_MACHINE_VISION = 0
TAB_INDEX_MACHINE_LEARNING = 1
TAB_INDEX_ENGINEERING = 2


def setup_file_menu(main_window: QMainWindow, file_menu: QMenu) -> None:
    """
    在已创建的「文件」菜单上添加条目。

    - 当前为「机器视觉」标签时：显示三个导入项（与侧边栏「数据源」按钮一致）。
    - 当前为「机器学习」标签时：显示「输入数据」（与侧栏「输入数据」一致）。
    - 当前为「工程计算」标签时：显示「选择模型目录」（与侧栏「选择模型」一致）。
    """
    extract_tab = main_window.extract_tab
    training_tab = main_window.training_tab
    model_tab = main_window.model_tab
    tab_widget = main_window.tab_widget

    act_import_images = QAction("导入火球图像序列", main_window)
    act_import_images.triggered.connect(extract_tab.select_image_sequence_folder)
    file_menu.addAction(act_import_images)

    act_import_sequence = QAction("导入爆炸序列文件", main_window)
    act_import_sequence.triggered.connect(extract_tab.select_sequence_folder)
    file_menu.addAction(act_import_sequence)

    act_import_temp = QAction("导入火球温度序列", main_window)
    act_import_temp.triggered.connect(extract_tab.select_temperature_sequence)
    file_menu.addAction(act_import_temp)

    act_import_training_data = QAction("输入数据", main_window)
    act_import_training_data.triggered.connect(training_tab.input_training_data)
    file_menu.addAction(act_import_training_data)

    act_import_train = QAction("选择模型目录", main_window)
    act_import_train.triggered.connect(model_tab.select_model_folder)
    file_menu.addAction(act_import_train)

    def update_import_visibility(index: int) -> None:
        mv = index == TAB_INDEX_MACHINE_VISION
        act_import_images.setVisible(mv)
        act_import_sequence.setVisible(mv)
        act_import_temp.setVisible(mv)
        act_import_training_data.setVisible(index == TAB_INDEX_MACHINE_LEARNING)
        act_import_train.setVisible(index == TAB_INDEX_ENGINEERING)

    tab_widget.currentChanged.connect(update_import_visibility)
    update_import_visibility(tab_widget.currentIndex())

    file_menu.addSeparator()

    quit_action = QAction("退出", main_window)
    quit_action.setShortcut(QKeySequence.Quit)
    quit_action.setMenuRole(QAction.MenuRole.NoRole)
    quit_action.triggered.connect(main_window.close)
    file_menu.addAction(quit_action)
