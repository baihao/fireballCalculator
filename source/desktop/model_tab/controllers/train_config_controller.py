#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练参数配置控制器
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QLineEdit,
    QDialogButtonBox,
)


@dataclass
class TrainConfig:
    algorithm: str = "T-Transformer"
    learning_rate: str = "0.0005"
    epochs: str = "50"


class TrainConfigController:
    """负责管理训练参数配置对话框"""

    def __init__(self, parent=None):
        self.parent = parent
        self._config = TrainConfig()

    def open_dialog(self) -> Optional[Dict[str, str]]:
        """打开训练参数配置对话框"""
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("训练参数配置")

        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        algo_combo = QComboBox()
        algo_combo.addItems(["T-Transformer"])
        algo_combo.setCurrentText(self._config.algorithm)
        form_layout.addRow("算法", algo_combo)

        lr_edit = QLineEdit(self._config.learning_rate)
        form_layout.addRow("学习率", lr_edit)

        epoch_edit = QLineEdit(self._config.epochs)
        form_layout.addRow("轮次", epoch_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            self._config.algorithm = algo_combo.currentText()
            self._config.learning_rate = lr_edit.text()
            self._config.epochs = epoch_edit.text()
            return self.get_config()

        return None

    def get_config(self) -> Dict[str, str]:
        """返回当前训练配置"""
        return asdict(self._config)

