#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分组检查条控件（CheckBar）

用途：在时间轴下方显示一个分组矩形条；若某一组包含已标注的数据点，则在该组区域内显示绿色对钩。

API：
- update(length: int, group_count: int, annotated_indices: list[int]) -> None
  - length: 数据总长度（例如帧数）
  - group_count: 需要分成的组数（>=1）
  - annotated_indices: 已标注数据的索引数组（0-based）

绘制规则：
- 整体为一条横条，按组等分；
- 每组边界绘制分割线；
- 若组内至少包含一个 annotated index，则在该组区域中心绘制绿色对钩；
- 自适应当前控件宽度与高度进行绘制。
"""

from typing import List, Set
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPolygon
from PySide6.QtCore import Qt, QRectF, QPoint


class CheckBar(QWidget):
    """分组检查条控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._length: int = 0
        self._group_count: int = 1
        self._annotated: Set[int] = set()
        self._focus_index: int | None = None

        # 外观参数
        self._bg_color = QColor('#111827')
        self._bar_color = QColor('#1f2937')
        self._border_color = QColor('#374151')
        self._split_color = QColor('#4b5563')
        self._check_color = QColor('#22c55e')  # 绿色对钩

        self.setMinimumHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # 对外接口：更新数据
    def update(self, length: int, group_count: int, annotated_indices: List[int]):  # type: ignore[override]
        self._length = max(0, int(length))
        self._group_count = max(1, int(group_count))
        try:
            self._annotated = {int(i) for i in annotated_indices}
        except Exception:
            self._annotated = set()
        self.repaint()

    def set_focus(self, index: int):
        """设置当前焦点图片索引，所属分组将被高亮。"""
        try:
            if index is None:
                self._focus_index = None
            else:
                self._focus_index = max(0, min(int(index), max(0, self._length - 1)))
        except Exception:
            self._focus_index = None
        self.repaint()

    # 计算每组的 [start, end) 索引范围
    def _group_ranges(self) -> List[tuple]:
        if self._group_count <= 1 or self._length <= 0:
            return [(0, self._length)]
        base = self._length // self._group_count
        rem = self._length % self._group_count
        ranges = []
        start = 0
        for g in range(self._group_count):
            size = base + (1 if g < rem else 0)
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges

    def paintEvent(self, event):  # noqa: N802
        width = self.width()
        height = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 背景
        painter.fillRect(self.rect(), self._bg_color)

        # 外框矩形
        outer_rect = QRectF(2, height * 0.25, width - 4, height * 0.5)
        painter.setBrush(QBrush(self._bar_color))
        painter.setPen(QPen(self._border_color, 1))
        painter.drawRoundedRect(outer_rect, 6, 6)

        # 组分割线
        ranges = self._group_ranges()
        if ranges:
            group_w = outer_rect.width() / max(1, len(ranges))
            painter.setPen(QPen(self._split_color, 1))
            for i in range(1, len(ranges)):
                x = outer_rect.left() + i * group_w
                painter.drawLine(int(x), int(outer_rect.top()), int(x), int(outer_rect.bottom()))

            # 高亮焦点所在分组（半透明覆盖）
            try:
                focus_group = None
                if self._focus_index is not None and self._length > 0:
                    for gi, (s, e) in enumerate(ranges):
                        if s <= self._focus_index < e:
                            focus_group = gi
                            break
                if focus_group is not None:
                    highlight = QColor('#60a5fa')  # 天蓝
                    highlight.setAlpha(64)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(highlight))
                    group_left = outer_rect.left() + focus_group * group_w
                    painter.drawRoundedRect(QRectF(group_left, outer_rect.top(), group_w, outer_rect.height()), 6, 6)
            except Exception:
                pass

        # 在每个组内判断是否含已标注数据，并绘制绿色对钩
        for idx, (start, end) in enumerate(ranges):
            if start >= end:
                continue
            has_annotation = any((i in self._annotated) for i in range(start, end))
            if not has_annotation:
                continue

            # 计算该组中心区域
            group_left = outer_rect.left() + idx * (outer_rect.width() / max(1, len(ranges)))
            group_right = outer_rect.left() + (idx + 1) * (outer_rect.width() / max(1, len(ranges)))
            cx = (group_left + group_right) / 2.0
            cy = outer_rect.center().y()

            # 绘制对钩（简单的两段折线）
            painter.setPen(QPen(self._check_color, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            size = max(6.0, outer_rect.height() * 0.35)
            p1 = QPoint(int(cx - size * 0.6), int(cy))
            p2 = QPoint(int(cx - size * 0.2), int(cy + size * 0.4))
            p3 = QPoint(int(cx + size * 0.6), int(cy - size * 0.6))
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)

        painter.end()


    # 查询每组是否已有标注
    def groups_marked(self) -> List[bool]:
        ranges = self._group_ranges()
        marked = []
        for start, end in ranges:
            if start >= end:
                marked.append(False)
                continue
            has_annotation = any((i in self._annotated) for i in range(start, end))
            marked.append(has_annotation)
        return marked

    # 是否所有分组都已覆盖至少一个标注
    def is_all_groups_marked(self) -> bool:
        if self._length <= 0:
            return False
        return all(self.groups_marked()) if self._group_count >= 1 else False

def create_checkbar(parent=None) -> CheckBar:
    """工厂函数，便于 UI 构建器统一创建。"""
    return CheckBar(parent)


