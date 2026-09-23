"""La ligne en train de s'écrire — le « maintenant » de la ligne de temps.

C'est la dernière ligne du document, et le seul endroit de l'app où le rouge
d'enregistrement apparaît : un point plein posé sur la ligne de temps, à
l'aplomb exact des ticks de minute, suivi de la forme d'onde qui danse tant que
Benji entend quelque chose. Pas de filet horizontal au-dessus — il couperait la
ligne de temps précisément là où elle doit rester continue.

Le texte partiel est composé dans la même face à lire que le transcript figé,
mais en gris : il n'est pas encore acquis. Le curseur ▏ est fusionné au texte
(rich text) pour suivre le dernier mot.
"""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from benji.ui.style import READING_LEADING, current_theme, reading_qss
from benji.ui.widgets.chat_item import (
    _GUTTER_WIDTH,
    _READING_SIZE,
    _SPINE_X,
    _TEXT_MAX_WIDTH,
    _TEXT_X,
)
from benji.ui.widgets.waveform import WaveformDot

_DOT_RADIUS = 3.5


class PartialBubble(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._text: str = ""

        self.wave = WaveformDot(bar_width=2, gap=2, height=14)
        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setMaximumWidth(_TEXT_MAX_WIDTH)

        # La ligne en cours occupe exactement la grille du transcript : l'onde
        # prend la place de l'heure dans la gouttière — pour la ligne vivante, le
        # temps *est* l'onde — le point se pose sur la ligne de temps, et le
        # texte démarre à la même abscisse que tout le reste.
        gutter = QWidget()
        gutter.setFixedWidth(_GUTTER_WIDTH)
        gutter_row = QHBoxLayout(gutter)
        gutter_row.setContentsMargins(0, 0, 0, 0)
        gutter_row.addStretch(1)
        gutter_row.addWidget(self.wave, 0, Qt.AlignmentFlag.AlignTop)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 8, 0, 0)
        row.setSpacing(0)
        row.addWidget(gutter, 0, Qt.AlignmentFlag.AlignTop)
        row.addSpacing(_TEXT_X - _GUTTER_WIDTH)
        row.addWidget(self.text_label, 1)

        self.apply_theme()
        self.setVisible(False)

    def set_text(self, text: str) -> None:
        self._text = text
        self._render()
        visible = bool(text)
        self.setVisible(visible)
        self.wave.set_active(visible)

    def paintEvent(self, event):
        """Prolonge la ligne de temps et y pose le point du direct."""
        super().paintEvent(event)
        t = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = 8 + 9  # même ancrage vertical que la première ligne d'un ChatItem
        painter.setPen(t.spine)
        painter.drawLine(_SPINE_X, 0, _SPINE_X, y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(t.record))
        painter.drawEllipse(
            int(_SPINE_X - _DOT_RADIUS), int(y - _DOT_RADIUS),
            int(_DOT_RADIUS * 2), int(_DOT_RADIUS * 2),
        )
        painter.end()

    def _render(self) -> None:
        if not self._text:
            self.text_label.setText("")
            return
        t = current_theme()
        cursor = f"rgba({t.record.red()},{t.record.green()},{t.record.blue()},255)"
        self.text_label.setText(
            f'<div style="line-height:{READING_LEADING}%;">{escape(self._text)}'
            f'<span style="color:{cursor};"> ▏</span></div>'
        )

    def apply_theme(self) -> None:
        t = current_theme()
        self.wave.set_color(t.record)
        # Même face que le transcript, en gris : dit, pas encore acquis.
        self.text_label.setStyleSheet(reading_qss(t, size=_READING_SIZE, color=t.ink_muted))
        self._render()
        self.update()
