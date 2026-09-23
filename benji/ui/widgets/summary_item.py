"""Item de la liste des résumés."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from benji.ui.style import FONT_MONO, FONT_UI, current_theme


class SummaryItem(QWidget):
    def __init__(self, dt: datetime, snippet: str, parent=None):
        super().__init__(parent)
        self._dt = dt
        self._snippet = snippet

        # Le jour est déjà dans l'en-tête de groupe : l'item dit l'heure, puis
        # de quoi on a parlé — deux lignes d'extrait, pas une date répétée.
        self.date_label = QLabel(self._format_date(dt))
        self.date_label.hide()
        self.time_label = QLabel(dt.strftime("%H:%M"))
        self.snippet_label = QLabel(snippet or "Résumé sans contenu")
        self.snippet_label.setTextFormat(Qt.TextFormat.PlainText)
        # Ellipse plutôt que retour à la ligne : un QLabel à retours dans un
        # item de QListWidget ne connaît pas sa largeur au moment où l'item
        # fixe sa hauteur, et le texte était coupé net au bord.
        self.snippet_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(self.date_label, 0)
        top.addWidget(self.time_label, 0)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)
        layout.addLayout(top)
        layout.addWidget(self.snippet_label)

        self.apply_theme()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        text = self._snippet or "Résumé sans contenu"
        width = max(self.snippet_label.width(), 1)
        metrics = self.snippet_label.fontMetrics()
        self.snippet_label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideRight, width))
        self.snippet_label.setToolTip(text if metrics.horizontalAdvance(text) > width else "")

    @staticmethod
    def _format_date(dt: datetime) -> str:
        # `%b` suit la locale du process : « 28 Aug » dans une app en français.
        from benji.ui.summaries_tab import _MOIS

        return f"{dt.day} {_MOIS[dt.month - 1]}"

    def apply_theme(self) -> None:
        t = current_theme()
        self.date_label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; font-weight: 600; "
            f"color: rgba({t.label.red()},{t.label.green()},{t.label.blue()},{t.label.alpha()}); "
            "background: transparent;"
        )
        self.time_label.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 11px; "
            f"color: rgba({t.tertiary_label.red()},{t.tertiary_label.green()},{t.tertiary_label.blue()},{t.tertiary_label.alpha()}); "
            "background: transparent;"
        )
        self.snippet_label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; "
            f"color: rgba({t.label.red()},{t.label.green()},{t.label.blue()},{t.label.alpha()}); "
            "background: transparent;"
        )
        # Le corps vient de changer : l'ellipse se recalcule avec la nouvelle métrique.
        self._elide()
