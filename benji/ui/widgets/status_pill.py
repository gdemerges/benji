"""Pastille de statut : forme d'onde, réunion en cours, durée.

Quand un `title_provider` est fourni, la pastille affiche le **nom de la réunion
en cours** plutôt que « En écoute » / « En attente » : l'onde dit déjà si Benji
entend quelque chose, répéter l'état en toutes lettres à côté d'elle est une
information de trop, alors que savoir *dans quelle réunion* on est ne se lit
nulle part ailleurs dans la fenêtre. L'état ne reprend la main que lorsqu'il
n'est plus déductible de l'onde : micro en pause.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from benji.ui.style import FONT_MONO, FONT_UI, current_theme
from benji.ui.widgets.waveform import WaveformDot


class StatusPill(QWidget):
    def __init__(self, session_start: datetime, title_provider=None, parent=None):
        super().__init__(parent)
        self._session_start = session_start
        self._title_provider = title_provider
        self._speaking = False
        self._paused = False

        self.wave = WaveformDot(bar_width=2, gap=2, height=12)
        self.status_label = QLabel("En attente")
        self.timer_label = QLabel("00:00")

        # Titre et durée séparés par de l'espace, pas par un « · » : deux voix
        # (SF Pro, SF Mono) les distinguent déjà.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 12, 4)
        layout.setSpacing(0)
        layout.addWidget(self.wave)
        layout.addSpacing(10)
        layout.addWidget(self.status_label)
        layout.addSpacing(12)
        layout.addWidget(self.timer_label)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start()

        self._refresh_state()
        self.apply_theme()

    def apply_theme(self) -> None:
        t = current_theme()
        # Plus de fond de pastille : posé sur le plan de travail, le titre de
        # la réunion n'a pas besoin d'une boîte pour se lire comme un titre.
        self.setStyleSheet(f"""
            QLabel {{
                font-family: {FONT_UI};
                font-size: 12px;
                color: rgba({t.secondary_label.red()},{t.secondary_label.green()},{t.secondary_label.blue()},{t.secondary_label.alpha()});
                background: transparent;
            }}
        """)
        self.status_label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; font-weight: 600; "
            f"color: rgba({t.ink.red()},{t.ink.green()},{t.ink.blue()},{t.ink.alpha()}); "
            "background: transparent;"
        )
        self.timer_label.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 12px; "
            f"color: rgba({t.tertiary_label.red()},{t.tertiary_label.green()},{t.tertiary_label.blue()},{t.tertiary_label.alpha()}); "
            "background: transparent;"
        )
        self._refresh_wave()

    def set_speaking(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        self._refresh_state()

    def set_paused(self, paused: bool) -> None:
        """État pause micro : prime sur « En écoute »/« En attente »."""
        if paused == self._paused:
            return
        self._paused = paused
        if paused:
            self._speaking = False
        self._refresh_state()

    def _refresh_state(self) -> None:
        if self._paused:
            self.status_label.setText("Micro en pause")
        elif self._title_provider is not None:
            self.status_label.setText(self._title() or "Aucune réunion en cours")
        else:
            self.status_label.setText("En écoute" if self._speaking else "En attente")
        self._refresh_wave()

    def _title(self) -> str:
        try:
            return (self._title_provider() or "").strip()
        except Exception:
            return ""

    def _refresh_wave(self) -> None:
        t = current_theme()
        if self._paused:
            self.wave.set_color(t.quaternary_label)
        else:
            self.wave.set_color(t.live_red if self._speaking else t.tertiary_label)
        self.wave.set_active(self._speaking and not self._paused)

    def _tick(self) -> None:
        # Le titre peut changer sous nos pieds (nouvelle réunion, renommage) :
        # il est relu à chaque seconde, c'est une lecture en mémoire.
        if self._title_provider is not None and not self._paused:
            title = self._title() or "Aucune réunion en cours"
            if title != self.status_label.text():
                self.status_label.setText(title)
        delta: timedelta = datetime.now() - self._session_start
        total = int(delta.total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")
