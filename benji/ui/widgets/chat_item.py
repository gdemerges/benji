"""Une prise de parole, posée le long de la ligne de temps.

L'élément signature de Benji : la gouttière n'est pas une marge, c'est une
**ligne de temps continue**. Chaque item peint son propre segment de filet, si
bien que les items empilés forment un trait ininterrompu du haut du transcript
jusqu'à la ligne en cours. S'y accrochent deux marques, qui portent chacune une
information vraie plutôt qu'un ornement :

- un **tick** horizontal quand la minute change (l'heure est écrite en regard,
  dans la gouttière) ;
- une **tige** colorée, à la couleur du locuteur, qui court sur toute la hauteur
  de sa prise de parole — on voit donc *qui* a parlé et *combien de temps* en
  parcourant la marge des yeux, sans lire une ligne.

La pastille ● devant le nom a disparu : la tige dit déjà la couleur, et deux
marques pour la même information, c'est une de trop.

**Mise en page de procès-verbal.** Le nom du locuteur a sa propre colonne, à
gauche du texte, comme dans un compte rendu de séance : on lit *qui* en
descendant la colonne, *quoi* en lisant la page. Il n'apparaît qu'au début d'une
prise de parole, en casse normale et à taille de lecture — les petites
capitales espacées de 10 px d'avant se faisaient écraser par le texte, et « B »
seul y ressemblait à une coquille.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from benji.ui.style import (
    FONT_UI,
    current_theme,
    meta_qss,
    reading_html,
    reading_qss,
    speaker_color,
)

# Gouttière de l'heure, la ligne de temps, la tige du locuteur, la colonne des
# noms, puis le texte.
_GUTTER_WIDTH = 52
_SPINE_X = _GUTTER_WIDTH + 10   # abscisse du filet vertical
_STEM_X = _SPINE_X + 9          # abscisse de la tige colorée
_NAME_X = _STEM_X + 14          # début de la colonne des noms
_NAME_WIDTH = 92                # « Marie-Claire » tient ; au-delà, ellipse
_TEXT_X = _NAME_X + _NAME_WIDTH + 8   # début du texte
# Mesure du texte : ~70 signes en New York 16 px. Plus large, l'œil perd la
# ligne suivante au retour ; la feuille garde le reste en marge.
_TEXT_MAX_WIDTH = 620
_READING_SIZE = 16
_TICK_HALF = 3                  # demi-longueur du tick horizontal
_MARK_RADIUS = 3.0              # encoche d'un moment marqué, posée sur le filet


class ChatItem(QWidget):
    # Clic sur l'en-tête du locuteur : l'étiquette (jamais le nom affiché).
    speaker_clicked = Signal(str)

    def __init__(self, text: str, ts: datetime | None = None, speaker: str | None = None,
                 show_header: bool = True, show_ts: bool = True, seq: int | None = None,
                 name: str | None = None, parent=None):
        """`speaker` est l'étiquette du moteur — elle porte la couleur et reste
        stable ; `name` est le nom donné par l'utilisateur, qui n'est qu'un
        affichage. Les confondre ferait changer la couleur d'un locuteur au
        moment où on le nomme."""
        super().__init__(parent)
        self._text = text
        self._ts = ts or datetime.now()
        self._speaker = speaker
        self._name = name
        self._marked = False
        self._show_header = show_header
        self._show_ts = show_ts
        self.seq = seq  # permet à LiveTab de remplacer le texte corrigé

        # Gouttière : l'heure, en mono, seulement quand la minute change.
        self.ts_label = QLabel(self._ts.strftime("%H:%M") if show_ts else "")
        self.ts_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.ts_label.setFixedWidth(_GUTTER_WIDTH)

        # Colonne des noms : le nom seul, en début de prise de parole ; une
        # cellule vide sinon, pour que le texte garde son aplomb.
        self.speaker_label: QLabel | None = None
        name_cell = QWidget()
        name_cell.setFixedWidth(_NAME_WIDTH)
        name_col = QVBoxLayout(name_cell)
        name_col.setContentsMargins(0, 0, 0, 0)
        if speaker and show_header:
            self.speaker_label = QLabel()
            self.speaker_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            # Nommer un locuteur n'était accessible que par un clic droit, sans
            # rien qui le laisse deviner : l'en-tête lui-même est le bon endroit.
            self.speaker_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self.speaker_label.setToolTip("Cliquer pour nommer ce locuteur")
            self.speaker_label.installEventFilter(self)
            self._set_speaker_text()
            name_col.addWidget(self.speaker_label, 0, Qt.AlignmentFlag.AlignTop)
        name_col.addStretch(1)

        self.text_label = QLabel()
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setWordWrap(True)
        self.text_label.setMaximumWidth(_TEXT_MAX_WIDTH)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_label.setText(reading_html(self._text))

        layout = QHBoxLayout(self)
        # Nouvelle prise de parole : de l'air au-dessus ; suite : serré.
        top = 14 if show_header else 2
        layout.setContentsMargins(0, top, 0, 2)
        layout.setSpacing(0)
        layout.addWidget(self.ts_label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addSpacing(_NAME_X - _GUTTER_WIDTH)
        layout.addWidget(name_cell)
        layout.addSpacing(_TEXT_X - _NAME_X - _NAME_WIDTH)
        # Pas de drapeau d'alignement ici : avec lui, Qt donne au libellé sa
        # largeur « idéale » et calcule mal la hauteur d'un texte à retours à
        # la ligne — des lignes entières disparaissaient.
        layout.addWidget(self.text_label, 1)
        layout.addStretch(0)

        self.apply_theme()
        self._fade_in()

    def eventFilter(self, obj, event):
        if (obj is self.speaker_label
                and event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton):
            self.speaker_clicked.emit(self._speaker)
            return True
        return super().eventFilter(obj, event)

    def set_speaker_name(self, name: str | None) -> None:
        """Change le nom **affiché** du locuteur, jamais son étiquette."""
        self._name = name
        self._set_speaker_text()

    def _set_speaker_text(self) -> None:
        if self.speaker_label is None or not self._speaker:
            return
        full = self._name or self._speaker
        metrics = self.speaker_label.fontMetrics()
        self.speaker_label.setText(
            metrics.elidedText(full, Qt.TextElideMode.ElideRight, _NAME_WIDTH - 4)
        )

    def set_marked(self, marked: bool) -> None:
        """Marque (ou démarque) ce moment sur la ligne de temps."""
        if marked == self._marked:
            return
        self._marked = marked
        self.update()

    @property
    def marked(self) -> bool:
        return self._marked

    def set_text(self, text: str) -> None:
        """Remplace le texte affiché (correction LLM asynchrone)."""
        self._text = text
        self.text_label.setText(reading_html(text))

    # --- peinture de la ligne de temps ---

    def paintEvent(self, event):
        """Peint le segment de filet, le tick de minute et la tige du locuteur.

        Chaque item peint de `y=0` à `y=height()` : mis bout à bout, les segments
        ne laissent aucun trou et le filet paraît continu sur tout le transcript.
        """
        super().paintEvent(event)
        t = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(t.spine, 1))
        painter.drawLine(_SPINE_X, 0, _SPINE_X, self.height())

        if self._show_ts:
            y = self._first_line_y()
            painter.drawLine(_SPINE_X - _TICK_HALF, y, _SPINE_X + _TICK_HALF, y)

        if self._speaker:
            # Une prise de parole s'étale sur plusieurs items (un par segment
            # final). La tige part du haut de l'item quand celui-ci prolonge le
            # groupe : mises bout à bout, les tiges forment un ruban continu qui
            # dit la durée réelle du tour de parole, pas la hauteur d'une ligne.
            stem_top = (self._first_line_y() - 5) if self._show_header else 0
            stem_bottom = self.height()
            if stem_bottom > stem_top:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(speaker_color(self._speaker)))
                painter.drawRoundedRect(
                    _STEM_X, stem_top, 2, stem_bottom - stem_top, 1, 1
                )

        if self._marked:
            # Un moment marqué est une **encoche sur la ligne de temps**, pas une
            # icône à côté du texte : la marge dit déjà quand et qui, elle est
            # l'endroit juste pour dire « ça ». Peinte en encre et non en rouge —
            # le rouge ne signifie que le direct, et une marque survit à la
            # réunion. On perce le filet pour que l'encoche s'y inscrive au lieu
            # de flotter dessus.
            y = self._first_line_y()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(t.paper))
            painter.drawEllipse(QPointF(_SPINE_X, y), _MARK_RADIUS + 2, _MARK_RADIUS + 2)
            painter.setBrush(QColor(t.ink))
            painter.drawEllipse(QPointF(_SPINE_X, y), _MARK_RADIUS, _MARK_RADIUS)
        painter.end()

    def _first_line_y(self) -> int:
        """Ordonnée de la première ligne de texte, tick et tige s'y accrochent."""
        return self.text_label.y() + 9

    def apply_theme(self) -> None:
        t = current_theme()
        self.ts_label.setStyleSheet(meta_qss(t) + " padding-top: 8px;")
        if self.speaker_label is not None:
            # Aligné sur la première ligne du texte : la hauteur d'x du SF à
            # 13 px tombe sur celle du New York à 16 px avec ce retrait.
            self.speaker_label.setStyleSheet(
                f"font-family: {FONT_UI}; font-size: 13px; font-weight: 600; "
                f"color: {_rgba(speaker_color(self._speaker))}; "
                "background: transparent; padding-top: 1px;"
            )
            self._set_speaker_text()
        self.text_label.setStyleSheet(reading_qss(t, size=_READING_SIZE))
        self.update()

    def _fade_in(self) -> None:
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.0)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(220)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


def _rgba(color: QColor) -> str:
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"
