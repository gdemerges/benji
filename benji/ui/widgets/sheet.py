"""La feuille de lecture — le plan surélevé sur lequel se pose le transcript.

Ce qui fait qu'une interface de lecture paraît récente n'est pas la teinte du
fond, c'est le **relief** : une fenêtre à un seul aplat se lit comme un panneau
de réglages, deux plans se lisent comme un document posé sur un plan de travail.
Le fond de fenêtre est donc plus profond que la feuille, et la ligne de temps
court sur la feuille — jamais sur le fond.

L'ombre est peinte à la main (quelques rectangles arrondis d'opacité
décroissante) plutôt que confiée à un `QGraphicsDropShadowEffect` : un effet
graphique sur un conteneur qui embarque une zone défilante force un
re-rendu hors écran de tout le sous-arbre à chaque frame de scroll.

**Les anneaux s'additionnent.** Chaque rectangle recouvre les plus grands : près
du bord de la feuille, les opacités se cumulent. La version d'avant visait 20 %
par anneau et obtenait une bande grise de 7 px, lue comme un cadre daté plutôt
que comme un relief. Les valeurs ci-dessous sont donc des **incréments** : leur
somme (~12 % en clair) est l'ombre de contact au ras de la feuille.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

from benji.ui.style import current_theme

_RADIUS = 10
# Épaisseur de l'ombre portée, en px. Volontairement courte : une ombre longue
# fait flotter la feuille, on veut qu'elle soit *posée*.
_SHADOW_SPREAD = 7
# Opacité ajoutée par anneau, du plus large au plus serré (cf. docstring : elles
# s'additionnent). Décalée d'un pixel vers le bas : la lumière vient d'en haut.
_RING_ALPHA_LIGHT = (1, 1, 1, 2, 2, 3, 4)
_RING_ALPHA_DARK = (2, 2, 3, 4, 5, 7, 9)


class Sheet(QWidget):
    """Conteneur qui se peint en feuille surélevée. `layout` reçoit le contenu."""

    def __init__(self, margins: tuple[int, int, int, int] = (0, 0, 0, 0), parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(0)

    def paintEvent(self, event):
        t = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Ombre : des halos concentriques de plus en plus transparents. Plus
        # marquée en thème clair — dans le sombre, c'est la feuille *plus claire*
        # que le fond qui porte le relief, une ombre noire n'y ferait rien.
        rings = _RING_ALPHA_DARK if t.is_dark else _RING_ALPHA_LIGHT
        painter.setPen(Qt.PenStyle.NoPen)
        for step, alpha in enumerate(rings):
            inset = step + 1  # 1 = anneau le plus large
            grow = _SHADOW_SPREAD - inset
            painter.setBrush(QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(
                QRectF(self.rect()).adjusted(inset, inset + 1, -inset, -inset + 1),
                _RADIUS + grow, _RADIUS + grow,
            )

        rect = QRectF(self.rect()).adjusted(
            _SHADOW_SPREAD, _SHADOW_SPREAD, -_SHADOW_SPREAD, -_SHADOW_SPREAD
        )
        painter.setBrush(QColor(t.sheet))
        painter.setPen(QPen(QColor(t.sheet_edge), 1))
        painter.drawRoundedRect(rect, _RADIUS, _RADIUS)
        painter.end()

    def content_margins(self) -> int:
        """Marge à respecter par le contenu pour ne pas déborder de l'ombre."""
        return _SHADOW_SPREAD
