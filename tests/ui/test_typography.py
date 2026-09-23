"""Les trois voix typographiques sont bien celles qu'on croit, et le transcript
est composé en procès-verbal (noms en colonne, à gauche du texte)."""

from __future__ import annotations

import sys

import pytest
from PySide6.QtGui import QFont, QFontInfo

from benji.ui import style
from benji.ui.widgets.chat_item import ChatItem


@pytest.mark.skipif(sys.platform != "darwin", reason="fontes système macOS")
def test_new_york_et_sf_mono_sont_reellement_chargees(qtbot):
    # Demandées par leur nom, elles retombaient en silence sur Charter et Menlo.
    style.ensure_system_fonts()
    assert QFontInfo(QFont(".New York")).family() == ".New York"
    assert QFontInfo(QFont(".SF NS Mono")).family() == ".SF NS Mono"


def test_le_texte_transcrit_est_echappe():
    html = style.reading_html("a < b & c")
    assert "a &lt; b &amp; c" in html
    assert "line-height" in html


def test_le_nom_est_dans_sa_colonne_a_gauche_du_texte(qtbot):
    item = ChatItem("Une phrase assez longue pour tenir sur une ligne.",
                    speaker="A", name="Marie")
    qtbot.addWidget(item)
    item.resize(800, 100)
    item.show()
    qtbot.waitExposed(item)
    name, text = item.speaker_label, item.text_label
    assert name.text() == "Marie"  # casse normale, plus de capitales espacées
    assert name.mapTo(item, name.rect().topRight()).x() < text.mapTo(item, text.rect().topLeft()).x()
    # Même ligne : le nom s'aligne sur la première ligne du texte.
    assert abs(name.mapTo(item, name.rect().topLeft()).y()
               - text.mapTo(item, text.rect().topLeft()).y()) <= 4


def test_un_nom_trop_long_est_elide(qtbot):
    item = ChatItem("Oui.", speaker="A", name="Marie-Charlotte de La Rochefoucauld")
    qtbot.addWidget(item)
    assert item.speaker_label.text().endswith("…")
