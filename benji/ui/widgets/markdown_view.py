"""Rendu markdown partagé — les résumés s'affichent, ils ne se lisent pas en source.

Le même rendu sert à l'onglet Résumés et à la fenêtre Résumé en direct. Cette
dernière affichait jusqu'ici le markdown brut dans une boîte monospace : on y
lisait `**Décision**` au lieu de voir une décision. Deux surfaces qui montrent le
même objet doivent le montrer de la même façon.

`QTextBrowser.setMarkdown` ignore les marges CSS des titres ; elles sont donc
reposées sur les `QTextBlockFormat` après rendu (cf. `render_markdown`).
"""

from __future__ import annotations

import re

from PySide6.QtGui import QFont, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextBrowser

from benji.ui.style import FONT_DISPLAY, FONT_MONO, FONT_READING, FONT_UI, Theme, reading_font

# Marges (haut, bas) en px par niveau de titre.
_HEADING_MARGINS = {1: (2, 14), 2: (22, 6), 3: (16, 4)}
# Corps et graisse des titres. `setMarkdown` ignore aussi le `font-size` du CSS :
# les titres tombaient sur les tailles par défaut de Qt, dans la face à lire —
# un H1 en serif gras de 32 pt qui criait plus fort que tout le résumé.
_HEADING_FONTS = {1: (16, QFont.Weight.DemiBold), 2: (13, QFont.Weight.DemiBold), 3: (12, QFont.Weight.DemiBold)}
# Retrait d'une puce. Les 40 px par défaut de Qt décollaient les listes du texte.
_LIST_INDENT = 20
# « Sujets abordés : » — le deux-points final est une habitude du modèle, pas
# de la typographie : un titre ne s'annonce pas.
_HEADING_COLON = re.compile(r"^(#{1,6}\s.*?)\s*:\s*$", re.MULTILINE)


def _rgba(color) -> str:
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"


def markdown_css(theme: Theme) -> str:
    """Feuille de style d'un résumé : titres en SF Pro, corps dans la face à lire.

    Un résumé est un texte suivi, pas une fiche technique : il est composé dans
    la même face que le transcript, pour que l'app garde une seule voix pour tout
    ce qui vient de la réunion.
    """
    code_bg = _rgba(theme.ink_alpha(7))
    return f"""
        body {{
            font-family: {FONT_READING};
            font-size: 15px;
            line-height: 1.7;
            color: {_rgba(theme.ink)};
            padding: 20px 24px;
        }}
        h1 {{ font-family: {FONT_DISPLAY}; font-size: 21px; font-weight: 600; margin: 0 0 16px 0; }}
        h2 {{ font-family: {FONT_UI}; font-size: 15px; font-weight: 700; margin: 20px 0 10px 0; }}
        h3 {{ font-family: {FONT_UI}; font-size: 13px; font-weight: 700; margin: 16px 0 8px 0; }}
        p {{ margin: 0 0 12px 0; }}
        strong {{ font-weight: 700; }}
        code {{
            font-family: {FONT_MONO}; font-size: 13px;
            background-color: {code_bg}; padding: 1px 5px; border-radius: 3px;
        }}
        pre {{ background-color: {code_bg}; padding: 10px 14px; border-radius: 6px; }}
        pre code {{ background: transparent; padding: 0; }}
        blockquote {{
            border-left: 2px solid {_rgba(theme.spine)};
            padding-left: 14px; color: {_rgba(theme.ink_muted)}; margin: 12px 0;
        }}
        ul, ol {{ margin: 0 0 12px 18px; padding: 0; }}
        li {{ margin-bottom: 5px; }}
        a {{ color: {_rgba(theme.ink)}; text-decoration: underline; }}
    """


def apply_ink(browser: QTextBrowser, theme: Theme) -> None:
    """Couleur du texte par la palette : `setMarkdown` ignore le `color` du CSS.

    Sans elle, le corps suit la palette de la plateforme, qui ne connaît ni
    l'encre brune de Benji ni, hors session macOS, le thème sombre.
    """
    palette = browser.palette()
    palette.setColor(QPalette.ColorRole.Text, theme.ink)
    browser.setPalette(palette)


def render_markdown(browser: QTextBrowser, text: str) -> None:
    """Rend `text` puis repose les marges de titre que le CSS ne peut pas fixer."""
    browser.setMarkdown(_HEADING_COLON.sub(r"\1", text))
    apply_heading_margins(browser.document())


def apply_heading_margins(doc) -> None:
    """Repose les marges de titre sur un `QTextDocument` déjà rendu.

    Séparé de `render_markdown` parce que l'export PDF compose un document nu,
    sans widget autour : il a besoin des mêmes marges, sinon les sections du
    compte rendu se collent les unes aux autres sur le papier.
    """
    doc.setIndentWidth(_LIST_INDENT)
    # La voix de l'app, première famille de la pile (`setFontFamilies` veut des noms nus).
    ui_family = FONT_UI.split(",")[0].strip().strip('"')
    block = doc.begin()
    while block.isValid():
        level = block.blockFormat().headingLevel()
        if level in _HEADING_MARGINS:
            top, bottom = _HEADING_MARGINS[level]
            fmt = block.blockFormat()
            fmt.setTopMargin(top)
            fmt.setBottomMargin(bottom)
            cursor = QTextCursor(block)
            cursor.setBlockFormat(fmt)
            size, weight = _HEADING_FONTS.get(level, _HEADING_FONTS[3])
            char = QTextCharFormat()
            char.setFontFamilies([ui_family])
            char.setFontPointSize(size)
            char.setFontWeight(weight)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.mergeCharFormat(char)
        block = block.next()


class MarkdownView(QTextBrowser):
    """Panneau de lecture markdown, transparent, sans cadre."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.viewport().setAutoFillBackground(False)
        self.document().setDefaultFont(reading_font())
        self.document().setDocumentMargin(20)

    def apply_theme(self, theme: Theme) -> None:
        self.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        self.document().setDefaultStyleSheet(markdown_css(theme))
        apply_ink(self, theme)
        # Re-rendre pour que la nouvelle feuille prenne : le document garde son
        # markdown source, pas le HTML déjà composé.
        current = self.property("_markdown_source") or ""
        if current:
            render_markdown(self, current)

    def set_markdown(self, text: str) -> None:
        self.setProperty("_markdown_source", text)
        render_markdown(self, text)
