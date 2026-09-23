"""Palette adaptive, helpers QSS et vibrancy macOS pour la fenêtre principale."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication
from PySide6.QtWidgets import QWidget

from benji.config import IS_MACOS, IS_WINDOWS

log = logging.getLogger(__name__)


def vibrancy_enabled() -> bool:
    """Vibrancy macOS native : opt-in via BENJI_VIBRANCY (1/true/yes).

    Désactivée par défaut — le swap de contentView NSVisualEffectView doit être
    validé en live (grab() ne capture pas la composition native), et le fallback
    dégradé plat reste sûr sur toutes les versions de Qt."""
    return os.environ.get("BENJI_VIBRANCY", "").lower() in ("1", "true", "yes")


# --- Tokens de couleur ------------------------------------------------------
#
# Direction : Benji est un instrument sténographique, pas une app système. Deux
# règles portent tout le reste.
#
# 1. **Une seule couleur saturée, le rouge d'enregistrement.** Il ne signifie
#    qu'une chose : « on est en train de prendre au mot, maintenant ». Il n'est
#    donc jamais utilisé pour une action — un bouton principal rouge se lirait
#    comme un danger sur macOS. L'action principale est un aplat d'encre.
# 2. **Le papier est de la pierre.** Un gris neutre à peine chaud : ni le
#    gris-bleu des Réglages Système (que Benji refuse d'imiter), ni un crème
#    de papeterie. Le document posé dessus, lui, est blanc.
#
# Les couleurs ne sont plus dérivées de la couleur d'accentuation du système :
# c'est précisément ce qui faisait ressembler Benji à une boîte de dialogue.

# Deux plans, pas un aplat : `paper` est le plan de travail (plus profond),
# `sheet` la feuille de lecture posée dessus. C'est le relief, pas la teinte, qui
# fait qu'une surface de lecture paraît récente.
_LIGHT = {
    "paper": QColor("#EDECE9"),
    "sheet": QColor("#FFFFFF"),
    "ink": QColor("#1D1C1A"),
    "record": QColor("#E5484D"),
}
_DARK = {
    "paper": QColor("#0F0F10"),
    "sheet": QColor("#1A1A1C"),
    "ink": QColor("#ECEAE6"),
    "record": QColor("#FF5C61"),
}


@dataclass(frozen=True)
class Theme:
    is_dark: bool
    paper: QColor      # plan de travail : le fond de fenêtre
    sheet: QColor      # la feuille de lecture, surélevée sur le papier
    sheet_edge: QColor # liseré de la feuille
    card: QColor       # creux dans la feuille (champs, listes déroulantes)
    ink: QColor        # texte principal
    ink_muted: QColor  # texte secondaire
    ink_faint: QColor  # métadonnées, horodatages
    ink_ghost: QColor  # traits, états désactivés
    spine: QColor      # la ligne de temps et les séparateurs
    record: QColor     # le direct — la seule couleur saturée

    # --- alias hérités (le code existant lit encore ces noms) ---
    @property
    def label(self) -> QColor:
        return self.ink

    @property
    def secondary_label(self) -> QColor:
        return self.ink_muted

    @property
    def tertiary_label(self) -> QColor:
        return self.ink_faint

    @property
    def quaternary_label(self) -> QColor:
        return self.ink_ghost

    @property
    def window_background(self) -> QColor:
        return self.paper

    @property
    def separator(self) -> QColor:
        return self.spine

    @property
    def accent(self) -> QColor:
        """Le direct. Réservé à l'état « en train d'enregistrer »."""
        return self.record

    @property
    def live_red(self) -> QColor:
        return self.record

    def accent_alpha(self, pct: int) -> QColor:
        return self.color_alpha(self.accent, pct)

    def ink_alpha(self, pct: int) -> QColor:
        return self.color_alpha(self.ink, pct)

    def label_alpha(self, pct: int) -> QColor:
        return self.ink_alpha(pct)

    @staticmethod
    def color_alpha(color: QColor, pct: int) -> QColor:
        c = QColor(color)
        c.setAlpha(int(255 * pct / 100))
        return c


def _is_dark() -> bool:
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return scheme == Qt.ColorScheme.Dark
    except Exception:
        return False


def _alpha(color: QColor, pct: int) -> QColor:
    c = QColor(color)
    c.setAlpha(int(255 * pct / 100))
    return c


def current_theme() -> Theme:
    ensure_system_fonts()
    return theme_for(_is_dark())


def light_theme() -> Theme:
    """La variante claire, quel que soit le thème du système.

    Une seule surface l'exige aujourd'hui : le PDF. Le papier est blanc chez tout
    le monde — exporter le thème sombre donnerait un document au texte presque
    blanc sur fond blanc, illisible à l'impression comme à l'écran.
    """
    return theme_for(False)


def theme_for(dark: bool) -> Theme:
    p = _DARK if dark else _LIGHT
    ink = p["ink"]
    return Theme(
        is_dark=dark,
        paper=p["paper"],
        sheet=p["sheet"],
        sheet_edge=_alpha(ink, 10 if dark else 8),
        # Un champ blanc sur une feuille blanche est invisible : les creux sont
        # une teinte d'encre, ils fonctionnent sur les deux thèmes.
        card=_alpha(ink, 5),
        ink=ink,
        ink_muted=_alpha(ink, 62),
        ink_faint=_alpha(ink, 38),
        ink_ghost=_alpha(ink, 16),
        spine=_alpha(ink, 14 if dark else 12),
        record=p["record"],
    )


def install_theme_listener(callback: Callable[[], None]) -> None:
    """Appelle `callback` chaque fois que le système bascule light/dark."""
    QGuiApplication.styleHints().colorSchemeChanged.connect(lambda _scheme: callback())


# Couleurs de locuteur : une **famille sourde**, pas un arc-en-ciel. Ce sont des
# tiges de quelques pixels le long de la ligne de temps ; elles doivent se
# distinguer sans crier, et surtout ne jamais rivaliser avec le rouge du direct,
# seule couleur saturée de l'interface. Cinq teintes suffisent : au-delà, une
# réunion devient illisible bien avant de manquer de couleurs.
_SPEAKER_PALETTE_LIGHT = [
    QColor("#5B6C8F"),  # ardoise
    QColor("#A2604B"),  # terre
    QColor("#5C7A52"),  # mousse
    QColor("#7A5B7E"),  # prune
    QColor("#8A7238"),  # ocre
]
_SPEAKER_PALETTE_DARK = [
    QColor("#8FA3CC"),
    QColor("#D69277"),
    QColor("#8FB183"),
    QColor("#B392B8"),
    QColor("#C4A863"),
]


def speaker_color(label: str, on_dark: bool | None = None) -> QColor:
    """Couleur stable et lisible pour un locuteur (« A », « B », « S26 »…).

    `on_dark` force la variante claire de la famille : l'overlay est posé sur un
    fond noir quel que soit le thème du système, les teintes sombres y seraient
    illisibles. None = suit le thème courant.
    """
    dark = _is_dark() if on_dark is None else on_dark
    palette = _SPEAKER_PALETTE_DARK if dark else _SPEAKER_PALETTE_LIGHT
    key = sum(ord(c) for c in label) if label else 0
    return QColor(palette[key % len(palette)])


# --- Tokens typographiques --------------------------------------------------
#
# Trois voix, volontairement distinctes : l'interface parle en SF Pro (la voix de
# l'app), les paroles transcrites sont composées en **New York**, le serif système
# de macOS (la voix des gens — c'est un compte rendu, pas un fil de discussion), et
# le temps est en SF Mono, tabulaire, pour que les ticks de la ligne de temps
# s'alignent au pixel. Voir d'un coup d'œil ce qui a été *dit* de ce que l'app
# *dit* est la moitié de la lisibilité d'un transcript.
#
# **Piège : Qt ne voit ni New York ni SF Mono.** macOS les cache à l'énumération
# des familles ; demandées par leur nom, elles retombent en silence sur Charter
# et Menlo — deux des trois voix n'étaient donc pas celles qu'on croyait. Les
# fichiers sont pourtant dans le système : `ensure_system_fonts()` les enregistre
# (familles `.New York`, `.SF NS Mono`). Rien n'est embarqué dans l'app. Hors
# macOS, les piles retombent sur des équivalents.
#
# Pas de famille générique (`serif`, `sans-serif`) en fin de pile : Qt les prend
# pour des noms de famille, ne les trouve pas, et repeuple toute sa table
# d'alias pour rien (~40 ms, avertissement au démarrage).
# Même raison pour des piles propres à chaque OS : une famille absente de la pile
# (« Segoe UI » sur un Mac) déclenche la même recherche.
if IS_MACOS:
    FONT_UI = '".AppleSystemUIFont"'
    FONT_READING = '".New York", Charter, Georgia'
    FONT_MONO = '".SF NS Mono", Menlo'
elif IS_WINDOWS:
    FONT_UI = '"Segoe UI"'
    FONT_READING = 'Georgia, Cambria'
    FONT_MONO = 'Consolas'
else:
    FONT_UI = '"Cantarell", "DejaVu Sans"'
    FONT_READING = '"DejaVu Serif"'
    FONT_MONO = '"DejaVu Sans Mono"'
FONT_DISPLAY = FONT_UI

_SYSTEM_FONT_FILES = (
    "/System/Library/Fonts/NewYork.ttf",
    "/System/Library/Fonts/NewYorkItalic.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
)
_fonts_installed = False


def ensure_system_fonts() -> None:
    """Enregistre New York et SF Mono auprès de Qt, une fois par process.

    Idempotent et sans effet avant la création du `QGuiApplication` (Qt refuse
    d'enregistrer une fonte sans elle) : appelé depuis `current_theme()`, que
    tout widget lit avant de se composer.
    """
    global _fonts_installed
    if _fonts_installed or QGuiApplication.instance() is None:
        return
    _fonts_installed = True
    if not IS_MACOS:
        return
    for path in _SYSTEM_FONT_FILES:
        if os.path.exists(path) and QFontDatabase.addApplicationFont(path) < 0:
            log.warning("Fonte système non chargée : %s", os.path.basename(path))


def reading_font(size: int = 11) -> QFont:
    """La face à lire, en `QFont` — pas en pile CSS.

    `QTextBrowser.setMarkdown` compose lui-même les formats de caractère et
    ignore `setDefaultStyleSheet` : sur un document markdown, seule la fonte par
    défaut du document a un effet. On résout donc la première famille réellement
    installée plutôt que d'espérer qu'une pile CSS soit lue.

    `size` est en **points** (unité de QFont), pas en pixels comme le reste des
    helpers QSS : 11 pt tombe à peu près sur les 15 px du transcript.
    """
    ensure_system_fonts()
    for family in (".New York", "Charter", "Georgia"):
        if family in QFontDatabase.families():
            return QFont(family, size)
    font = QFont()
    font.setStyleHint(QFont.StyleHint.Serif)
    font.setPointSize(size)
    return font


def _rgba(color: QColor) -> str:
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"


def _rgb(color: QColor) -> str:
    return f"rgb({color.red()},{color.green()},{color.blue()})"


# --- Helpers QSS ------------------------------------------------------------
# Source de vérité du look des fenêtres. Toute couleur en dur ailleurs est un bug.


def panel_background_qss(theme: Theme, selector: str = "QWidget") -> str:
    """Fond de fenêtre : un aplat de papier, franc.

    Plus de dégradé vertical : un dégradé sur un fond de fenêtre est un tic des
    années 2010 qui salit la couleur sans rien apporter — le papier doit être
    d'un seul ton pour que la ligne de temps s'y détache.
    """
    return f"""
    {selector} {{ background-color: {_rgb(theme.paper)}; }}"""


def reading_qss(theme: Theme, size: int = 15, color: QColor | None = None) -> str:
    """Composition des paroles transcrites : serif, interligne large."""
    return (
        f"font-family: {FONT_READING}; font-size: {size}px; "
        f"line-height: 1.7; color: {_rgba(color or theme.ink)}; background: transparent;"
    )


# Interligne des paroles. `line-height` en QSS est ignoré par `QLabel` : il ne
# prend effet que dans le texte riche, d'où `reading_html`. Un serif veut plus
# d'air qu'un sans-serif — à 1,0 les lignes de New York se touchaient presque.
# Attention : le pourcentage de Qt s'applique à la hauteur de ligne *native* de
# la fonte, déjà munie d'un interligne ; 148 donnait ~190 % du corps.
READING_LEADING = 118


def reading_html(text: str) -> str:
    """Texte transcrit, échappé et composé avec l'interligne de lecture."""
    from html import escape

    return f'<div style="line-height:{READING_LEADING}%;">{escape(text)}</div>'


def meta_qss(theme: Theme, size: int = 11) -> str:
    """Métadonnées (heures, compteurs) : mono, tabulaire, discret."""
    return (
        f"font-family: {FONT_MONO}; font-size: {size}px; "
        f"letter-spacing: 0.3px; color: {_rgba(theme.ink_faint)}; background: transparent;"
    )


def card_qss(theme: Theme, selector: str = "QWidget", radius: int = 10) -> str:
    """Surface surélevée : carte claire sur le papier, trait d'un cheveu."""
    return f"""
    {selector} {{
        background-color: {_rgba(theme.card)};
        border: 1px solid {_rgba(theme.spine)};
        border-radius: {radius}px;
    }}"""


def text_panel_qss(theme: Theme) -> str:
    """Panneau de lecture (QTextEdit) : c'est un document, pas un log."""
    return f"""
    QTextEdit {{
        font-family: {FONT_READING};
        font-size: 15px;
        color: {_rgba(theme.ink)};
        background-color: transparent;
        border: none;
        padding: 4px 0;
        selection-background-color: {_rgba(theme.ink_alpha(16))};
        selection-color: {_rgba(theme.ink)};
    }}"""


def primary_button_qss(theme: Theme) -> str:
    """Action principale : aplat d'encre.

    Volontairement pas rouge — le rouge ne dit qu'une chose dans Benji, « on
    enregistre ». Un bouton rouge sur macOS se lit comme un danger.
    """
    fg = theme.paper if theme.is_dark else QColor("#FFFFFF")
    # Opaque en sombre : sur la feuille (#191C21) un aplat translucide se délave
    # et le bouton principal perd son autorité.
    fill = theme.ink if theme.is_dark else theme.ink_alpha(92)
    return f"""
    QPushButton {{
        font-family: {FONT_UI};
        font-size: 12px;
        font-weight: 600;
        color: {_rgb(fg)};
        background-color: {_rgba(fill)};
        border: none;
        padding: 7px 16px;
        border-radius: 7px;
    }}
    QPushButton:hover {{ background-color: {_rgba(theme.ink)}; }}
    QPushButton:pressed {{ background-color: {_rgba(theme.ink_alpha(78))}; }}
    QPushButton:disabled {{
        background-color: transparent;
        border: 1px solid {_rgba(theme.spine)};
        color: {_rgba(theme.ink_faint)};
    }}"""


def secondary_button_qss(theme: Theme) -> str:
    """Action secondaire : texte d'encre, fond qui n'apparaît qu'au survol."""
    return f"""
    QPushButton {{
        font-family: {FONT_UI};
        font-size: 12px;
        font-weight: 500;
        color: {_rgba(theme.ink_muted)};
        background-color: transparent;
        border: none;
        padding: 7px 12px;
        border-radius: 7px;
    }}
    QPushButton:hover {{
        color: {_rgba(theme.ink)};
        background-color: {_rgba(theme.ink_alpha(7))};
    }}
    QPushButton:pressed {{ background-color: {_rgba(theme.ink_alpha(12))}; }}
    QPushButton:disabled {{ color: {_rgba(theme.ink_ghost)}; background-color: transparent; }}"""


def destructive_button_qss(theme: Theme) -> str:
    """Action destructive : rouge, mais jamais un aplat — on n'invite pas à cliquer."""
    return f"""
    QPushButton {{
        font-family: {FONT_UI};
        font-size: 12px;
        font-weight: 500;
        color: {_rgba(theme.record)};
        background-color: transparent;
        border: none;
        padding: 7px 12px;
        border-radius: 7px;
    }}
    QPushButton:hover {{ background-color: {_rgba(Theme.color_alpha(theme.record, 12))}; }}
    QPushButton:disabled {{ color: {_rgba(theme.ink_ghost)}; }}"""


def field_qss(theme: Theme) -> str:
    """Champs et listes déroulantes : alignés sur la carte, pas sur le natif Qt."""
    return f"""
    QComboBox, QLineEdit {{
        font-family: {FONT_UI};
        font-size: 12px;
        color: {_rgba(theme.ink)};
        background-color: {_rgba(theme.card)};
        border: 1px solid {_rgba(theme.spine)};
        border-radius: 7px;
        padding: 6px 10px;
        selection-background-color: {_rgba(theme.ink_alpha(16))};
        selection-color: {_rgba(theme.ink)};
    }}
    QComboBox:hover, QLineEdit:hover {{ border-color: {_rgba(theme.ink_alpha(24))}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        font-family: {FONT_UI};
        font-size: 12px;
        color: {_rgba(theme.ink)};
        background-color: {_rgb(theme.paper)};
        border: 1px solid {_rgba(theme.spine)};
        border-radius: 7px;
        padding: 4px;
        outline: none;
        selection-background-color: {_rgba(theme.ink_alpha(10))};
        selection-color: {_rgba(theme.ink)};
    }}"""


def apply_window_vibrancy(window: QWidget) -> bool:
    """macOS vibrancy via NSVisualEffectView en remplaçant la contentView.

    Pattern "wrap" : la contentView Qt de la NSWindow est remplacée par un
    NSVisualEffectView (flou behind-window), l'ancienne vue Qt étant rattachée
    comme subview redimensionnable. Le fond Qt doit être transparent pour laisser
    voir le flou (cf. `_apply_theme` qui teste `_vibrancy_active`).

    Renvoie True si le swap a réussi. No-op + False si désactivé (BENJI_VIBRANCY),
    hors macOS, ou si AppKit échoue — l'appelant retombe alors sur le dégradé plat.
    """
    if not IS_MACOS or not vibrancy_enabled():
        return False
    try:
        import objc
        from AppKit import (
            NSViewHeightSizable,
            NSViewWidthSizable,
            NSVisualEffectView,
        )

        nsview = objc.objc_object(c_void_p=int(window.winId()))
        nswindow = nsview.window()
        if nswindow is None:
            return False

        effect = NSVisualEffectView.alloc().init()
        effect.setFrame_(nsview.bounds())
        effect.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        # BlendingModeBehindWindow=0, StateActive=1, Material UnderWindowBackground=21
        effect.setBlendingMode_(0)
        effect.setState_(1)
        try:
            effect.setMaterial_(21)
        except Exception:
            pass  # matériau indispo sur cette version : garde le défaut

        content = nswindow.contentView()
        nswindow.setContentView_(effect)
        effect.addSubview_(content)
        return True
    except Exception as e:
        log.warning("Vibrancy indisponible, fallback flat : %s", e)
        return False
