"""Raccourci clavier **global** — celui qui marche quand Benji n'a pas le focus.

Les raccourcis existants sont des `QShortcut` posés sur l'overlay : ils ne
répondent que si Benji est au premier plan. Or pendant une réunion, le focus est
sur Teams ou Zoom, en plein écran — c'est-à-dire exactement la situation où l'on
veut couper le micro d'un geste. Un raccourci qui exige d'aller cliquer sur
Benji d'abord ne sert à rien.

Une implémentation par système, les API n'ayant rien de commun :

- `macos.GlobalHotkeys` — Carbon `RegisterEventHotKey`, sans autorisation.
- `windows.WindowsHotkeys` — `RegisterHotKey` + `WM_HOTKEY` sur la boucle Qt.
- `linux.LinuxHotkeys` — `XGrabKey` (X11 seulement, rien sous Wayland).

Seule l'analyse « Ctrl+Alt+Cmd+B » → codes natifs est partagée, dans
`parsing` : c'est la partie pure et testable. Le reste est du câblage ctypes
qu'aucun test ne peut exercer hors d'une session graphique, enveloppé de
garde-fous : un chargement qui échoue, une combinaison illisible ou déjà prise
par une autre app ne font que journaliser et rendre `register()` False. Un
raccourci absent est une gêne ; une app qui ne démarre pas est une panne.

**Windows et Linux ne sont pas validés sur machine réelle** (aucun poste
disponible) — même statut que Carbon à sa création.
"""

from __future__ import annotations

from benji.config import IS_LINUX, IS_WINDOWS
from benji.hotkeys.macos import GlobalHotkeys
from benji.hotkeys.parsing import parse_shortcut, parse_shortcut_windows, parse_shortcut_x11

__all__ = [
    "GlobalHotkeys",
    "build_hotkeys",
    "parse_shortcut",
    "parse_shortcut_windows",
    "parse_shortcut_x11",
]


def build_hotkeys():
    """Choisit l'implémentation selon l'OS courant.

    Les modules Windows et Linux ne sont importés que sur leur système : leur
    câblage n'a rien à faire en mémoire ailleurs.
    """
    if IS_WINDOWS:
        from benji.hotkeys.windows import WindowsHotkeys

        return WindowsHotkeys()
    if IS_LINUX:
        from benji.hotkeys.linux import LinuxHotkeys

        return LinuxHotkeys()
    return GlobalHotkeys()
