"""« Ctrl+Alt+Cmd+B » → codes natifs, pour chacun des trois systèmes.

Module **pur** : c'est toute la logique lisible du raccourci global, et la seule
testable hors d'une session graphique. Le tokenizer est partagé ; chaque
système n'a en propre que son jeu de codes de touches et de modificateurs.
"""

from __future__ import annotations

# Alias de modificateurs, communs aux trois systèmes : c'est ce que
# l'utilisateur tape ("Cmd" a un sens même sur Windows/Linux, où il vise le
# meilleur équivalent local — la touche Windows / Super). Seul l'encodage en
# code natif diffère ensuite, par système.
_MOD_ALIASES = {
    "cmd": "cmd", "command": "cmd", "meta": "cmd", "win": "cmd", "super": "cmd", "⌘": "cmd",
    "shift": "shift", "⇧": "shift",
    "alt": "alt", "opt": "alt", "option": "alt", "⌥": "alt",
    "ctrl": "ctrl", "control": "ctrl", "⌃": "ctrl",
}


def _tokenize(text: str) -> tuple[list[str], str] | None:
    """« Ctrl+Alt+Cmd+B » → (["ctrl", "alt", "cmd"], "b"), pure et partagée.

    Seule la validation générique (combinaison vide, deux touches non
    modificatrices, aucun modificateur) est commune : reconnaître la touche
    elle-même et l'encoder revient à l'appelant, chaque système ayant son
    propre jeu de codes.
    """
    if not text:
        return None
    mods: list[str] = []
    key: str | None = None
    for part in (p.strip().lower() for p in text.split("+")):
        if not part:
            continue
        if part in _MOD_ALIASES:
            mods.append(_MOD_ALIASES[part])
        elif key is None:
            key = part
        else:
            return None  # deux touches non modificatrices : combinaison absurde
    if key is None or not mods:
        return None
    return mods, key


def _encode(text: str, keys: dict, modifiers: dict) -> tuple | None:
    tok = _tokenize(text)
    if tok is None:
        return None
    mods, key = tok
    if key not in keys:
        return None
    mask = 0
    for m in mods:
        mask |= modifiers[m]
    return keys[key], mask


# --- macOS : Carbon ---

# Masques de modificateurs Carbon (Events.h). Ce ne sont **pas** ceux de Cocoa.
_MODIFIERS = {"cmd": 0x0100, "shift": 0x0200, "alt": 0x0800, "ctrl": 0x1000}

# Codes de touches virtuelles (kVK_ANSI_*, Carbon/HIToolbox). Ils désignent une
# **position** sur le clavier, pas un caractère : sur un AZERTY, le code 0 est la
# touche marquée « Q ». C'est le comportement attendu d'un raccourci système.
_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
    "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38,
    "k": 40, "n": 45, "m": 46,
    "return": 36, "tab": 48, "space": 49, "escape": 53, "esc": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
    "left": 123, "right": 124, "down": 125, "up": 126,
}


def parse_shortcut(text: str) -> tuple[int, int] | None:
    """« Ctrl+Alt+Cmd+B » → (code de touche Carbon, masque de modificateurs).

    Renvoie None si la combinaison est vide, illisible, ou **sans modificateur** :
    réserver une touche nue à l'échelle du système la retirerait de toutes les
    autres applications.
    """
    return _encode(text, _KEYCODES, _MODIFIERS)


# --- Windows : RegisterHotKey (WinUser.h) ---

_WIN_MODIFIERS = {"alt": 0x0001, "ctrl": 0x0002, "shift": 0x0004, "cmd": 0x0008}  # MOD_*
# Codes de touche virtuelle (VK_*). Lettres/chiffres = leur code ASCII majuscule,
# c'est la seule coïncidence pratique de tout ce paquet.
_WIN_VK = {
    **{c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz"},
    **{d: ord(d) for d in "0123456789"},
    "return": 0x0D, "tab": 0x09, "space": 0x20, "escape": 0x1B, "esc": 0x1B,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}


def parse_shortcut_windows(text: str) -> tuple[int, int] | None:
    """« Ctrl+Alt+Cmd+B » → (code VK, masque MOD_*). None si illisible."""
    return _encode(text, _WIN_VK, _WIN_MODIFIERS)


# --- Linux/X11 : XGrabKey (Xlib) ---

_X11_MODIFIERS = {"shift": 1, "ctrl": 4, "alt": 8, "cmd": 64}  # Shift/Control/Mod1/Mod4
# Noms de touche attendus par `XStringToKeysym` — pas nos propres alias.
_X11_KEYSYMS = {
    **{c: c for c in "abcdefghijklmnopqrstuvwxyz"},
    **{d: d for d in "0123456789"},
    "return": "Return", "tab": "Tab", "space": "space",
    "escape": "Escape", "esc": "Escape",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5", "f6": "F6",
    "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
    "left": "Left", "up": "Up", "right": "Right", "down": "Down",
}


def parse_shortcut_x11(text: str) -> tuple[str, int] | None:
    """« Ctrl+Alt+Cmd+B » → (nom de touche X11, masque Shift/Control/Mod1/Mod4).

    Le nom (pas un code) parce que la conversion en `KeyCode` dépend de la
    disposition clavier active, résolue seulement à l'enregistrement — via
    `XStringToKeysym` puis `XKeysymToKeycode`, contre un vrai `Display`.
    """
    return _encode(text, _X11_KEYSYMS, _X11_MODIFIERS)
