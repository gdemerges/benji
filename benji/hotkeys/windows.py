"""Raccourci global Windows — `RegisterHotKey` + interception de `WM_HOTKEY`.

Qt fait déjà tourner la boucle de messages Windows pour ses propres fenêtres :
`QAbstractNativeEventFilter` intercepte les messages qui y transitent au lieu
d'ouvrir un fil dédié. Le callback s'exécute donc sur le thread Qt.

**Non validé sur machine réelle** — aucun poste Windows disponible pour ce
projet ; même statut que Carbon à sa création (cf. le suivi « À valider au
runtime » du vault).
"""

from __future__ import annotations

import ctypes
import logging

from benji.config import IS_WINDOWS
from benji.hotkeys.parsing import parse_shortcut_windows

log = logging.getLogger(__name__)

_WM_HOTKEY = 0x0312
_MOD_NOREPEAT = 0x4000


class WindowsHotkeys:
    """Table de raccourcis globaux Windows, vivante tant que l'objet l'est."""

    def __init__(self):
        from PySide6.QtCore import QAbstractNativeEventFilter

        class _Filter(QAbstractNativeEventFilter):
            def __init__(self, owner):
                super().__init__()
                self._owner = owner

            def nativeEventFilter(self, _event_type, message):
                self._owner._on_native_event(message)
                return False, 0

        self._filter = _Filter(self)
        self._user32 = None
        self._callbacks: dict[int, callable] = {}
        self._next_id = 1
        self._installed = False

    def _load(self):
        if self._user32 is not None:
            return self._user32
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            # Comme Carbon/X11 (cf. GlobalHotkeys._load, LinuxHotkeys._load) :
            # sans argtypes déclarés, ctypes peut tronquer HWND (un pointeur)
            # en c_int et faire dérailler le process sur SIGSEGV.
            user32.RegisterHotKey.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
            ]
            user32.RegisterHotKey.restype = ctypes.c_int
            user32.UnregisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
            user32.UnregisterHotKey.restype = ctypes.c_int
        except Exception as e:
            log.warning("user32 indisponible — pas de raccourci global (%s)", e)
            return None
        self._user32 = user32
        return user32

    def register(self, shortcut: str, callback) -> bool:
        if not IS_WINDOWS:
            return False
        parsed = parse_shortcut_windows(shortcut)
        if parsed is None:
            log.warning("Raccourci global illisible ou sans modificateur : %r", shortcut)
            return False
        user32 = self._load()
        if user32 is None:
            return False

        vk, mods = parsed
        hotkey_id = self._next_id
        # MOD_NOREPEAT : un seul déclenchement par appui, pas une rafale tant
        # que la touche reste enfoncée.
        if not user32.RegisterHotKey(None, hotkey_id, mods | _MOD_NOREPEAT, vk):
            log.warning("Raccourci global %s refusé par le système", shortcut)
            return False

        if not self._installed:
            from PySide6.QtCore import QCoreApplication

            app = QCoreApplication.instance()
            if app is None:
                # Ne devrait pas arriver (app.py installe après _create_qapp),
                # mais un raccourci sans app plutôt qu'un crash au démarrage.
                user32.UnregisterHotKey(None, hotkey_id)
                log.warning("Aucun QCoreApplication actif — raccourci global impossible")
                return False
            app.installNativeEventFilter(self._filter)
            self._installed = True

        self._callbacks[hotkey_id] = callback
        self._next_id += 1
        log.info("Raccourci global actif : %s", shortcut)
        return True

    def _on_native_event(self, message) -> None:
        try:
            import ctypes.wintypes as wintypes

            msg = wintypes.MSG.from_address(int(message))
        except Exception:
            return
        if msg.message != _WM_HOTKEY:
            return
        callback = self._callbacks.get(msg.wParam)
        if callback is not None:
            try:
                callback()
            except Exception as e:
                log.warning("Raccourci global : action en échec (%s)", e)

    def unregister_all(self) -> None:
        if self._user32 is None:
            return
        for hotkey_id in self._callbacks:
            try:
                self._user32.UnregisterHotKey(None, hotkey_id)
            except Exception:
                pass
        self._callbacks.clear()
