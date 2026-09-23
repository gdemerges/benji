"""Raccourci global Linux (X11 seulement) — `XGrabKey`.

Xlib en ctypes, comme Carbon : pas de nouvelle dépendance. Connexion X11
**dédiée**, pompée par un fil à part (Qt utilise XCB, pas Xlib — mélanger les
deux sur la même connexion n'est pas sûr). Les actions, elles, sont **rendues
au thread Qt** : elles touchent l'interface (notification du tray), et un appel
Qt depuis un autre fil est un plantage qui attend son heure.

Grab posé en quatre combinaisons (avec/sans Verr Num, Verr Maj) : X11 inclut
ces verrous dans l'état des modificateurs, les ignorer ferait rater le
raccourci selon l'état du clavier au moment de l'appui.

**Sous Wayland, ça échoue silencieusement** — la capture globale de touches y
est interdite par design sans portail spécifique au compositeur, aucune
solution unifiée n'existe ; `register()` rend False comme n'importe quel
raccourci indisponible, sans jamais bloquer le démarrage.

**Non validé sur machine réelle.**
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading

from PySide6.QtCore import QObject, Qt, Signal

from benji.config import IS_LINUX
from benji.hotkeys.parsing import parse_shortcut_x11

log = logging.getLogger(__name__)

# Verrous que X11 mélange à l'état des modificateurs (Xlib ne les ignore pas
# de lui-même) : sans les inclure dans les combinaisons grabées, le raccourci
# raterait chaque fois que Verr Num ou Verr Maj est actif.
_X11_LOCK_MASKS = (0, 2, 16, 2 | 16)  # aucun, Verr Maj (Lock), Verr Num (Mod2), les deux

_KEY_PRESS = 2


class _XKeyEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int), ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong), ("root", ctypes.c_ulong),
        ("subwindow", ctypes.c_ulong), ("time", ctypes.c_ulong),
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("x_root", ctypes.c_int), ("y_root", ctypes.c_int),
        ("state", ctypes.c_uint), ("keycode", ctypes.c_uint),
        ("same_screen", ctypes.c_int),
    ]


class _XEvent(ctypes.Union):
    """`XEvent` est une union C ; `pad` réserve la taille réelle (24 longs,
    la marge que Xlib.h lui-même garantit) pour que `XNextEvent` n'écrive
    jamais hors de ce que ctypes a alloué, quel que soit le type reçu."""

    _fields_ = [("type", ctypes.c_int), ("xkey", _XKeyEvent), ("pad", ctypes.c_long * 24)]


class _MainThreadInvoker(QObject):
    """Rejoue un callable sur le thread qui a créé l'objet — celui de Qt.

    Même mécanisme que `DisplayBus._wake` : un signal émis depuis un autre fil,
    connecté en `QueuedConnection` à une méthode de ce QObject, est livré par la
    boucle d'événements du thread propriétaire.
    """

    _call = Signal(object)

    def __init__(self):
        super().__init__()
        self._call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def invoke(self, fn) -> None:
        self._call.emit(fn)

    def _run(self, fn) -> None:
        try:
            fn()
        except Exception as e:
            log.warning("Raccourci global : action en échec (%s)", e)


class LinuxHotkeys:
    """Table de raccourcis globaux X11, vivante tant que l'objet l'est.

    À construire sur le thread Qt : c'est là que les actions seront rejouées.
    """

    def __init__(self):
        self._x11 = None
        self._display = None
        self._callbacks: dict[tuple[int, int], callable] = {}  # (keycode, state) → callback
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._invoker = _MainThreadInvoker()

    def register(self, shortcut: str, callback) -> bool:
        if not IS_LINUX:
            return False
        parsed = parse_shortcut_x11(shortcut)
        if parsed is None:
            log.warning("Raccourci global illisible ou sans modificateur : %r", shortcut)
            return False
        x11 = self._load()
        if x11 is None:
            return False

        keysym_name, mods = parsed
        keysym = x11.XStringToKeysym(keysym_name.encode("ascii"))
        keycode = x11.XKeysymToKeycode(self._display, keysym)
        if keycode == 0:
            log.warning("Raccourci global %s : touche introuvable sur ce clavier", shortcut)
            return False
        root = x11.XDefaultRootWindow(self._display)
        try:
            for lock in _X11_LOCK_MASKS:
                x11.XGrabKey(
                    self._display, keycode, mods | lock, root, True, 1, 1  # GrabModeAsync
                )
        except Exception as e:
            log.warning("Raccourci global %s refusé par le système (%s)", shortcut, e)
            return False
        x11.XFlush(self._display)

        for lock in _X11_LOCK_MASKS:
            self._callbacks[(keycode, mods | lock)] = callback
        self._ensure_pump()
        log.info("Raccourci global actif : %s", shortcut)
        return True

    def _load(self):
        if self._x11 is not None:
            return self._x11
        try:
            path = ctypes.util.find_library("X11")
            x11 = ctypes.CDLL(path)
            # Comme Carbon (cf. GlobalHotkeys._load) : sans argtypes déclarés,
            # ctypes peut tronquer un pointeur 64 bits en c_int et faire
            # dérailler le process sur SIGSEGV, sans exception à rattraper.
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
            x11.XStringToKeysym.restype = ctypes.c_ulong
            x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            x11.XKeysymToKeycode.restype = ctypes.c_ubyte
            x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            x11.XDefaultRootWindow.restype = ctypes.c_ulong
            x11.XGrabKey.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            x11.XGrabKey.restype = ctypes.c_int
            x11.XUngrabKey.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong,
            ]
            x11.XUngrabKey.restype = ctypes.c_int
            x11.XFlush.argtypes = [ctypes.c_void_p]
            x11.XFlush.restype = ctypes.c_int
            x11.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.POINTER(_XEvent)]
            x11.XNextEvent.restype = ctypes.c_int
            display = x11.XOpenDisplay(None)
            if not display:
                # Pas de serveur X — Wayland pur, ou aucune session graphique.
                log.warning("Aucune connexion X11 — pas de raccourci global (Wayland ?)")
                return None
        except Exception as e:
            log.warning("Xlib indisponible — pas de raccourci global (%s)", e)
            return None
        self._x11 = x11
        self._display = display
        return x11

    def _ensure_pump(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._pump, daemon=True, name="X11-hotkeys"
        )
        self._thread.start()

    def _pump(self) -> None:
        event = _XEvent()
        while not self._stop.is_set():
            # Bloquant : un fil démon dédié, jamais celui de Qt — une frappe
            # réservée ne doit pas attendre le tick de la boucle applicative.
            try:
                self._x11.XNextEvent(self._display, ctypes.byref(event))
            except Exception:
                return
            if event.type != _KEY_PRESS:
                continue
            self._dispatch(event.xkey.keycode, event.xkey.state)

    def _dispatch(self, keycode: int, state: int) -> None:
        """Appelé sur le fil X11 : ne fait que confier l'action au thread Qt."""
        callback = self._callbacks.get((keycode, state))
        if callback is not None:
            self._invoker.invoke(callback)

    def unregister_all(self) -> None:
        if self._x11 is not None and self._display is not None:
            try:
                root = self._x11.XDefaultRootWindow(self._display)
                for keycode, mods in self._callbacks:
                    self._x11.XUngrabKey(self._display, keycode, mods, root)
                self._x11.XFlush(self._display)
            except Exception:
                pass
        self._callbacks.clear()
        self._stop.set()
