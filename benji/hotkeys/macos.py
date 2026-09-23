"""Raccourci global macOS — Carbon `RegisterEventHotKey`.

**Pourquoi Carbon et pas `NSEvent.addGlobalMonitorForEventsMatchingMask_`.** Le
moniteur global Cocoa exige l'autorisation « Surveillance de la saisie » dans
Réglages Système : sans elle il ne lève aucune erreur, il ne se déclenche jamais
— le pire mode d'échec possible. `RegisterEventHotKey`, l'API historique des
raccourcis globaux, **ne demande aucune autorisation** : le système réserve la
combinaison et ne livre que celle-là, ce qui est aussi la garantie de vie privée
qu'on veut donner à l'utilisateur (Benji ne voit pas les autres frappes).

Câblage ctypes qu'aucun test ne peut exercer hors d'une session graphique.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

from benji.config import IS_MACOS
from benji.hotkeys.parsing import parse_shortcut

log = logging.getLogger(__name__)


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_EVENT_CLASS_KEYBOARD = _fourcc("keyb")
_EVENT_HOTKEY_PRESSED = 5
_PARAM_DIRECT_OBJECT = _fourcc("obj ")
_TYPE_HOTKEY_ID = _fourcc("hkid")
_SIGNATURE = _fourcc("bnji")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)


class GlobalHotkeys:
    """Table de raccourcis globaux, vivante tant que l'objet l'est.

    Les références Python (le trampoline ctypes, les identifiants) doivent
    survivre à `register()` : le ramasse-miettes libérerait le callback pendant
    que Carbon en tient encore l'adresse, et la première frappe planterait le
    process. C'est la raison d'être de l'instance.
    """

    def __init__(self):
        self._carbon = None
        self._handler = None      # trampoline ctypes — à garder vivant
        self._callbacks: dict[int, callable] = {}
        self._refs: list = []     # EventHotKeyRef, à garder vivants aussi
        self._next_id = 1

    # --- API ---

    def register(self, shortcut: str, callback) -> bool:
        """Réserve *shortcut* auprès du système. False = raccourci indisponible."""
        if not IS_MACOS:
            return False
        parsed = parse_shortcut(shortcut)
        if parsed is None:
            log.warning("Raccourci global illisible ou sans modificateur : %r", shortcut)
            return False
        carbon = self._load()
        if carbon is None:
            return False

        key_code, modifiers = parsed
        hotkey_id = self._next_id
        try:
            self._install_handler(carbon)
            ref = ctypes.c_void_p()
            status = carbon.RegisterEventHotKey(
                ctypes.c_uint32(key_code),
                ctypes.c_uint32(modifiers),
                _EventHotKeyID(_SIGNATURE, hotkey_id),
                carbon.GetApplicationEventTarget(),
                ctypes.c_uint32(0),
                ctypes.byref(ref),
            )
        except Exception as e:
            log.warning("Raccourci global %s indisponible (%s)", shortcut, e)
            return False
        if status != 0:
            # Le plus souvent : une autre application tient déjà la combinaison.
            log.warning("Raccourci global %s refusé par le système (statut %s)",
                        shortcut, status)
            return False

        self._callbacks[hotkey_id] = callback
        self._refs.append(ref)
        self._next_id += 1
        log.info("Raccourci global actif : %s", shortcut)
        return True

    def unregister_all(self) -> None:
        carbon = self._carbon
        if carbon is None:
            return
        for ref in self._refs:
            try:
                carbon.UnregisterEventHotKey(ref)
            except Exception:
                pass
        self._refs.clear()
        self._callbacks.clear()

    # --- câblage ---

    def _load(self):
        if self._carbon is not None:
            return self._carbon
        try:
            path = ctypes.util.find_library("Carbon")
            carbon = ctypes.CDLL(path)
            # Toute fonction appelée ici doit déclarer ses `argtypes` : sans
            # eux, ctypes passe un pointeur Python en `c_int` et rabote les 32
            # bits de poids fort. Carbon déréférence alors une demi-adresse et
            # le process meurt sur SIGSEGV, sans exception à rattraper.
            carbon.GetApplicationEventTarget.argtypes = []
            carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
            carbon.RegisterEventHotKey.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID,
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
            ]
            carbon.RegisterEventHotKey.restype = ctypes.c_int32
            carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
            carbon.UnregisterEventHotKey.restype = ctypes.c_int32
            carbon.InstallEventHandler.argtypes = [
                ctypes.c_void_p, _HANDLER, ctypes.c_ulong,
                ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            carbon.InstallEventHandler.restype = ctypes.c_int32
            carbon.GetEventParameter.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
            ]
            carbon.GetEventParameter.restype = ctypes.c_int32
        except Exception as e:
            log.warning("Carbon indisponible — pas de raccourci global (%s)", e)
            return None
        self._carbon = carbon
        return carbon

    def _install_handler(self, carbon) -> None:
        """Un seul gestionnaire pour tous les raccourcis, posé au premier."""
        if self._handler is not None:
            return
        handler = _HANDLER(self._dispatch)
        spec = _EventTypeSpec(_EVENT_CLASS_KEYBOARD, _EVENT_HOTKEY_PRESSED)
        status = carbon.InstallEventHandler(
            carbon.GetApplicationEventTarget(), handler,
            1, ctypes.byref(spec), None, None,
        )
        if status != 0:
            raise OSError(f"InstallEventHandler a échoué (statut {status})")
        # Assigné seulement en cas de succès : un handler mémorisé alors que
        # Carbon ne l'a pas pris ferait croire à `register()` que le câblage est
        # posé, et le raccourci suivant ne réessaierait jamais.
        self._handler = handler

    def _dispatch(self, _next_handler, event, _user_data) -> int:
        """Appelé par Carbon sur le thread principal, à chaque frappe réservée."""
        try:
            hotkey = _EventHotKeyID()
            self._carbon.GetEventParameter(
                event, _PARAM_DIRECT_OBJECT, _TYPE_HOTKEY_ID, None,
                ctypes.c_uint32(ctypes.sizeof(hotkey)), None, ctypes.byref(hotkey),
            )
            callback = self._callbacks.get(hotkey.id)
            if callback is not None:
                callback()
        except Exception as e:
            # Une exception qui remonterait dans Carbon tuerait le process : le
            # raccourci ne doit jamais pouvoir faire tomber une réunion en cours.
            log.warning("Raccourci global : action en échec (%s)", e)
        return 0  # noErr
