"""Analyse d'une combinaison de raccourci global.

Seule cette partie est testable hors session graphique : l'enregistrement passe
par Carbon et n'existe que dans une vraie app. Le reste du module est donc écrit
pour ne jamais lever — un raccourci absent est une gêne, une app qui ne démarre
pas est une panne.
"""

import threading

import benji.hotkeys as hotkeys_mod
import benji.hotkeys.linux as linux_mod
import benji.hotkeys.macos as macos_mod
import benji.hotkeys.windows as windows_mod
from benji.hotkeys import (
    GlobalHotkeys,
    build_hotkeys,
    parse_shortcut,
    parse_shortcut_windows,
    parse_shortcut_x11,
)
from benji.hotkeys.linux import LinuxHotkeys
from benji.hotkeys.windows import WindowsHotkeys

CMD, SHIFT, ALT, CTRL = 0x0100, 0x0200, 0x0800, 0x1000


def test_combinaison_complete():
    assert parse_shortcut("Ctrl+Alt+Cmd+B") == (11, CTRL | ALT | CMD)


def test_insensible_a_la_casse_et_aux_espaces():
    assert parse_shortcut(" ctrl + SHIFT + r ") == parse_shortcut("Ctrl+Shift+R")


def test_les_synonymes_de_modificateurs():
    assert parse_shortcut("Option+Cmd+M") == parse_shortcut("Alt+Command+M")
    assert parse_shortcut("⌃+⌥+F5") == parse_shortcut("Ctrl+Alt+F5")


def test_une_touche_nue_est_refusee():
    """Réserver une touche sans modificateur la retirerait de toutes les autres
    applications du système."""
    assert parse_shortcut("B") is None
    assert parse_shortcut("F5") is None


def test_combinaisons_illisibles():
    assert parse_shortcut("") is None
    assert parse_shortcut("Ctrl+") is None
    assert parse_shortcut("Ctrl+Nope") is None
    assert parse_shortcut("Ctrl+A+B") is None
    assert parse_shortcut("Ctrl+Shift") is None  # que des modificateurs


def test_un_raccourci_illisible_ne_leve_pas(monkeypatch):
    """Le démarrage de l'app ne doit jamais dépendre d'un raccourci."""
    monkeypatch.setattr(macos_mod, "IS_MACOS", True)
    assert GlobalHotkeys().register("Ctrl+Nope", lambda: None) is False


def test_carbon_indisponible_degrade_en_silence(monkeypatch):
    monkeypatch.setattr(macos_mod, "IS_MACOS", True)
    hotkeys = GlobalHotkeys()
    monkeypatch.setattr(hotkeys, "_load", lambda: None)

    assert hotkeys.register("Ctrl+Alt+Cmd+B", lambda: None) is False
    hotkeys.unregister_all()  # ne doit pas lever non plus


# --- Windows : RegisterHotKey ---

WIN_ALT, WIN_CTRL, WIN_SHIFT, WIN_CMD = 0x0001, 0x0002, 0x0004, 0x0008


def test_windows_combinaison_complete():
    assert parse_shortcut_windows("Ctrl+Alt+Cmd+B") == (ord("B"), WIN_CTRL | WIN_ALT | WIN_CMD)


def test_windows_meme_tokenizer_que_carbon():
    """Le partage du tokenizer se vérifie par les mêmes refus."""
    assert parse_shortcut_windows("") is None
    assert parse_shortcut_windows("Ctrl+") is None
    assert parse_shortcut_windows("B") is None  # sans modificateur
    assert parse_shortcut_windows("Ctrl+A+B") is None


def test_windows_hotkeys_hors_windows_ne_fait_rien(monkeypatch):
    monkeypatch.setattr(windows_mod, "IS_WINDOWS", False)
    assert WindowsHotkeys().register("Ctrl+Alt+Cmd+B", lambda: None) is False


def test_windows_combinaison_illisible_ne_leve_pas(monkeypatch):
    monkeypatch.setattr(windows_mod, "IS_WINDOWS", True)

    assert WindowsHotkeys().register("Ctrl+Nope", lambda: None) is False


def test_windows_user32_indisponible_degrade_en_silence(monkeypatch):
    """`_load` neutralisé plutôt que compter sur l'absence de `ctypes.windll` :
    sur le runner Windows de la CI, la DLL existe et le raccourci serait
    réellement réservé."""
    monkeypatch.setattr(windows_mod, "IS_WINDOWS", True)

    hotkeys = WindowsHotkeys()
    monkeypatch.setattr(hotkeys, "_load", lambda: None)
    assert hotkeys.register("Ctrl+Alt+Cmd+B", lambda: None) is False
    hotkeys.unregister_all()  # ne doit pas lever non plus


# --- Linux/X11 : XGrabKey ---

X11_SHIFT, X11_CTRL, X11_ALT, X11_CMD = 1, 4, 8, 64


def test_x11_combinaison_complete():
    assert parse_shortcut_x11("Ctrl+Alt+Cmd+B") == ("b", X11_CTRL | X11_ALT | X11_CMD)


def test_x11_touches_speciales():
    assert parse_shortcut_x11("Ctrl+F5") == ("F5", X11_CTRL)
    assert parse_shortcut_x11("Ctrl+Escape") == ("Escape", X11_CTRL)


def test_x11_meme_tokenizer_que_carbon():
    assert parse_shortcut_x11("") is None
    assert parse_shortcut_x11("B") is None
    assert parse_shortcut_x11("Ctrl+A+B") is None


def test_linux_hotkeys_hors_linux_ne_fait_rien(monkeypatch):
    monkeypatch.setattr(linux_mod, "IS_LINUX", False)
    assert LinuxHotkeys().register("Ctrl+Alt+Cmd+B", lambda: None) is False


def test_linux_combinaison_illisible_ne_leve_pas(monkeypatch):
    monkeypatch.setattr(linux_mod, "IS_LINUX", True)

    assert LinuxHotkeys().register("Ctrl+Nope", lambda: None) is False


def test_x11_indisponible_degrade_en_silence(monkeypatch):
    """Couvre aussi bien Xlib absente qu'une session Wayland pure (`_load`
    rend None dans les deux cas, cf. sa docstring)."""
    monkeypatch.setattr(linux_mod, "IS_LINUX", True)

    hotkeys = LinuxHotkeys()
    monkeypatch.setattr(hotkeys, "_load", lambda: None)

    assert hotkeys.register("Ctrl+Alt+Cmd+B", lambda: None) is False
    hotkeys.unregister_all()  # ne doit pas lever non plus


def test_linux_rend_l_action_au_thread_qt(qtbot):
    """Le fil X11 ne doit jamais exécuter l'action lui-même : elle touche le
    tray, et un appel Qt hors du thread principal finit en plantage."""
    hotkeys = LinuxHotkeys()
    fils = []
    hotkeys._callbacks[(56, 4)] = lambda: fils.append(threading.current_thread())

    fil_x11 = threading.Thread(target=hotkeys._dispatch, args=(56, 4))
    fil_x11.start()
    fil_x11.join()

    qtbot.waitUntil(lambda: fils != [], timeout=500)
    assert fils == [threading.main_thread()]


# --- sélection par OS ---


def test_build_hotkeys_choisit_selon_l_os(monkeypatch):
    monkeypatch.setattr(hotkeys_mod, "IS_WINDOWS", True)
    monkeypatch.setattr(hotkeys_mod, "IS_LINUX", False)
    assert isinstance(build_hotkeys(), WindowsHotkeys)

    monkeypatch.setattr(hotkeys_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(hotkeys_mod, "IS_LINUX", True)
    assert isinstance(build_hotkeys(), LinuxHotkeys)

    monkeypatch.setattr(hotkeys_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(hotkeys_mod, "IS_LINUX", False)
    assert isinstance(build_hotkeys(), GlobalHotkeys)
