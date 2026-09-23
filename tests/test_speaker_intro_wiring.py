"""Une présentation nomme le locuteur partout — sans écraser, sans écrire avant l'accord."""

from types import SimpleNamespace

import pytest

from benji import meetings
from benji.app import BenjiApplication


class _Overlay:
    def __init__(self, names=None):
        self.names = dict(names or {})

    def speaker_name(self, label):
        return self.names.get(label)

    def set_speaker_name(self, label, name):
        self.names[label] = name


@pytest.fixture
def app(monkeypatch):
    persisted: list[tuple[str, str]] = []
    monkeypatch.setattr(meetings, "name_speaker", lambda label, name: persisted.append((label, name)))
    a = BenjiApplication()
    a.overlay = _Overlay()
    a.main_window = None
    a.transcriber = SimpleNamespace(consent=SimpleNamespace(armed=True))
    a.persisted = persisted
    return a


def _final(text, speaker="A", **extra):
    return {"type": "final_text", "text": text, "speaker": speaker, **extra}


def test_une_presentation_nomme_le_locuteur(app):
    app._on_display_event(_final("Bonjour je m'appelle Guillaume"))
    assert app.overlay.names == {"A": "Guillaume"}
    assert app.persisted == [("A", "Guillaume")]


def test_un_nom_deja_pose_n_est_jamais_remplace(app):
    app.overlay.names["A"] = "Marie"
    app._on_display_event(_final("Moi c'est Karim."))
    assert app.overlay.names == {"A": "Marie"}
    assert app.persisted == []


def test_rien_sur_disque_avant_l_accord(app):
    app.transcriber.consent.armed = False
    app._on_display_event(_final("Je m'appelle Léa", speaker="B"))
    assert app.overlay.names == {"B": "Léa"}
    assert app.persisted == []


@pytest.mark.parametrize("item", [
    _final("Je m'appelle Léa", speaker=None),
    _final("Je m'appelle Léa", corrected=True),
    _final("Je m'appelle Léa", drop=True),
    {"type": "partial", "text": "Je m'appelle Léa", "speaker": "A"},
])
def test_seules_les_lignes_finales_attribuees_comptent(app, item):
    app._on_display_event(item)
    assert app.overlay.names == {}
