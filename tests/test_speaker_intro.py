"""Quelqu'un qui se présente est nommé ; personne d'autre ne l'est."""

import pytest

from benji.speaker_intro import detect_self_name


@pytest.mark.parametrize("text, name", [
    ("Bonjour je m'appelle Guillaume", "Guillaume"),
    ("Bonjour, je m’appelle Guillaume et je suis développeur.", "Guillaume"),
    ("Je m'appelle Guillaume Demergès, enchanté.", "Guillaume Demergès"),
    ("Salut, moi c'est Jean-Pierre.", "Jean-Pierre"),
    ("Mon prénom est Léa.", "Léa"),
    ("Mon nom c'est Karim.", "Karim"),
    ("Je me nomme Éloïse", "Éloïse"),
    ("Hi everyone, my name is Sarah.", "Sarah"),
])
def test_une_presentation_donne_un_nom(text, name):
    assert detect_self_name(text) == name


@pytest.mark.parametrize("text", [
    "",
    "Il s'appelle Marc, il arrive.",           # troisième personne
    "Comment vous vous appelez ?",
    "Je ne m'appelle pas Paul.",
    "Moi c'est bon pour jeudi.",               # pas de majuscule
    "Moi c'est Bon pour moi.",                 # mot courant capitalisé
    "je m'appelle guillaume",                  # casse perdue : on s'abstient
    "Je suis Guillaume.",                      # « je suis » n'est pas un déclencheur
    "On rappelle Guillaume demain.",
])
def test_rien_d_autre_ne_nomme(text):
    assert detect_self_name(text) is None
