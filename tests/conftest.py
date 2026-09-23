"""Garde-fou global : aucun test ne touche le vrai dossier personnel.

Benji lit et écrit des données réelles sous `~` (historique des réunions,
résumés, identifiants) et sait migrer d'un emplacement à l'autre. Un test qui
laisse fuiter le vrai HOME ne se contente pas de lire à côté : il peut
*déplacer* les transcriptions de l'utilisateur. HOME est donc réécrit vers un
répertoire temporaire pour toute la suite, et l'état de module des réunions est
remis à zéro entre les tests.
"""

import sys

import pytest


def pytest_collection_modifyitems(config, items):
    """`posix_perms` : les bits 0600 n'existent pas sous Windows — `chmod` n'y
    touche que le drapeau lecture seule, NTFS protège par ACL. Le test n'y
    vérifierait rien, il échouerait seulement."""
    if sys.platform != "win32":
        return
    skip = pytest.mark.skip(reason="permissions POSIX sans objet sous Windows (ACL NTFS)")
    for item in items:
        if "posix_perms" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    # Windows : `Path.home()` lit USERPROFILE, pas HOME. Sans cette ligne,
    # l'isolation ne tient que sur macOS/Linux.
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    # Filet : la suite est hermétique et hors ligne. Un test qui oublie de
    # simuler un moteur construirait un vrai backend et téléchargerait un modèle
    # de plusieurs gigaoctets — vu une fois, la suite « figeait » sans rien dire.
    # Hors ligne, l'appel échoue tout de suite et le test pointe le vrai oubli.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from benji import meetings

    meetings.reset_for_tests()
    yield home
    meetings.reset_for_tests()
