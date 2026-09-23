"""Historique : découpage par réunion, permissions, et coût amorti de `add`.

`add()` est sur le chemin chaud (un appel par segment final, depuis le thread
STT). Le test `test_add_ne_relit_pas_le_fichier_a_chaque_ajout` verrouille
précisément la régression que ce module a connue : une troncature qui relisait
tout le fichier à chaque ajout.
"""

import json
import os
import stat

import pytest

from benji import meetings
from benji.history import TranscriptionHistory


@pytest.fixture
def history(tmp_path):
    return TranscriptionHistory(path=tmp_path / "history.jsonl")


def test_add_tague_la_reunion_courante(history):
    history.add("bonjour")
    entries = history.get_recent()
    assert entries[0]["text"] == "bonjour"
    assert entries[0]["meeting"] == meetings.current_meeting().id


@pytest.mark.posix_perms
def test_fichier_cree_en_0600(history):
    history.add("secret de réunion")
    mode = stat.S_IMODE(os.stat(history.history_file).st_mode)
    assert mode == 0o600


def test_add_ne_relit_pas_le_fichier_a_chaque_ajout(history, monkeypatch):
    """Le compteur de lignes est tenu en mémoire : un seul comptage initial."""
    calls = []
    original = history._count_lines
    monkeypatch.setattr(history, "_count_lines", lambda: (calls.append(1), original())[1])

    for i in range(50):
        history.add(f"segment {i}")

    assert len(calls) == 1


def test_troncature_amortie_puis_effective(tmp_path):
    history = TranscriptionHistory(max_entries=10, path=tmp_path / "h.jsonl")
    for i in range(12):
        history.add(f"s{i}")
    # Sous le plafond + la marge : rien n'est encore tronqué.
    assert len(history.get_recent(100)) == 12

    for i in range(12, 600):
        history.add(f"s{i}")
    entries = history.get_recent(1000)
    assert len(entries) <= 10 + 500
    # La troncature garde bien la fin, pas le début.
    assert entries[0]["text"] == "s599"


def test_get_for_meeting_filtre(history):
    first = meetings.current_meeting().id
    history.add("dans la première")
    second = meetings.start_meeting().id
    history.add("dans la seconde")

    assert [e["text"] for e in history.get_for_meeting(first)] == ["dans la première"]
    assert [e["text"] for e in history.get_for_meeting(second)] == ["dans la seconde"]


def test_entrees_heritees_regroupees_sous_legacy(history):
    # Entrée écrite par une version antérieure : pas de champ `meeting`.
    history.history_file.write_text(
        json.dumps({"timestamp": "2026-01-01T10:00:00", "text": "ancienne"}) + "\n",
        encoding="utf-8",
    )
    history.add("nouvelle")

    assert history.has_legacy_entries() is True
    legacy = history.get_for_meeting(meetings.LEGACY_ID)
    assert [e["text"] for e in legacy] == ["ancienne"]


def test_clear_cible_une_seule_reunion(history):
    first = meetings.current_meeting().id
    history.add("à garder")
    second = meetings.start_meeting().id
    history.add("à effacer")

    history.clear(second)

    assert [e["text"] for e in history.get_for_meeting(first)] == ["à garder"]
    assert history.get_for_meeting(second) == []


def test_clear_global_supprime_tout(history):
    history.add("tout")
    history.clear()
    assert history.get_recent() == []


def test_ligne_corrompue_ignoree(history):
    history.add("valide")
    with open(history.history_file, "a", encoding="utf-8") as f:
        f.write("{ceci n'est pas du json\n")
    assert [e["text"] for e in history.get_recent()] == ["valide"]


def test_migration_depuis_le_cache_legacy(isolated_home):
    """Les données de `~/.cache/benji` sont récupérées, pas abandonnées."""
    legacy_dir = isolated_home / ".cache" / "benji"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "history.jsonl").write_text(
        json.dumps({"timestamp": "2026-01-01T10:00:00", "text": "réunion d'avant"}) + "\n",
        encoding="utf-8",
    )

    history = TranscriptionHistory()

    assert [e["text"] for e in history.get_recent()] == ["réunion d'avant"]
    assert not (legacy_dir / "history.jsonl").exists()
