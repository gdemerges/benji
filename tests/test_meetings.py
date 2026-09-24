"""Registre des réunions : cycle de vie, persistance, robustesse du fichier."""

import json
import os
import stat
from datetime import datetime, timedelta

import pytest

from benji import meetings
from benji.meetings import Meeting, MeetingStore


@pytest.fixture
def store(tmp_path):
    return MeetingStore(path=tmp_path / "meetings.json")


def test_start_ouvre_une_reunion_titree(store):
    meeting = store.start(now=datetime(2026, 8, 21, 14, 30))
    assert meeting.title == "Réunion du 21/08 à 14:30"
    assert meeting.ended_at is None
    assert [m.id for m in store.list()] == [meeting.id]


def test_start_clot_la_precedente(store):
    first = store.start(now=datetime(2026, 8, 21, 9, 0))
    store.start(now=datetime(2026, 8, 21, 10, 0))

    reloaded = store.get(first.id)
    assert reloaded.ended_at == datetime(2026, 8, 21, 10, 0)


def test_list_trie_du_plus_recent_au_plus_ancien(store):
    old = store.start(now=datetime(2026, 8, 20, 9, 0))
    recent = store.start(now=datetime(2026, 8, 21, 9, 0))
    assert [m.id for m in store.list()] == [recent.id, old.id]


def test_rename_et_delete(store):
    meeting = store.start()
    store.rename(meeting.id, "  Point produit  ")
    assert store.get(meeting.id).title == "Point produit"

    # Un titre vide ne doit pas effacer le nom existant.
    store.rename(meeting.id, "   ")
    assert store.get(meeting.id).title == "Point produit"

    store.delete(meeting.id)
    assert store.get(meeting.id) is None


@pytest.mark.posix_perms
def test_fichier_ecrit_en_0600(store):
    store.start()
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600


def test_fichier_corrompu_ne_fait_pas_perdre_l_app(store):
    store.path.write_text("{pas du json", encoding="utf-8")
    assert store.list() == []
    meeting = store.start()  # on repart proprement
    assert store.get(meeting.id) is not None


def test_entree_invalide_ignoree_sans_perdre_les_autres(store):
    valid = store.start(now=datetime(2026, 8, 21, 9, 0))
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw.append({"id": "cassée"})  # pas de started_at
    store.path.write_text(json.dumps(raw), encoding="utf-8")

    assert [m.id for m in store.list()] == [valid.id]


def test_meeting_roundtrip():
    started = datetime(2026, 8, 21, 9, 0)
    meeting = Meeting(id="x", title="T", started_at=started, ended_at=started + timedelta(hours=1))
    assert Meeting.from_dict(meeting.to_dict()) == meeting


# --- état de module (réunion courante) ---


def test_current_meeting_id_ne_cree_rien():
    """Un chemin de lecture ne doit jamais ouvrir une réunion vide."""
    assert meetings.current_meeting_id() is None
    assert meetings.store().list() == []


def test_current_meeting_est_stable_entre_appels():
    first = meetings.current_meeting()
    assert meetings.current_meeting().id == first.id
    assert meetings.current_meeting_id() == first.id


def test_start_meeting_clot_la_courante():
    first = meetings.current_meeting()
    second = meetings.start_meeting("Rétro")

    assert second.id != first.id
    assert second.title == "Rétro"
    assert meetings.store().get(first.id).ended_at is not None


def test_end_current_meeting_horodate_la_fin():
    meeting = meetings.current_meeting()
    meetings.end_current_meeting()

    assert meetings.store().get(meeting.id).ended_at is not None
    assert meetings.current_meeting_id() is None


# --- noms de locuteurs et moments marqués ---


def test_un_locuteur_nomme_le_reste_dune_relecture_a_lautre(tmp_path):
    """Nommé pendant la réunion, quand on sait encore qui parle : trois jours
    plus tard, plus personne ne se souvient de qui était « SPEAKER_01 »."""
    store = meetings.MeetingStore(path=tmp_path / "meetings.json")
    meeting = store.start("Point produit")

    store.name_speaker(meeting.id, "SPEAKER_01", "Alice")

    assert store.get(meeting.id).speakers == {"SPEAKER_01": "Alice"}


def test_un_nom_vide_denomme_le_locuteur(tmp_path):
    store = meetings.MeetingStore(path=tmp_path / "meetings.json")
    meeting = store.start()
    store.name_speaker(meeting.id, "A", "Alice")

    store.name_speaker(meeting.id, "A", "")

    assert store.get(meeting.id).speakers == {}


def test_un_registre_ecrit_avant_les_noms_reste_lisible(tmp_path):
    """Absent n'est pas invalide : une ligne sans `speakers` ni `marks` doit
    continuer de se charger, sinon une mise à jour perdrait tout l'historique."""
    path = tmp_path / "meetings.json"
    path.write_text(
        '[{"id": "abc", "title": "Ancienne", "started_at": "2026-08-01T10:00:00",'
        ' "ended_at": null}]',
        encoding="utf-8",
    )
    store = meetings.MeetingStore(path=path)

    meeting = store.get("abc")
    assert meeting is not None
    assert meeting.speakers == {}
    assert meeting.marks == []


def test_un_moment_marque_survit_a_la_reunion(tmp_path):
    from datetime import datetime

    store = meetings.MeetingStore(path=tmp_path / "meetings.json")
    meeting = store.start()
    at = datetime(2026, 8, 31, 14, 32, 10)

    store.add_mark(meeting.id, at)
    store.add_mark(meeting.id, at)  # deux fois le même instant = une marque

    assert store.get(meeting.id).marks == [at]


def test_une_marque_designe_ce_qui_vient_detre_dit():
    """On marque en réaction : la marque tombe *après* la phrase visée."""
    from datetime import datetime

    entries = [
        {"timestamp": "2026-08-31T14:00:00", "text": "Premier point."},
        {"timestamp": "2026-08-31T14:05:00", "text": "Le chiffre important."},
        {"timestamp": "2026-08-31T14:09:00", "text": "Autre chose."},
    ]
    marks = [datetime(2026, 8, 31, 14, 5, 20)]

    assert meetings.marked_indices(entries, marks) == {1}


def test_une_marque_anterieure_au_transcript_naccroche_rien():
    """Plutôt que de décorer la première phrase venue."""
    from datetime import datetime

    entries = [{"timestamp": "2026-08-31T14:00:00", "text": "Premier point."}]

    assert meetings.marked_indices(entries, [datetime(2026, 8, 31, 13, 0)]) == set()


def test_effacer_la_reunion_en_cours_l_oublie_comme_courante():
    """Sinon la suite de la transcription s'écrivait sous l'identifiant d'une
    réunion effacée : des entrées orphelines, sans titre ni registre."""
    first = meetings.current_meeting()
    meetings.delete_meeting(first.id)

    assert meetings.current_meeting_id() is None
    assert meetings.store().get(first.id) is None
    assert meetings.current_meeting().id != first.id


def test_effacer_une_autre_reunion_garde_la_courante():
    old = meetings.start_meeting("Ancienne")
    current = meetings.start_meeting("En cours")
    meetings.delete_meeting(old.id)

    assert meetings.current_meeting_id() == current.id
