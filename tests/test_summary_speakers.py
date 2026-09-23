"""Le résumé sait qui parle : étiquettes, et prénoms quand on les a donnés."""

from benji import meetings
from benji.llm import summarizer

_LONG = "Je propose qu'on décale la mise en production à la semaine prochaine."


def test_les_repliques_portent_leur_locuteur():
    text = summarizer.prepare_transcription([
        {"text": _LONG, "speaker": "A"},
        {"text": "D'accord.", "speaker": "B"},
        {"text": "Sans locuteur."},
    ])
    assert text.splitlines() == [f"A : {_LONG}", "B : D'accord.", "Sans locuteur."]


def test_les_prenoms_remplacent_les_etiquettes():
    meeting = meetings.store().start()
    meetings.name_speaker("A", "Marie", meeting.id)

    text = summarizer.prepare_transcription([
        {"text": _LONG, "speaker": "A", "meeting": meeting.id},
        {"text": "D'accord.", "speaker": "B", "meeting": meeting.id},
    ])

    assert text.splitlines() == [f"Marie : {_LONG}", "B : D'accord."]


def test_les_noms_restent_propres_a_leur_reunion():
    first, second = meetings.store().start(), meetings.store().start()
    meetings.name_speaker("A", "Marie", first.id)

    renamed = summarizer.with_speaker_names([
        {"text": "x", "speaker": "A", "meeting": first.id},
        {"text": "y", "speaker": "A", "meeting": second.id},
    ])
    assert [e["speaker"] for e in renamed] == ["Marie", "A"]


def test_le_seuil_de_longueur_ignore_les_prefixes():
    # « A : » ajoute des caractères, pas du contenu : trop court reste trop court.
    entries = [{"text": "Oui.", "speaker": "SPEAKER_01"}] * 5
    assert summarizer.prepare_transcription(entries) is None


def test_un_registre_illisible_garde_les_etiquettes(monkeypatch):
    def boom(_):
        raise OSError("registre indisponible")

    monkeypatch.setattr(meetings, "speaker_names", boom)
    renamed = summarizer.with_speaker_names([{"text": "x", "speaker": "A", "meeting": "m1"}])
    assert renamed[0]["speaker"] == "A"


def test_le_prompt_explique_les_locuteurs():
    assert "locuteur" in summarizer.build_user_prompt("A : bonjour")
