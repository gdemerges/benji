"""Transcript figé : regroupement, noms de locuteurs, recomposition."""

from __future__ import annotations

from benji.ui.widgets.transcript_view import TranscriptView


def _entry(ts: str, text: str, speaker: str | None = None) -> dict:
    e = {"timestamp": f"2026-08-21T{ts}", "text": text}
    if speaker:
        e["speaker"] = speaker
    return e


def test_groupe_par_locuteur(qtbot):
    view = TranscriptView()
    qtbot.addWidget(view)
    view.set_entries([
        _entry("10:00:01", "Première.", "A"),
        _entry("10:00:05", "Deuxième.", "A"),
        _entry("10:00:09", "Réponse.", "B"),
    ])

    headers = [i._show_header for i in view._items]
    assert headers == [True, False, True]


def test_heure_affichee_au_changement_de_minute(qtbot):
    view = TranscriptView()
    qtbot.addWidget(view)
    view.set_entries([
        _entry("10:00:01", "Une.", "A"),
        _entry("10:00:40", "Deux.", "B"),
        _entry("10:01:10", "Trois.", "A"),
    ])

    assert [i.ts_label.text() for i in view._items] == ["10:00", "", "10:01"]


def test_les_noms_choisis_remplacent_les_labels(qtbot):
    view = TranscriptView()
    qtbot.addWidget(view)
    view.set_entries([_entry("10:00:01", "Bonjour.", "A")], {"A": "Alice"})

    item = view._items[0]
    # Le nom est un affichage : l'étiquette du moteur reste, sinon la couleur du
    # locuteur changerait au moment où on le nomme.
    assert item._speaker == "A"
    assert item.speaker_label.text() == "Alice"


def test_entrees_desordonnees_sont_remises_dans_l_ordre(qtbot):
    view = TranscriptView()
    qtbot.addWidget(view)
    view.set_entries([
        _entry("10:00:09", "Ensuite.", "A"),
        _entry("10:00:01", "D'abord.", "A"),
    ])

    assert view.plain_text().splitlines() == ["D'abord.", "Ensuite."]


def test_entrees_vides_ignorees_et_etat_vide(qtbot):
    view = TranscriptView()
    qtbot.addWidget(view)
    view.set_entries([_entry("10:00:01", "   ")])

    assert view._items == []
    assert view.empty.isVisible() or not view.scroll.isVisible()


def test_recomposer_remplace_au_lieu_d_empiler(qtbot):
    view = TranscriptView()
    qtbot.addWidget(view)
    view.set_entries([_entry("10:00:01", "Ancienne.", "A")])
    view.set_entries([_entry("10:00:01", "Nouvelle.", "A")])

    assert view.plain_text() == "Nouvelle."
