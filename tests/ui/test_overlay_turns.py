"""Overlay : un énoncé peut contenir plusieurs tours de parole.

Un segment VAD tient souvent deux locuteurs qui s'enchaînent sans pause
(cf. `benji/stt/diarization.py`). Le transcripteur émet alors un `final_text` par
tour : l'overlay doit les empiler, pas les écraser l'un après l'autre.
"""

from __future__ import annotations

from queue import Queue

from benji.ui.display_bus import DisplayBus
from benji.ui.overlay import SubtitleOverlay


def _overlay(qtbot) -> SubtitleOverlay:
    w = SubtitleOverlay(DisplayBus(Queue()))
    qtbot.addWidget(w)
    return w


def _final(text, speaker=None, **kw):
    d = {"type": "final_text", "text": text}
    if speaker:
        d["speaker"] = speaker
    d.update(kw)
    return d


def test_two_turns_of_one_segment_are_stacked(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Je pense que oui", "A"))
    w._update_word(_final("Moi non", "B"))

    shown = w.label.text()
    assert "Je pense que oui" in shown  # la réplique du premier n'est pas effacée
    assert "Moi non" in shown
    assert "<br>" in shown  # une ligne par tour


def test_a_new_utterance_clears_the_previous_turns(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Je pense que oui", "A"))
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Autre chose", "A"))

    assert "Je pense que oui" not in w.label.text()


def test_correction_replaces_its_own_turn_only(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("je pense que oui", "A", seq=1))
    w._update_word(_final("moi non", "B", seq=2))
    w._update_word(_final("Moi, non.", "B", seq=2, corrected=True))

    shown = w.label.text()
    assert "je pense que oui" in shown
    assert "Moi, non." in shown
    assert "moi non" not in shown


def test_late_correction_for_a_gone_segment_is_ignored(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("premier", "A", seq=1))
    w._update_word({"type": "segment_start"})
    w._update_word(_final("second", "A", seq=2))
    w._update_word(_final("Premier.", "A", seq=1, corrected=True))

    assert w.label.text().endswith("second")
    assert "Premier." not in w.label.text()


def test_markup_in_a_transcription_is_escaped(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("a <b> & c", "A"))

    assert "&lt;b&gt;" in w.label.text()


def test_drop_clears_every_turn(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Je pense que oui", "A"))
    w._update_word({"type": "final_text", "text": "", "drop": True})

    assert w.label.text() == ""
    assert w._final_lines == []


def test_une_pile_de_tours_ne_deborde_pas_du_plafond(qtbot):
    """La fenêtre est plafonnée à 40 % de l'écran et le label ne défile pas.

    Au-delà, le texte était coupé en silence : 820 px voulus pour 320
    disponibles sur six tours empilés, et la réplique de quelqu'un disparaissait
    sans que rien ne le signale.
    """
    w = _overlay(qtbot)
    long_turn = "Une replique de reunion assez longue pour occuper plusieurs lignes " * 3
    w._update_word({"type": "segment_start"})
    for i in range(6):
        w._update_word(_final(f"[{i}] {long_turn} fin{i}", f"S{i % 3}"))

    budget = w.maximumHeight()
    assert w.label.sizeHint().height() <= budget
    # Le tour le plus récent est celui qu'on lit : c'est lui qui reste. La sonde
    # est en **fin** de tour : quand un seul tour subsiste et deborde encore, son
    # debut est legitimement remplace par une ellipse (cf. le test suivant), donc
    # un marqueur en tete ne survivrait pas a une police un peu plus haute.
    text = w.label.text()
    assert "fin5" in text
    assert "fin4" not in text


def test_un_tour_unique_trop_long_montre_sa_fin(qtbot):
    """On ne peut pas retirer le seul tour présent : on rogne son début."""
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("DEBUT " + "mot " * 400 + "FIN", "A"))

    assert w.label.sizeHint().height() <= w.maximumHeight()
    assert "FIN" in w.label.text()
    assert "…" in w.label.text()
    # Le texte complet reste intact : une correction tardive vise le vrai texte.
    assert w._final_lines[0]["text"].startswith("DEBUT")


def test_une_passe_partielle_est_peinte_en_une_fois(qtbot):
    """Le moteur local publie une passe entière par message.

    Peindre mot à mot coûtait un `setText` + `adjustSize` + `move` par mot —
    10 ms mesurés pour 37 mots — sur un état que personne n'a le temps de lire.
    """
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word({"type": "partial", "words": [
        {"text": "bonjour"}, {"text": "le"}, {"text": "monde"}, {"text": "."},
    ]})

    assert w.label.text() == "bonjour le monde."  # ponctuation recollée


def test_le_mode_remote_reste_servi_mot_a_mot(qtbot):
    """Les events du backend arrivent réellement un par un (cf. stt/remote.py)."""
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    for mot in ("bonjour", "le", "monde"):
        w._update_word({"type": "word", "text": mot})

    assert w.label.text() == "bonjour le monde"


def test_les_sous_titres_affichent_le_nom_donne(qtbot):
    w = _overlay(qtbot)
    w.set_speaker_name("A", "Alice")
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Je pense que oui", "A"))

    shown = w.label.text()
    assert "Alice" in shown
    assert ">A<" not in shown


def test_nommer_repeint_la_replique_deja_a_lecran(qtbot):
    w = _overlay(qtbot)
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Je pense que oui", "A"))
    color_before = w.label.text().split("color:")[1][:7]

    w.set_speaker_name("A", "Alice")

    assert "Alice" in w.label.text()
    # La couleur suit l'étiquette, pas le nom : personne ne change de teinte.
    assert w.label.text().split("color:")[1][:7] == color_before


def test_le_nom_est_echappe(qtbot):
    w = _overlay(qtbot)
    w.set_speaker_name("A", "<b>Bob</b>")
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Salut", "A"))
    assert "&lt;b&gt;Bob" in w.label.text()


def test_une_nouvelle_reunion_rend_les_etiquettes(qtbot):
    w = _overlay(qtbot)
    w.set_speaker_name("A", "Alice")
    w.clear_speaker_names()
    w._update_word({"type": "segment_start"})
    w._update_word(_final("Salut", "A"))
    assert "Alice" not in w.label.text()
