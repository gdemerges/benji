"""Fenêtre d'historique : lecture, export et effacement par réunion."""

import json

import pytest
from PySide6.QtWidgets import QMessageBox

from benji import meetings
from benji.ui.history_window import HistoryWindow


@pytest.fixture
def window(qtbot):
    w = HistoryWindow()
    qtbot.addWidget(w)
    return w


def _shown(window) -> str:
    return window.transcript.plain_text()


def test_sans_reunion_l_ecran_est_vide(window):
    assert window.meeting_list.count() == 0
    assert _shown(window) == ""
    assert window.transcript.empty.isVisible() or not window.transcript.scroll.isVisible()
    assert not window.copy_btn.isEnabled()


def test_la_reunion_en_cours_est_selectionnee(window):
    window.history.add("Bonjour.", speaker="A")
    window.reload_meetings()

    assert window.meeting_list.count() == 1
    assert window._meeting_id == meetings.current_meeting_id()
    assert "Bonjour." in _shown(window)


def test_chaque_reunion_montre_ses_propres_entrees(window):
    window.history.add("Dans la première.")
    first = meetings.current_meeting_id()
    meetings.start_meeting("Deuxième")
    window.history.add("Dans la seconde.")
    window.reload_meetings()

    assert "Dans la seconde." in _shown(window)
    assert "Dans la première." not in _shown(window)

    window.meeting_list.setCurrentRow(window._row_for(first))
    assert "Dans la première." in _shown(window)
    assert "Dans la seconde." not in _shown(window)


def test_changer_de_reunion_oublie_les_noms_de_locuteurs(window):
    window.history.add("Bonjour.", speaker="A")
    first = meetings.current_meeting_id()
    meetings.start_meeting()
    window.history.add("Salut.", speaker="A")
    window.reload_meetings()
    window._speaker_names = {"A": "Alice"}

    # « A » n'est pas la même personne d'une réunion à l'autre.
    window.meeting_list.setCurrentRow(window._row_for(first))
    assert window._speaker_names == {}


def test_les_entrees_heritees_restent_lisibles(window):
    window.history.history_file.write_text(
        json.dumps({"timestamp": "2026-01-01T10:00:00", "text": "Ancienne réunion."}) + "\n",
        encoding="utf-8",
    )
    window.history.add("Nouvelle.")
    window.reload_meetings()

    row = window._row_for(meetings.LEGACY_ID)
    assert row >= 0
    window.meeting_list.setCurrentRow(row)
    assert "Ancienne réunion." in _shown(window)
    # Groupe hérité : pas de titre à renommer.
    assert not window.rename_meeting_btn.isEnabled()


def test_nouvelle_reunion_depuis_la_fenetre(window):
    window.history.add("Avant.")
    first = meetings.current_meeting_id()
    window.reload_meetings()

    window._new_meeting()

    assert meetings.current_meeting_id() != first
    assert window._meeting_id == meetings.current_meeting_id()
    assert window.meeting_list.count() == 2


def test_effacer_ne_touche_que_la_reunion_affichee(window, monkeypatch):
    window.history.add("À garder.")
    first = meetings.current_meeting_id()
    second = meetings.start_meeting().id
    window.history.add("À effacer.")
    window.reload_meetings()

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.clear_history()

    assert window.history.get_for_meeting(first) != []
    assert window.history.get_for_meeting(second) == []
    assert meetings.store().get(second) is None


def test_effacer_demande_confirmation(window, monkeypatch):
    window.history.add("À garder.")
    window.reload_meetings()

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Cancel)
    window.clear_history()

    assert window._entries != []


def test_le_nom_de_fichier_d_export_suit_le_titre(window):
    window.history.add("Bonjour.")
    meetings.store().rename(meetings.current_meeting_id(), "Point Produit / Q3")
    window.reload_meetings()

    assert window._meeting_slug() == "point-produit-q3"


# --- recherche ---


def test_la_recherche_filtre_la_liste_des_reunions(window):
    window.history.add("On parle du budget.")
    meetings.start_meeting("Autre sujet")
    window.history.add("On parle de la livraison.")
    window.reload_meetings()
    assert window.meeting_list.count() == 2

    window.search.setText("budget")

    assert window.meeting_list.count() == 1


def test_la_recherche_filtre_aussi_le_compte_rendu(window):
    window.history.add("On parle du budget.")
    window.history.add("Et de la livraison.")
    window.reload_meetings()

    window.search.setText("budget")

    assert "budget" in _shown(window)
    assert "livraison" not in _shown(window)


def test_la_recherche_annonce_le_nombre_de_resultats(window):
    window.history.add("Le budget est validé.")
    window.history.add("Rien à voir.")
    window.reload_meetings()

    window.search.setText("budget")

    assert window.meta_label.text() == "1 résultat sur 2"


def test_effacer_la_recherche_rend_tout(window):
    window.history.add("On parle du budget.")
    window.history.add("Et de la livraison.")
    window.reload_meetings()
    window.search.setText("budget")

    window.search.setText("")

    assert "livraison" in _shown(window)
    assert window.meeting_list.count() == 1


def test_une_reunion_est_trouvee_par_son_titre(window):
    window.history.add("Contenu quelconque.")
    meetings.store().rename(meetings.current_meeting_id(), "Point produit")
    window.reload_meetings()

    window.search.setText("produit")

    assert window.meeting_list.count() == 1


def test_nouvelle_reunion_depuis_la_fenetre_previent_l_app(window, qtbot):
    """L'app doit redemander l'accord de conservation, comme depuis le tray :
    sans ce signal, l'accord de la réunion précédente valait pour la nouvelle."""
    window.history.add("Avant.")
    window.reload_meetings()

    with qtbot.waitSignal(window.current_meeting_changed, timeout=1000):
        window._new_meeting()


def test_effacer_la_reunion_en_cours_previent_l_app_et_l_oublie(window, qtbot, monkeypatch):
    window.history.add("Ce qu'on dit en ce moment.")
    current = meetings.current_meeting_id()
    window.reload_meetings()
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)

    with qtbot.waitSignal(window.current_meeting_changed, timeout=1000):
        window.clear_history()

    # La suite ne s'écrit pas sous l'identifiant d'une réunion effacée.
    assert meetings.current_meeting_id() is None
    window.history.add("La suite.")
    assert meetings.current_meeting_id() != current


def test_effacer_une_ancienne_reunion_ne_touche_pas_a_l_accord(window, qtbot, monkeypatch):
    window.history.add("Ancienne.")
    old = meetings.current_meeting_id()
    meetings.start_meeting()
    window.history.add("En cours.")
    window.reload_meetings()
    window.meeting_list.setCurrentRow(window._row_for(old))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)

    with qtbot.assertNotEmitted(window.current_meeting_changed):
        window.clear_history()


def test_effacer_une_reunion_emporte_ses_resumes(window, monkeypatch):
    from benji.llm import summarizer

    window.history.add("À effacer.")
    doomed = meetings.current_meeting_id()
    path = summarizer.save_summary("Résumé.", window.history.get_for_meeting(doomed))
    window.reload_meetings()
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)

    window.clear_history()

    assert not path.exists()


def test_resumer_passe_par_le_provider_de_l_app(qtbot):
    """Et non par le modèle local d'office : sur Windows, ou avec le cloud
    choisi, il n'y a pas de modèle local."""
    seen = []

    class Provider:
        def summarize(self, entries, on_token=None):
            seen.append(len(entries))
            return "Résumé cloud."

    w = HistoryWindow(summary_provider=Provider())
    qtbot.addWidget(w)
    w.history.add("Bonjour.")
    w.reload_meetings()
    ready = []
    w._summary_ready.connect(lambda s, p: ready.append(s))
    w._on_summary_ready = lambda s, p: None  # pas de boîte modale en test

    w._run_summarize()

    assert seen == [1]
    assert ready == ["Résumé cloud."]


def test_resumer_en_echec_rend_la_main(qtbot, monkeypatch):
    """Une exception du provider figeait le bouton sur « Génération… »."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    class Broken:
        def summarize(self, entries, on_token=None):
            raise RuntimeError("401")

    w = HistoryWindow(summary_provider=Broken())
    qtbot.addWidget(w)
    w.history.add("Bonjour.")
    w.reload_meetings()
    errors = []
    w._summary_error.connect(errors.append)

    w._run_summarize()

    assert errors and "401" in errors[0]
    assert w.summarize_btn.text() == "Résumer"
