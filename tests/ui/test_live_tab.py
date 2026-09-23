"""LiveTab redesign : état vide, regroupement par locuteur, corrections."""

from __future__ import annotations

import pytest

from benji.ui.live_tab import LiveTab


def _items(tab: LiveTab):
    """ChatItems présents dans la colonne (sans le stretch final)."""
    out = []
    for i in range(tab.content_layout.count()):
        w = tab.content_layout.itemAt(i).widget()
        if w is not None:
            out.append(w)
    return out


def _final(text, speaker=None, seq=None, **kw):
    d = {"type": "final_text", "text": text, "speaker": speaker, "seq": seq}
    d.update(kw)
    return d


def test_empty_state_then_first_final(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.empty.isVisible()
    assert not tab.scroll.isVisible()

    tab.on_event(_final("Bonjour.", "A", 1))
    assert not tab.empty.isVisible()
    assert tab.scroll.isVisible()
    assert len(_items(tab)) == 1


def test_grouping_same_speaker_hides_header(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Première phrase.", "A", 1))
    tab.on_event(_final("Deuxième phrase.", "A", 2))
    tab.on_event(_final("Réponse.", "B", 3))
    items = _items(tab)
    assert items[0].speaker_label is not None  # nouveau groupe : en-tête
    assert items[1].speaker_label is None      # même locuteur : pas d'en-tête
    assert items[2].speaker_label is not None  # locuteur différent : en-tête


def test_timestamp_shown_once_per_minute(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Un.", "A", 1))
    tab.on_event(_final("Deux.", "B", 2))  # nouveau groupe, même minute
    items = _items(tab)
    assert items[0].ts_label.text() != ""
    assert items[1].ts_label.text() == ""


def test_correction_replaces_line_without_duplicate(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Texte brut avant correction.", "A", 7))
    tab.on_event(_final("Texte corrigé.", "A", 7, corrected=True))
    items = _items(tab)
    assert len(items) == 1
    assert "Texte corrigé." in items[0].text_label.text()


def test_stale_correction_is_ignored(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Phrase affichée.", "A", 1))
    tab.on_event(_final("Correction orpheline.", "A", 999, corrected=True))
    items = _items(tab)
    assert len(items) == 1
    assert "Phrase affichée." in items[0].text_label.text()


def test_vad_animates_empty_state_wave(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event({"type": "vad_status", "speaking": True})
    assert tab.empty.wave._active
    tab.on_event({"type": "vad_status", "speaking": False})
    assert not tab.empty.wave._active


def test_partial_line_shows_and_clears(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event({"type": "segment_start"})
    tab.on_event({"type": "word", "text": "bonjour"})
    tab.on_event({"type": "word", "text": "monde"})
    assert tab.partial.isVisible()
    assert "bonjour monde" in tab.partial.text_label.text()
    assert tab.partial.wave._active
    tab.on_event(_final("Bonjour monde.", "A", 1))
    assert not tab.partial.isVisible()


def test_items_are_capped_so_long_meetings_do_not_grow_forever(qtbot, monkeypatch):
    """Au-delà du plafond, les lignes les plus anciennes sont retirées.

    Sans ça, une réunion longue laisse un QWidget par phrase en vie : la mémoire
    grimpe et chaque nouvelle ligne relayoute une pile de plus en plus lourde.
    """
    import benji.ui.live_tab as live_tab_mod

    monkeypatch.setattr(live_tab_mod, "_MAX_ITEMS", 5)
    tab = LiveTab()
    qtbot.addWidget(tab)

    for i in range(12):
        tab.on_event(_final(f"Phrase {i}.", "A", i))

    items = _items(tab)
    assert len(items) == 5
    # Ce sont bien les plus récentes qui restent.
    assert items[-1]._text == "Phrase 11."
    assert items[0]._text == "Phrase 7."


def test_trimmed_items_cannot_receive_a_late_correction(qtbot, monkeypatch):
    """Une correction tardive sur une ligne retirée ne touche pas un objet mort."""
    import benji.ui.live_tab as live_tab_mod

    monkeypatch.setattr(live_tab_mod, "_MAX_ITEMS", 2)
    monkeypatch.setattr(live_tab_mod, "_MAX_CORRECTABLE", 10)
    tab = LiveTab()
    qtbot.addWidget(tab)

    for i in range(5):
        tab.on_event(_final(f"Phrase {i}.", "A", i))

    assert all(c in _items(tab) for c in tab._correctable)
    tab.on_event(_final("Corrigée.", "A", 0, corrected=True))  # seq retiré : sans effet
    assert [i._text for i in _items(tab)] == ["Phrase 3.", "Phrase 4."]


def test_le_transcript_est_ancre_en_bas(qtbot):
    """Le ressort est en tête : la ligne en cours colle à la dernière phrase.

    Ancré en haut, un début de réunion laissait la ligne vivante flotter à des
    centaines de pixels sous le texte, au bas d'une fenêtre vide.
    """
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Bonjour.", "A", 1))

    first = tab.content_layout.itemAt(0)
    assert first.widget() is None  # un ressort, pas une ligne
    assert first.expandingDirections() != 0


def _fill_past_the_fold(tab, qtbot, count=30):
    """Assez de lignes pour dépasser la hauteur visible."""
    tab.resize(800, 400)
    tab.show()
    qtbot.waitExposed(tab)
    for i in range(count):
        tab.on_event(_final(f"Ligne {i} " + "mot " * 25, "A", i))
    qtbot.wait(50)


def test_le_transcript_defile_quand_il_depasse_la_fenetre(qtbot):
    """Le bas reste collé : la dernière phrase ne passe jamais sous le pli.

    Un scroll déclenché à l'ajout du widget visait un maximum périmé — la ligne
    venait d'être insérée mais n'avait pas encore sa hauteur (retour à la ligne
    calculé au layout) — et l'affichage restait une ligne en arrière.
    """
    tab = LiveTab()
    qtbot.addWidget(tab)
    _fill_past_the_fold(tab, qtbot)

    sb = tab.scroll.verticalScrollBar()
    assert sb.maximum() > 0  # le contenu dépasse bien la fenêtre
    assert sb.value() == sb.maximum()


def test_une_correction_qui_rallonge_la_ligne_garde_le_bas_colle(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    _fill_past_the_fold(tab, qtbot)

    tab.on_event(_final("Corrigé " + "mot " * 80, "A", 29, corrected=True))
    qtbot.wait(50)
    sb = tab.scroll.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_remonter_dans_le_transcript_suspend_le_defilement(qtbot):
    """Relire une phrase plus haut ne doit pas être interrompu par la suite."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    _fill_past_the_fold(tab, qtbot)

    sb = tab.scroll.verticalScrollBar()
    sb.setValue(0)
    assert tab._user_scrolled_up

    tab.on_event(_final("Nouvelle phrase pendant la lecture.", "A", 100))
    qtbot.wait(50)
    assert sb.value() == 0

    # Redescendre au bas réarme le suivi.
    sb.setValue(sb.maximum())
    tab.on_event(_final("Encore une phrase.", "A", 101))
    qtbot.wait(50)
    assert sb.value() == sb.maximum()


def test_le_bouton_de_retour_au_direct_napparait_que_decroche(qtbot):
    """Le défilement automatique laissait sans issue : rien ne disait qu'on ne
    suivait plus, et il fallait redescendre à la main."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    _fill_past_the_fold(tab, qtbot)
    assert not tab.jump_btn.isVisible()

    sb = tab.scroll.verticalScrollBar()
    sb.setValue(0)
    assert tab.jump_btn.isVisible()

    tab.jump_btn.click()
    qtbot.wait(50)
    assert sb.value() == sb.maximum()
    assert not tab.jump_btn.isVisible()


def test_la_recherche_filtre_le_transcript_en_cours(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event(_final("On parle du budget client.", "Alice", 1))
    tab.on_event(_final("Rien à voir avec ce sujet.", "Bob", 2))

    tab.toggle_search()
    tab.search_field.setText("budget")

    visibles = [i for i in _items(tab) if i.isVisible()]
    assert [i._text for i in visibles] == ["On parle du budget client."]
    assert "1 sur 2" in tab.search_count.text()

    # Fermer rend le transcript entier — jamais un filtre orphelin.
    tab.close_search()
    assert all(i.isVisible() for i in _items(tab))


def test_la_recherche_trouve_par_locuteur(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event(_final("Le point sur le budget.", "Marie", 1))
    tab.on_event(_final("Le point sur le budget.", "Paul", 2))

    tab.toggle_search()
    tab.search_field.setText("marie budget")

    assert [i._speaker for i in _items(tab) if i.isVisible()] == ["Marie"]


def test_une_phrase_qui_arrive_pendant_une_recherche_est_filtree(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event(_final("Budget valide.", "Alice", 1))
    tab.toggle_search()
    tab.search_field.setText("budget")

    tab.on_event(_final("Autre chose entierement.", "Bob", 2))

    assert [i._text for i in _items(tab) if i.isVisible()] == ["Budget valide."]


def test_chercher_ne_ramene_pas_au_bas_a_la_phrase_suivante(qtbot):
    """Chercher, c'est relire : le suivi du direct doit lâcher prise."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    _fill_past_the_fold(tab, qtbot)
    tab.toggle_search()
    tab.search_field.setText("Ligne 3")

    assert tab._user_scrolled_up


def test_copier_le_transcript_rend_un_compte_rendu_lisible(qtbot):
    """Le transcript est fait de QLabel indépendants : une sélection ne franchit
    pas la ligne, donc copier tout est le seul geste qui rende le direct
    exploitable sans quitter la réunion."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event(_final("Premier point.", "Alice", 1))
    tab.on_event(_final("Deuxieme point.", "Bob", 2))

    lignes = tab.transcript_text().splitlines()
    assert len(lignes) == 2
    assert lignes[0].endswith("Alice : Premier point.")
    assert lignes[0].startswith("[") and ":" in lignes[0]


def test_la_copie_se_limite_aux_resultats_de_la_recherche(qtbot):
    """Les actions portent sur ce qui est à l'écran, comme dans Réunions."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event(_final("On parle du budget.", "Alice", 1))
    tab.on_event(_final("Rien a voir.", "Bob", 2))
    tab.toggle_search()
    tab.search_field.setText("budget")

    assert tab.transcript_text().count("\n") == 0
    assert "budget" in tab.transcript_text()


def test_le_bandeau_dit_que_rien_nest_conserve(qtbot):
    """Un utilisateur qui croit enregistrer alors que non est le pire état."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    assert not tab.consent.isVisible()  # pas de portillon = pas de bandeau

    tab.set_consent_pending(True)
    assert tab.consent.isVisible()
    assert "rien n'est encore conservé" in tab.consent.label.text()

    demandes = []
    tab.save_requested.connect(lambda: demandes.append(1))
    tab.consent.button.click()
    assert demandes == [1]


def test_le_bandeau_ne_prend_pas_le_rouge_du_direct(qtbot):
    """Le rouge ne signifie qu'« on prend au mot, maintenant » — or c'est
    précisément ce qui n'a pas lieu tant que rien n'est conservé."""
    from benji.ui.style import current_theme

    tab = LiveTab()
    qtbot.addWidget(tab)
    rouge = current_theme().record.name().lower()

    assert rouge not in tab.consent.button.styleSheet().lower()
    assert rouge not in tab.consent.label.styleSheet().lower()


def test_marquer_designe_la_derniere_chose_dite(qtbot, monkeypatch):
    """On réagit à ce qu'on vient d'entendre, jamais à ce qui va se dire."""
    import benji.ui.live_tab as live_tab_mod

    marques = []
    monkeypatch.setattr(live_tab_mod.meetings, "add_mark", lambda: marques.append(1))
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Une phrase.", "A", 1))
    tab.on_event(_final("Le chiffre important.", "A", 2))

    assert tab.mark_moment() is True

    items = _items(tab)
    assert [i.marked for i in items] == [False, True]
    assert marques == [1]


def test_marquer_sans_rien_de_transcrit_ne_fait_rien(qtbot, monkeypatch):
    import benji.ui.live_tab as live_tab_mod

    monkeypatch.setattr(live_tab_mod.meetings, "add_mark",
                        lambda: pytest.fail("aucune marque attendue"))
    tab = LiveTab()
    qtbot.addWidget(tab)

    assert tab.mark_moment() is False


def test_un_registre_indisponible_laisse_la_marque_a_lecran(qtbot, monkeypatch):
    """Marquer est un confort : ça n'interrompt pas une réunion."""
    import benji.ui.live_tab as live_tab_mod

    def _boom():
        raise OSError("disque plein")

    monkeypatch.setattr(live_tab_mod.meetings, "add_mark", _boom)
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Une phrase.", "A", 1))

    assert tab.mark_moment() is True
    assert _items(tab)[-1].marked


def test_nommer_un_locuteur_change_laffichage_pas_letiquette(qtbot):
    """L'étiquette porte la couleur : la changer ferait sauter la teinte du
    locuteur au moment précis où on le nomme."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Bonjour.", "SPEAKER_01", 1))

    tab.set_speaker_name("SPEAKER_01", "Alice")

    item = _items(tab)[0]
    assert item._speaker == "SPEAKER_01"
    assert item.speaker_label.text() == "Alice"


def test_le_nom_vaut_aussi_pour_les_phrases_suivantes(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Bonjour.", "SPEAKER_01", 1))
    tab.set_speaker_name("SPEAKER_01", "Alice")

    tab.on_event(_final("Autre chose.", "SPEAKER_02", 2))
    tab.on_event(_final("Je reprends.", "SPEAKER_01", 3))

    assert _items(tab)[-1].speaker_label.text() == "Alice"


def test_apprendre_un_terme_corrige_la_ligne_deja_affichee(qtbot, monkeypatch, tmp_path):
    """Sans ça, le terme appris ne corrigerait que la suite — et la faute qu'on
    regardait resterait à l'écran."""
    from benji.stt import lexicon

    glossaire = tmp_path / "glossary.txt"
    monkeypatch.setattr(lexicon, "glossary_path", lambda: glossaire)
    monkeypatch.setattr(lexicon, "load_terms", lambda path=None: ["Datadog"])

    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.set_learn_handler(lambda term: True)
    tab.on_event(_final("On surveille avec data dogue.", "A", 1))

    tab._actions._reapply_lexicon()

    assert "Datadog" in _items(tab)[0]._text


def test_la_copie_utilise_le_nom_donne_pas_letiquette(qtbot):
    """Un compte rendu qui part à quelqu'un ne dit pas « SPEAKER_01 »."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.show()
    tab.on_event(_final("Bonjour.", "SPEAKER_01", 1))
    tab.set_speaker_name("SPEAKER_01", "Alice")

    assert "Alice : Bonjour." in tab.transcript_text()
    assert "SPEAKER_01" not in tab.transcript_text()


def test_nommer_un_locuteur_previent_lexterieur(qtbot):
    """L'overlay s'abonne à ce signal : sans lui, les sous-titres garderaient « A »."""
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Bonjour.", "A", 1))

    with qtbot.waitSignal(tab.speaker_named) as blocker:
        tab.set_speaker_name("A", "Alice")
    assert blocker.args == ["A", "Alice"]


def test_cliquer_sur_len_tete_propose_de_nommer(qtbot, monkeypatch):
    """Nommer n'était accessible que par un clic droit que rien ne laissait deviner."""
    from PySide6.QtCore import Qt

    from benji.ui import live_tab as live_tab_mod

    monkeypatch.setattr(live_tab_mod.QInputDialog, "getText",
                        staticmethod(lambda *a, **kw: ("Alice", True)))
    monkeypatch.setattr(live_tab_mod.meetings, "name_speaker", lambda *a, **kw: None)
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Bonjour.", "A", 1))
    item = _items(tab)[0]

    qtbot.mouseClick(item.speaker_label, Qt.MouseButton.LeftButton)

    assert item.speaker_label.text() == "Alice"


def test_une_nouvelle_reunion_oublie_les_noms(qtbot):
    tab = LiveTab()
    qtbot.addWidget(tab)
    tab.on_event(_final("Bonjour.", "A", 1))
    tab.set_speaker_name("A", "Alice")

    tab.clear_speaker_names()
    tab.on_event(_final("Autre chose.", "B", 2))
    tab.on_event(_final("Moi.", "A", 3))

    assert _items(tab)[0].speaker_label.text() == "Alice"  # passé inchangé
    assert _items(tab)[-1].speaker_label.text() == "A"     # « A » n'est plus Alice
