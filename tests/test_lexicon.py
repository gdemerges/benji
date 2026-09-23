"""Glossaire utilisateur appliqué au texte final.

Deux exigences opposées : rattraper un nom propre que le moteur a massacré, et
ne **jamais** toucher à une phrase où le terme n'a rien à faire. Le second cas
est le plus important — une substitution abusive fabrique un faux dans un
compte rendu de réunion.
"""

import pytest

from benji.stt import lexicon
from benji.stt.lexicon import (
    apply_lexicon,
    compile_terms,
    keys_match,
    load_terms,
    parse_glossary,
    phonetic_key,
)

# --- clé phonétique ---

def test_orthographes_proches_partagent_la_cle():
    assert phonetic_key("Kubernetes") == phonetic_key("kubernétesse")
    assert phonetic_key("Sentry") == phonetic_key("centrie")
    assert phonetic_key("Agricole") == phonetic_key("agricol")


def test_mots_distincts_ne_partagent_pas_la_cle():
    assert phonetic_key("Datadog") != phonetic_key("Kubernetes")
    assert phonetic_key("réunion") != phonetic_key("décision")


def test_cle_vide_pour_un_fragment_sans_lettre():
    assert phonetic_key("42") == ""
    assert phonetic_key("") == ""


def test_keys_match_exige_l_egalite_sur_les_cles_courtes():
    assert keys_match("abc", "abc") is True
    assert keys_match("abc", "abd") is False


# --- lecture du glossaire ---

def test_parse_glossary_ignore_commentaires_et_doublons():
    raw = "\n".join([
        "Kubernetes",
        "# un commentaire",
        "",
        "   Datadog   # inline",
        "kubernetes",       # doublon insensible à la casse
        "ok",               # trop court
        "un terme beaucoup trop long pour être apparié",
    ])
    assert parse_glossary(raw) == ["Kubernetes", "Datadog"]


def test_load_terms_sans_fichier(tmp_path):
    assert load_terms(tmp_path / "absent.txt") == []


def test_load_terms_lit_le_fichier(tmp_path):
    path = tmp_path / lexicon.GLOSSARY_NAME
    path.write_text("Datadog\nCrédit Agricole\n", encoding="utf-8")
    assert load_terms(path) == ["Datadog", "Crédit Agricole"]


def test_load_terms_ne_loggue_jamais_le_contenu(tmp_path, caplog):
    """Un glossaire liste des clients et des projets : le fichier de log est
    persisté et joint aux rapports de bug (cf. CLAUDE.md)."""
    path = tmp_path / lexicon.GLOSSARY_NAME
    path.write_text("Vercingétorix SAS\n", encoding="utf-8")
    with caplog.at_level("DEBUG"):
        load_terms(path)
    assert "Vercingétorix" not in caplog.text


# --- application ---

def test_terme_massacre_est_rattrape():
    compiled = compile_terms(["Kubernetes"])
    assert apply_lexicon("On déploie sur kubernétesse demain.", compiled) == (
        "On déploie sur Kubernetes demain."
    )


def test_terme_decoupe_en_deux_mots_est_recolle():
    compiled = compile_terms(["Datadog"])
    assert apply_lexicon("On regarde data dogue ce soir.", compiled) == (
        "On regarde Datadog ce soir."
    )


def test_terme_multi_mots_recasse():
    compiled = compile_terms(["Crédit Agricole"])
    assert apply_lexicon("Le crédit agricole a validé.", compiled) == (
        "Le Crédit Agricole a validé."
    )


def test_ne_traverse_pas_la_ponctuation():
    """« Le crédit, agricole ou pas » est une énumération — pas la banque."""
    compiled = compile_terms(["Crédit Agricole"])
    text = "Le crédit, agricole ou pas, est validé."
    assert apply_lexicon(text, compiled) == text


def test_phrase_sans_rapport_est_intacte():
    compiled = compile_terms(["Kubernetes", "Datadog"])
    text = "Rien à voir avec le sujet du jour."
    assert apply_lexicon(text, compiled) == text


def test_glossaire_vide_rend_le_texte_tel_quel():
    text = "Le texte doit ressortir à l'identique."
    assert apply_lexicon(text, []) == text
    assert apply_lexicon(text, compile_terms([])) == text


def test_ponctuation_et_espaces_preserves():
    compiled = compile_terms(["Datadog"])
    assert apply_lexicon("Alors — data dogue, non ?", compiled) == "Alors — Datadog, non ?"


def test_terme_deja_correct_nest_pas_touche():
    compiled = compile_terms(["Kubernetes"])
    assert apply_lexicon("Kubernetes tourne.", compiled) == "Kubernetes tourne."


def test_termes_longs_essayes_avant_les_courts():
    """Sans cet ordre, « Crédit » consommerait le premier mot et « Agricole »
    resterait faux."""
    compiled = compile_terms(["Crédit", "Crédit Agricole"])
    assert compiled[0][0] == "Crédit Agricole"
    assert apply_lexicon("le crédit agricol appelle", compiled) == (
        "le Crédit Agricole appelle"
    )


def test_texte_vide():
    assert apply_lexicon("", compile_terms(["Datadog"])) == ""


# --- glossaire nourri depuis le transcript ---


def test_un_terme_appris_est_persiste(tmp_path):
    """Le glossaire est le seul levier sur les noms propres, mais il fallait
    aller le saisir dans les Préférences — au moment où on n'y pense pas."""
    path = tmp_path / "glossary.txt"
    path.write_text("Datadog\n", encoding="utf-8")

    assert lexicon.add_term("Kubernetes", path) is True
    assert lexicon.load_terms(path) == ["Datadog", "Kubernetes"]


def test_un_terme_deja_connu_nest_pas_ajoute_deux_fois(tmp_path):
    path = tmp_path / "glossary.txt"
    path.write_text("Datadog\n", encoding="utf-8")

    assert lexicon.add_term("datadog", path) is False
    assert lexicon.load_terms(path) == ["Datadog"]


def test_un_terme_vide_est_refuse(tmp_path):
    path = tmp_path / "glossary.txt"

    assert lexicon.add_term("   ", path) is False
    assert not path.exists()


def test_le_terme_appris_ne_fuite_pas_dans_les_logs(tmp_path, caplog):
    """Il nomme un client ou un projet — et le log part dans les rapports."""
    import logging

    path = tmp_path / "glossary.txt"
    with caplog.at_level(logging.DEBUG):
        lexicon.add_term("Groupe Bertrand", path)

    assert "Bertrand" not in caplog.text


@pytest.mark.posix_perms
def test_un_glossaire_appris_est_ecrit_en_0600(tmp_path):
    path = tmp_path / "glossary.txt"
    lexicon.add_term("Datadog", path)

    assert (path.stat().st_mode & 0o777) == 0o600
