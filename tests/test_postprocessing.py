from benji.stt.postprocessing import (
    NBSP,
    NNBSP,
    french_typography,
    is_hallucination,
    postprocess_text,
)


def test_empty():
    assert postprocess_text("") == ""
    assert postprocess_text("   ").strip() == ""


def test_capitalize_first():
    assert postprocess_text("bonjour") == "Bonjour"


def test_capitalize_after_period():
    assert postprocess_text("bonjour. ça va") == "Bonjour. Ça va"


def test_removes_french_hesitations():
    out = postprocess_text("euh je pense que, heu, oui")
    assert "euh" not in out.lower()
    assert "heu" not in out.lower()


def test_fix_french_apostrophe():
    assert postprocess_text("qu ' on parle") == "Qu'on parle"


def test_fix_hyphen():
    assert postprocess_text("est - ce que") == "Est-ce que"


def test_spacing_around_punctuation():
    out = postprocess_text("salut ,comment ça va ?")
    assert " ," not in out
    assert "?" in out


def test_english_contractions():
    out = postprocess_text("im sure i cant", language="en")
    assert "I'm" in out
    assert "can't" in out.lower() or "Can't" in out


def test_multiple_spaces_collapsed():
    assert "  " not in postprocess_text("hello    world")


def test_decimal_numbers_preserved():
    # Pas d'espace inséré entre deux chiffres (virgule ou point décimal).
    assert "2,5" in postprocess_text("il fait 2,5 degrés")
    assert "3.14" in postprocess_text("pi vaut 3.14 environ")


def test_eh_bien_preserved():
    # « eh », « ah », « oh » sont des mots légitimes en français.
    assert postprocess_text("eh bien voilà") == "Eh bien voilà"
    assert postprocess_text("ah bon, d'accord") == "Ah bon, d'accord"


def test_leading_hesitation_leaves_no_orphan_punctuation():
    # « Euh, oui » → suppression de l'hésitation → pas de « , oui » résiduel.
    assert postprocess_text("euh, oui") == "Oui"


def test_fix_apostrophe_and_hyphen_with_accents():
    assert postprocess_text("l ' été") == "L'été"
    assert postprocess_text("peut - être") == "Peut-être"


def test_compound_punctuation_stays_together():
    assert postprocess_text("quoi ?! bon") == "Quoi?! Bon"


# --- typographie française --------------------------------------------------


def test_french_spaces_before_high_punctuation():
    out = postprocess_text("vraiment ? on dit : oui ; non !", language="fr")
    assert out == f"Vraiment{NNBSP}? On dit{NBSP}: oui{NNBSP}; non{NNBSP}!"


def test_french_guillemets():
    assert postprocess_text("il a dit «bien»", language="fr") == f"Il a dit «{NBSP}bien{NBSP}»"


def test_french_typography_spares_times_and_compound_punctuation():
    out = postprocess_text("rendez-vous à 14:30 : ok ?!", language="fr")
    assert out == f"Rendez-vous à 14:30{NBSP}: ok{NNBSP}?!"


def test_french_typography_is_idempotent():
    once = french_typography("Vraiment ? « oui » : non ; bon !")
    assert french_typography(once) == once


def test_english_keeps_english_typography():
    assert postprocess_text("really ? ok", language="en") == "Really? Ok"


# --- filtre d'hallucinations -------------------------------------------------


def test_meeting_formulas_inside_a_real_sentence_are_kept():
    # Une formule de fin de vidéo au milieu d'une vraie phrase de réunion : le
    # tour entier partait à la poubelle.
    assert not is_hallucination("Merci à tous d'être venus, on commence par le budget.")
    assert not is_hallucination("Bon, à la prochaine réunion on tranche.")
    assert not is_hallucination("Merci de votre attention, des questions ?")


def test_formula_alone_is_a_hallucination():
    assert is_hallucination("Merci à tous.")
    assert is_hallucination("Merci d'avoir regardé et à la prochaine !")


def test_subtitle_credits_are_always_hallucinations():
    assert is_hallucination("Sous-titres réalisés par la société Radio-Canada")
    assert is_hallucination("Traduction amara.org")


def test_human_repetition_is_kept_engine_loop_is_not():
    assert not is_hallucination("non non non non je ne suis pas d'accord")
    assert is_hallucination("oui oui oui oui oui oui oui")
