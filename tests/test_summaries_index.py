"""Résumés rattachés à leurs réunions : effacer une réunion emporte ses résumés."""

from benji import meetings
from benji.llm import summarizer


def _entries(meeting_id, n=1):
    return [{"text": f"Phrase {i}.", "meeting": meeting_id} for i in range(n)]


def test_effacer_une_reunion_efface_ses_resumes():
    kept = summarizer.save_summary("Autre réunion.", _entries("m-garde"))
    doomed = summarizer.save_summary("Réunion effacée.", _entries("m-efface"))

    assert summarizer.delete_summaries("m-efface") == 1

    assert not doomed.exists()
    assert kept.exists()


def test_un_resume_a_cheval_part_avec_la_reunion_effacee():
    """Il cite des phrases de la réunion effacée : le garder, c'est les garder."""
    both = summarizer.save_summary("Deux réunions.", _entries("m-a") + _entries("m-b"))

    summarizer.delete_summaries("m-b")

    assert not both.exists()


def test_entrees_heritees_rattachees_a_legacy():
    path = summarizer.save_summary("Ancien.", [{"text": "Sans réunion."}])

    assert summarizer.delete_summaries(meetings.LEGACY_ID) == 1
    assert not path.exists()


def test_resume_sans_index_n_est_pas_devine():
    """Un résumé d'avant l'index n'est rattaché à rien : on ne le devine pas."""
    path = summarizer.save_summary("Sans entrées.")

    assert summarizer.delete_summaries("m-quelconque") == 0
    assert path.exists()


def test_l_index_n_est_pas_un_resume():
    """L'onglet Résumés reconnaît les fichiers `summary_*.md` : l'index n'en est pas."""
    path = summarizer.save_summary("Résumé.", _entries("m1"))
    names = [p.name for p in path.parent.glob("summary_*.md")]
    assert names == [path.name]
