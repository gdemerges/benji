import numpy as np

from benji.stt.diarization import (
    SpeakerTagger,
    _estimate_f0,
    label_windows,
    split_by_speaker,
)


def _sine(freq: float, duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_estimate_f0_on_sine():
    audio = _sine(150.0)
    est = _estimate_f0(audio)
    assert est is not None
    assert abs(est - 150.0) < 10


def test_estimate_f0_on_silence():
    audio = np.zeros(16000, dtype=np.float32)
    assert _estimate_f0(audio) is None


def test_speaker_tagger_two_voices():
    tagger = SpeakerTagger(f0_gap_hz=30.0)
    low = _sine(120.0)
    high = _sine(220.0)

    assert tagger.label(low) == "A"
    assert tagger.label(low) == "A"
    assert tagger.label(high) == "B"
    assert tagger.label(low) == "A"


def test_speaker_tagger_same_voice_stable():
    tagger = SpeakerTagger(f0_gap_hz=40.0)
    a1 = _sine(150.0)
    a2 = _sine(160.0)  # slight drift, same speaker
    assert tagger.label(a1) == "A"
    assert tagger.label(a2) == "A"


# --- Découpe en tours de parole ----------------------------------------------


class _ScriptedTagger:
    """Rend une étiquette par appel, dans l'ordre. Enregistre les durées reçues."""

    def __init__(self, labels):
        self._labels = list(labels)
        self.durations: list[float] = []

    def label(self, audio, sample_rate=16000):
        self.durations.append(len(audio) / sample_rate)
        return self._labels.pop(0) if self._labels else None


def _words(*spec):
    return [{"text": t, "start": s, "end": e} for t, s, e in spec]


def test_label_windows_short_buffer_stays_a_single_span():
    """Sous la taille d'une fenêtre : un seul appel, le comportement d'avant."""
    tagger = _ScriptedTagger(["A"])
    spans = label_windows(tagger, np.zeros(16000, dtype=np.float32), 16000,
                          window_s=1.5, hop_s=0.75)
    assert spans == [(0.0, 1.0, "A")]
    assert tagger.durations == [1.0]


def test_label_windows_slides_over_a_long_buffer():
    tagger = _ScriptedTagger(["A", "A", "B"])
    audio = np.zeros(int(3.0 * 16000), dtype=np.float32)
    spans = label_windows(tagger, audio, 16000, window_s=1.5, hop_s=0.75)

    assert [lbl for _, _, lbl in spans] == ["A", "A", "B"]
    assert spans[0][:2] == (0.0, 1.5)
    assert spans[-1][1] == 3.0  # la fin du tampon est couverte
    # Chaque fenêtre est soumise séparément, jamais le tampon entier.
    assert all(d <= 1.5 for d in tagger.durations)


def test_split_by_speaker_cuts_a_segment_holding_two_turns():
    """Le cas qui motive tout : deux personnes dans un seul segment VAD."""
    words = _words(
        ("je", 0.0, 0.3), ("pense", 0.3, 0.8), ("que", 0.8, 1.1), ("oui", 1.1, 1.4),
        ("moi", 1.6, 1.9), ("non", 1.9, 2.2), ("pas", 2.2, 2.5), ("du", 2.5, 2.7),
        ("tout", 2.7, 3.0),
    )
    spans = [(0.0, 1.5, "A"), (1.5, 3.0, "B")]

    turns = split_by_speaker(words, spans)

    assert [lbl for lbl, _ in turns] == ["A", "B"]
    assert [w["text"] for w in turns[0][1]] == ["je", "pense", "que", "oui"]
    assert [w["text"] for w in turns[1][1]] == ["moi", "non", "pas", "du", "tout"]


def test_split_by_speaker_keeps_one_turn_when_a_single_voice_speaks():
    words = _words(("bonjour", 0.0, 0.5), ("le", 0.5, 0.7), ("monde", 0.7, 1.2))
    turns = split_by_speaker(words, [(0.0, 1.0, "A"), (0.75, 1.2, "A")])
    assert turns == [("A", words)]


def test_split_by_speaker_without_spans_returns_one_unlabeled_turn():
    """Diarisation absente ou en échec : le chemin d'avant, sans locuteur."""
    words = _words(("bonjour", 0.0, 0.5))
    assert split_by_speaker(words, []) == [(None, words)]


def test_split_by_speaker_ignores_unlabeled_windows():
    """Une fenêtre sans étiquette n'ouvre pas un tour anonyme au milieu."""
    words = _words(("un", 0.0, 0.4), ("deux", 0.4, 0.8),
                   ("trois", 1.6, 2.0), ("quatre", 2.0, 2.4))
    turns = split_by_speaker(words, [(0.0, 1.5, "A"), (1.5, 2.5, None)])
    assert turns == [("A", words)]


def test_split_by_speaker_absorbs_a_one_word_flicker():
    """Une fenêtre isolée qui change d'avis ne doit pas faire du confetti."""
    words = _words(
        ("il", 0.0, 0.3), ("faut", 0.3, 0.6), ("qu'on", 0.6, 0.9),
        ("avance", 1.6, 2.2),  # fenêtre B parasite, un seul mot
        ("sur", 2.6, 2.9), ("le", 2.9, 3.1), ("sujet", 3.1, 3.5),
    )
    spans = [(0.0, 1.5, "A"), (1.5, 2.5, "B"), (2.5, 3.5, "A")]

    turns = split_by_speaker(words, spans, min_turn_words=2)

    assert len(turns) == 1
    assert turns[0][0] == "A"
    assert [w["text"] for w in turns[0][1]] == [w["text"] for w in words]


def test_split_by_speaker_words_without_timestamps_inherit_the_previous_turn():
    words = [
        {"text": "oui", "start": 0.0, "end": 0.4},
        {"text": "voilà", "start": 0.4, "end": 0.9},
        {"text": ".", "start": None, "end": None},
        {"text": "moi", "start": 2.0, "end": 2.3},
        {"text": "je", "start": 2.3, "end": 2.5},
        {"text": "dirais", "start": 2.5, "end": 3.0},
    ]
    turns = split_by_speaker(words, [(0.0, 1.5, "A"), (1.5, 3.0, "B")])

    assert [lbl for lbl, _ in turns] == ["A", "B"]
    assert [w["text"] for w in turns[0][1]] == ["oui", "voilà", "."]


def test_split_by_speaker_snaps_the_boundary_to_the_pause():
    """Les fenêtres se recouvrent : la frontière est floue à un demi-pas près.

    Le mot « moi » tombe dans la fenêtre A alors qu'il ouvre le tour de B ; le
    blanc de 300 ms qui le précède tranche mieux que les embeddings.
    """
    words = _words(
        ("que", 0.8, 1.1), ("oui", 1.1, 1.4),
        ("moi", 1.7, 2.0), ("non", 2.0, 2.3), ("pas", 2.3, 2.6),
    )
    spans = [(0.0, 1.5, "A"), (0.75, 2.25, "A"), (1.5, 3.0, "B")]

    turns = split_by_speaker(words, spans)

    assert [w["text"] for w in turns[0][1]] == ["que", "oui"]
    assert [w["text"] for w in turns[1][1]] == ["moi", "non", "pas"]


def test_split_by_speaker_leaves_the_boundary_alone_without_a_real_pause():
    """Sans blanc à proximité, on ne sait rien de mieux que les embeddings.

    Débit continu, aucun silence : la frontière reste là où les fenêtres l'ont
    mise. La déplacer vers un écart nul reviendrait à décider au hasard.
    """
    words = _words(
        ("il", 0.0, 0.5), ("faut", 0.5, 1.0), ("avancer", 1.0, 1.6),
        ("non", 1.6, 2.1), ("attends", 2.1, 2.6),
    )
    spans = [(0.0, 1.5, "A"), (1.5, 3.0, "B")]

    turns = split_by_speaker(words, spans)

    assert [w["text"] for w in turns[0][1]] == ["il", "faut", "avancer"]
    assert [w["text"] for w in turns[1][1]] == ["non", "attends"]


def test_split_by_speaker_snaps_forward_at_the_end_of_the_segment():
    """Frontière avancée vers une pause en fin de segment : elle est déjà placée.

    La boucle la retrouvait à sa nouvelle position et cherchait un meilleur blanc
    dans une fenêtre vide — `max()` levait et le segment final était perdu.
    """
    words = _words(("oui", 0.0, 0.3), ("voilà", 0.32, 0.6), ("moi", 1.0, 1.3))
    spans = [(0.0, 0.31, "A"), (0.31, 1.3, "B")]

    turns = split_by_speaker(words, spans, min_turn_words=1)

    assert [lbl for lbl, _ in turns] == ["A", "B"]
    assert [w["text"] for w in turns[0][1]] == ["oui", "voilà"]
    assert [w["text"] for w in turns[1][1]] == ["moi"]
