"""Streaming incrémental du Transcriber (LocalAgreement-2).

À chaque passe partielle le tampon est re-décodé **en entier**, et le préfixe sur
lequel deux passes successives s'accordent est figé : il ne bougera plus à
l'écran. Ces tests pilotent cette machine à états avec un faux backend scripté —
aucun modèle chargé, aucun audio décodé.
"""

from queue import Queue

import numpy as np
import pytest

import benji.stt.transcriber as transcriber_mod
from benji.config import STTConfig
from benji.recording import RecordingConsent
from benji.stt.transcriber import Transcriber

SR = 16000


class FakeBackend:
    """Rend une liste de mots scriptée à chaque appel de transcribe().

    Chaque mot est ``(texte, début, fin)`` en secondes depuis le début du tampon.
    L'audio est ignoré : ces tests exercent la comptabilité de l'accord, pas le
    décodage. La longueur de chaque tampon reçu est enregistrée dans ``calls``.
    """

    name = "fake"

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls: list[int] = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        words = self._scripts.pop(0) if self._scripts else []
        for text, start, end in words:
            yield {"text": text, "start": start, "end": end}


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _make(monkeypatch, scripts, **cfg) -> tuple[Transcriber, FakeBackend]:
    backend = FakeBackend(scripts)
    monkeypatch.setattr(transcriber_mod, "build_backend", lambda *a, **kw: backend)
    # Sans ça, le Transcriber construit un vrai moteur final et va télécharger
    # Whisper. None = « réutilise le moteur des partielles ».
    monkeypatch.setattr(transcriber_mod, "build_final_backend", lambda *a, **kw: None)
    cfg.setdefault("diarization", False)  # ces tests pilotent le tagger explicitement
    # Ces tests portent sur le décodage, pas sur l'accord de conservation : on
    # ouvre le portillon d'office (cf. benji/recording.py, testé à part).
    cfg.setdefault("confirm_before_saving", False)
    t = Transcriber(Queue(), Queue(), STTConfig(**cfg), stats=None, sample_rate=SR)
    return t, backend


def _partial_words(q: Queue) -> list[str]:
    """Mots du dernier instantané partiel publié (un message par passe)."""
    snapshots = [e for e in _drain(q) if e.get("type") == "partial"]
    return [w["text"] for w in snapshots[-1]["words"]] if snapshots else []


def _drain(q: Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_first_partial_commits_nothing(monkeypatch):
    # Sans passe précédente pour corroborer, rien n'est figé : tous les mots
    # restent des hypothèses que la passe suivante devra confirmer.
    t, backend = _make(monkeypatch, [[("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.0)]])

    t._run_partial(_audio(1.0))

    assert t._committed_words == []
    assert t._prev_words_norm == ["bonjour", "le", "monde"]
    assert backend.calls == [SR]


def test_second_partial_commits_agreed_prefix(monkeypatch):
    # Deux passes d'accord sur « bonjour le monde » → ce préfixe est figé.
    t, backend = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.0)],
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.0), ("est", 1.0, 1.4)],
    ])

    t._run_partial(_audio(1.0))
    t._run_partial(_audio(1.4))

    assert [w["text"] for w in t._committed_words] == ["bonjour", "le", "monde"]
    assert t._prev_words_norm == ["bonjour", "le", "monde", "est"]
    # Le tampon entier est redonné au moteur à chaque passe : plus de tranches.
    assert backend.calls == [SR, int(1.4 * SR)]


def test_les_horodatages_restent_absolus(monkeypatch):
    """Décoder le tampon entier rend les horodatages directement exploitables.

    L'ancien découpage en tranches obligeait à les recaler à la main ; c'était
    la partie la plus fragile de la boucle. Ils sont désormais bons d'origine.
    """
    t, _ = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6)],
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.4)],
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.4), ("est", 1.4, 1.8)],
    ])

    t._run_partial(_audio(0.6))
    t._run_partial(_audio(1.4))
    t._run_partial(_audio(1.8))

    assert [w["text"] for w in t._committed_words] == ["bonjour", "le", "monde"]
    assert t._committed_words[-1]["end"] == pytest.approx(1.4)


def test_le_prefixe_fige_ne_recule_jamais(monkeypatch):
    """Un mot déjà acquis n'est pas repris, même si le moteur change d'avis.

    Reprendre un mot déjà lu est plus déroutant qu'une petite erreur, que la
    passe finale corrigera de toute façon.
    """
    t, _ = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6)],
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6)],
        # Le moteur se ravise sur le deuxième mot.
        [("bonjour", 0.0, 0.4), ("la", 0.4, 0.6), ("suite", 0.6, 1.0)],
    ])

    t._run_partial(_audio(0.6))
    t._run_partial(_audio(0.6))
    assert [w["text"] for w in t._committed_words] == ["bonjour", "le"]

    t._run_partial(_audio(1.0))

    assert [w["text"] for w in t._committed_words] == ["bonjour", "le"]


def test_agreement_ignores_case_and_punctuation(monkeypatch):
    # Le moteur change une capitale ou recolle une ponctuation d'une passe à
    # l'autre ; l'accord les ignore, mais le mot figé garde son texte brut.
    t, _ = _make(monkeypatch, [
        [("Bonjour", 0.0, 0.5)],
        [("bonjour,", 0.0, 0.5), ("le", 0.5, 0.9)],
    ])

    t._run_partial(_audio(0.5))
    t._run_partial(_audio(0.9))

    assert [w["text"] for w in t._committed_words] == ["bonjour,"]  # texte brut préservé
    assert t._prev_words_norm == ["bonjour", "le"]


def test_un_tampon_trop_court_ne_declenche_pas_de_passe(monkeypatch):
    t, backend = _make(monkeypatch, [[("x", 0.0, 0.1)]])

    t._run_partial(_audio(0.2))  # 0,2 s < 0,3 s minimum

    assert backend.calls == []
    assert t._committed_words == []


def test_laffichage_montre_le_fige_puis_lhypothese(monkeypatch):
    t, _ = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6)],
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.0)],
    ])

    t._run_partial(_audio(0.6))
    t._run_partial(_audio(1.0))

    # Dernier instantané : les deux mots figés suivis de l'hypothèse courante,
    # publiés en **un seul** message (cf. la note de _run_partial).
    assert _partial_words(t.display_queue) == ["bonjour", "le", "monde"]


def test_final_segment_postprocesses_and_resets(monkeypatch):
    # A final pass post-processes the full text, persists it, emits final_text,
    # and clears the per-segment streaming state for the next utterance.
    t, _ = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.0)],
    ])
    saved: list[str] = []
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: saved.append(text))

    # Pretend we were mid-stream so we can prove the reset.
    t._committed_words = [{"text": "stale", "start": 0.0, "end": 0.1}]
    t._prev_words_norm = ["stale"]

    t._run_segment(_audio(1.0), is_final=True)

    events = _drain(t.display_queue)
    final = [e for e in events if e.get("type") == "final_text"]
    assert final and final[0]["text"] == "Bonjour le monde"
    assert saved == ["Bonjour le monde"]

    # État de streaming remis à zéro pour l'énoncé suivant.
    assert t._committed_words == []
    assert t._prev_words_norm == []


def test_final_segment_attaches_speaker_as_structured_field(monkeypatch):
    # With diarization on, the speaker label travels as a separate `speaker`
    # field — it is NOT glued into the text — and is persisted alongside.
    t, _ = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4), ("le", 0.4, 0.6), ("monde", 0.6, 1.0)],
    ])
    saved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: saved.append((text, speaker)))
    t.tagger = type("T", (), {"label": lambda self, a, sr: "B"})()

    t._run_segment(_audio(1.0), is_final=True)

    final = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"][0]
    assert final["text"] == "Bonjour le monde"  # text stays clean, no "B: " prefix
    assert final["speaker"] == "B"
    assert saved == [("Bonjour le monde", "B")]


def test_final_segment_omits_speaker_when_diarization_off(monkeypatch):
    # No tagger → no `speaker` key on the message at all.
    t, _ = _make(monkeypatch, [[("bonjour", 0.0, 0.4)]])
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: None)
    assert t.tagger is None

    t._run_segment(_audio(1.0), is_final=True)

    final = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"][0]
    assert "speaker" not in final


def test_transcribed_text_never_logged_above_debug(monkeypatch, caplog):
    # Le log est persisté sur disque (~/Library/Logs/Benji) et destiné à être
    # joint aux rapports de bug : le contenu transcrit ne doit apparaître qu'en
    # DEBUG, jamais au niveau INFO par défaut.
    t, _ = _make(monkeypatch, [
        [("secret", 0.0, 0.4), ("bancaire", 0.4, 1.0)],
    ])
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: None)

    with caplog.at_level("INFO", logger="benji"):
        t._run_segment(_audio(1.0), is_final=True)

    assert not any("secret" in r.getMessage().lower() for r in caplog.records)


def test_final_segment_drops_hallucination(monkeypatch):
    # Known Whisper hallucination on the final pass → emit a drop instead of
    # leaving the streamed words on screen, and don't persist anything.
    t, _ = _make(monkeypatch, [
        [("Merci", 0.0, 0.4), ("d'avoir", 0.4, 0.8), ("regardé", 0.8, 1.2)],
    ])
    saved: list[str] = []
    monkeypatch.setattr(t.history, "add", lambda text: saved.append(text))

    t._run_segment(_audio(1.2), is_final=True)

    events = _drain(t.display_queue)
    final = [e for e in events if e.get("type") == "final_text"]
    assert final and final[0]["text"] == "" and final[0]["drop"] is True
    assert saved == []  # nothing persisted


# --- diarisation : recouvrement avec le décodage final ---


def test_diarization_runs_while_the_final_pass_decodes(monkeypatch):
    """Le tagger démarre AVANT la fin du décodage, pas après.

    Le backend factice attend que la diarisation ait commencé pour rendre ses
    mots : enchaînée après le décodage (l'ancien comportement), elle ne
    démarrerait jamais et le décodage expirerait.
    """
    import threading

    started = threading.Event()

    class SlowTagger:
        def label(self, audio, sr):
            started.set()
            return "A"

    class WaitingBackend(FakeBackend):
        def transcribe(self, audio):
            assert started.wait(timeout=2.0), "diarisation non démarrée pendant le décodage"
            yield from super().transcribe(audio)

    backend = WaitingBackend([[("bonjour", 0.0, 0.5)]])
    monkeypatch.setattr(transcriber_mod, "build_backend", lambda *a, **kw: backend)
    # Sans ça, le Transcriber construit un vrai moteur final et va télécharger
    # Whisper. None = « réutilise le moteur des partielles ».
    monkeypatch.setattr(transcriber_mod, "build_final_backend", lambda *a, **kw: None)
    t = Transcriber(Queue(), Queue(), STTConfig(diarization=False), stats=None, sample_rate=SR)
    t.tagger = SlowTagger()
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: None)

    t._run_segment(_audio(1.0), is_final=True)

    final = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"][0]
    assert final["speaker"] == "A"


def test_hanging_diarizer_does_not_wedge_the_stt_loop(monkeypatch):
    """Un tagger bloqué rend un segment sans locuteur — jamais un thread figé."""
    import threading

    release = threading.Event()

    class HangingTagger:
        def label(self, audio, sr):
            release.wait(timeout=5.0)
            return "A"

    t, _ = _make(monkeypatch, [[("bonjour", 0.0, 0.5)]])
    t.tagger = HangingTagger()
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: None)
    monkeypatch.setattr(t, "_await_spans",
                        lambda future, timeout=0.05: Transcriber._await_spans(t, future, 0.05))

    try:
        t._run_segment(_audio(1.0), is_final=True)
        final = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"][0]
        assert final["text"] == "Bonjour"
        assert "speaker" not in final
    finally:
        release.set()


def test_tagger_error_is_not_fatal(monkeypatch):
    class BrokenTagger:
        def label(self, audio, sr):
            raise RuntimeError("modèle absent")

    t, _ = _make(monkeypatch, [[("bonjour", 0.0, 0.5)]])
    t.tagger = BrokenTagger()
    saved = []
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: saved.append(speaker))

    t._run_segment(_audio(1.0), is_final=True)

    assert saved == [None]


# --- routage des passes vers les deux moteurs ---


def _two_engine_transcriber(monkeypatch, partial_words, final_words):
    partial_backend = FakeBackend([partial_words, partial_words])
    final_backend = FakeBackend([final_words])
    final_backend.name = "final"
    monkeypatch.setattr(transcriber_mod, "build_backend", lambda *a, **kw: partial_backend)
    monkeypatch.setattr(transcriber_mod, "build_final_backend", lambda *a, **kw: final_backend)
    t = Transcriber(Queue(), Queue(), STTConfig(diarization=False), stats=None, sample_rate=SR)
    return t, partial_backend, final_backend


def test_les_partielles_et_la_finale_ne_vont_pas_au_meme_moteur(monkeypatch):
    """Vitesse là où le texte est jetable, garantie de langue là où il reste."""
    t, partial_backend, final_backend = _two_engine_transcriber(
        monkeypatch,
        [("brouillon", 0.0, 0.5)],
        [("définitif", 0.0, 0.5)],
    )
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: None)

    t._run_partial(_audio(1.0))
    assert len(partial_backend.calls) == 1
    assert final_backend.calls == []

    t._run_segment(_audio(1.0), is_final=True)
    assert len(partial_backend.calls) == 1  # la finale ne repasse pas par lui
    assert len(final_backend.calls) == 1

    final = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"][0]
    assert final["text"] == "Définitif"


def test_sans_moteur_final_le_moteur_des_partielles_prend_le_relais(monkeypatch):
    t, backend = _make(monkeypatch, [[("bonjour", 0.0, 0.5)]])

    assert t.final_backend is t.backend


# --- glossaire utilisateur ---


def test_le_glossaire_corrige_le_texte_final(monkeypatch):
    """Un terme du glossaire rattrape ce que le moteur a massacré.

    C'est le seul levier qui reste depuis le retrait d'`initial_prompt` avec
    Whisper : on ne souffle plus rien au modèle, on relit sa sortie.
    """
    from benji.stt import lexicon

    t, _ = _make(monkeypatch, [[("On", 0.0, 0.2), ("déploie", 0.2, 0.6),
                                ("kubernétesse", 0.6, 1.2)]])
    t._lexicon = lexicon.compile_terms(["Kubernetes"])
    # Le magasin est derrière le portillon de conservation : le remplacer seul
    # laisserait les écritures partir dans le vrai historique.
    t.history = _FakeHistory()
    t.consent = RecordingConsent(t.history, armed=True)

    t._run_segment(_audio(1.2), is_final=True)

    assert t.history.added == [("On déploie Kubernetes", None)]


def test_le_glossaire_ne_touche_pas_les_partielles(monkeypatch):
    """Voir un mot se réécrire sous les yeux est plus déroutant qu'une coquille
    passagère — et le texte partiel est remplacé par le final de toute façon."""
    from benji.stt import lexicon

    t, _ = _make(monkeypatch, [[("kubernétesse", 0.0, 0.6)]])
    t._lexicon = lexicon.compile_terms(["Kubernetes"])

    t._run_partial(_audio(0.6))

    assert _partial_words(t.display_queue) == ["kubernétesse"]


def test_glossaire_desactive_ne_lit_pas_le_disque(monkeypatch):
    """`STTConfig.glossary = False` doit court-circuiter jusqu'à la lecture."""
    read = []
    monkeypatch.setattr(transcriber_mod, "load_terms", lambda *a: read.append(1) or [])

    t, _ = _make(monkeypatch, [[]], glossary=False)

    assert t._lexicon == []
    assert read == []


# --- préchauffage sélectif ---


def test_un_backend_paresseux_nest_pas_prechauffe(monkeypatch):
    """Préchauffer Whisper au démarrage chargerait les ~1,5 Go que son
    chargement paresseux existe précisément pour éviter."""
    t, backend = _make(monkeypatch, [[], []])

    class _Lazy:
        name = "hybrid"
        eager_warmup = False

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio):
            self.calls += 1
            return iter(())

    lazy = _Lazy()
    t.final_backend = lazy

    t.warmup()

    assert lazy.calls == 0
    assert backend.calls, "le moteur des partielles, lui, doit être préchauffé"


class _FakeHistory:
    def __init__(self):
        self.added: list[tuple] = []

    def add(self, text, speaker=None, meeting_id=None):
        self.added.append((text, speaker))


def test_final_segment_splits_two_speakers_into_two_turns(monkeypatch):
    """Un segment VAD qui tient deux locuteurs rend deux finals, pas une phrase.

    C'est le cas des tours de parole rapprochés : sans pause franche, le VAD ne
    coupe pas, et une étiquette unique par segment fondait les deux voix.
    """
    t, _ = _make(monkeypatch, [[
        ("je", 0.0, 0.3), ("pense", 0.3, 0.8), ("que", 0.8, 1.1), ("oui", 1.1, 1.4),
        ("moi", 1.7, 2.0), ("non", 2.0, 2.3), ("pas", 2.3, 2.6), ("tout", 2.6, 3.0),
    ]])
    saved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(t.history, "add",
                        lambda text, speaker=None: saved.append((text, speaker)))

    labels = iter(["A", "A", "B"])

    class WindowTagger:
        def label(self, audio, sr):
            return next(labels, "B")

    t.tagger = WindowTagger()

    t._run_segment(_audio(3.0), is_final=True)

    finals = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"]
    assert [f["speaker"] for f in finals] == ["A", "B"]
    assert finals[0]["text"] == "Je pense que oui"
    assert finals[1]["text"] == "Moi non pas tout"
    # L'historique porte deux entrées distinctes, une par locuteur.
    assert saved == [("Je pense que oui", "A"), ("Moi non pas tout", "B")]


def test_single_speaker_segment_still_emits_one_final(monkeypatch):
    """Le cas courant ne paie rien : une voix, un tour, un seul final."""
    t, _ = _make(monkeypatch, [[
        ("bonjour", 0.0, 0.5), ("tout", 0.5, 0.9), ("le", 0.9, 1.1),
        ("monde", 1.1, 1.6), ("ça", 1.8, 2.0), ("va", 2.0, 2.4),
    ]])
    monkeypatch.setattr(t.history, "add", lambda text, speaker=None: None)
    t.tagger = type("T", (), {"label": lambda self, a, sr: "A"})()

    t._run_segment(_audio(2.5), is_final=True)

    finals = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"]
    assert len(finals) == 1
    assert finals[0]["speaker"] == "A"


def test_hallucinated_turn_is_dropped_without_taking_the_other_with_it(monkeypatch):
    """Une hallucination localisée n'emporte que son tour."""
    t, _ = _make(monkeypatch, [[
        ("Merci", 0.0, 0.4), ("d'avoir", 0.4, 0.8), ("regardé", 0.8, 1.2),
        ("on", 1.8, 2.0), ("reprend", 2.0, 2.5), ("lundi", 2.5, 3.0),
    ]])
    saved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(t.history, "add",
                        lambda text, speaker=None: saved.append((text, speaker)))

    labels = iter(["A", "A", "B"])
    t.tagger = type("T", (), {"label": lambda self, a, sr: next(labels, "B")})()

    t._run_segment(_audio(3.0), is_final=True)

    finals = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"]
    assert [f["text"] for f in finals] == ["On reprend lundi"]
    assert saved == [("On reprend lundi", "B")]


def test_une_passe_partielle_publie_un_seul_message(monkeypatch):
    """`display_queue` est bornée et les `put` sont bloquants.

    Publier mot à mot faisait attendre le thread STT sur le tick de 16 ms de
    `DisplayBus` — 53 ms mesurés sur 37 mots, pour une passe qui n'en coûte que
    150 de décodage — et déclenchait un relayout de l'overlay par mot.
    """
    mots = [(f"mot{i}", i * 0.1, i * 0.1 + 0.1) for i in range(37)]
    t, _ = _make(monkeypatch, [mots])

    t._run_partial(_audio(4.0))

    events = _drain(t.display_queue)
    assert [e["type"] for e in events] == ["segment_start", "partial"]
    assert len(events[1]["words"]) == 37


def test_segment_start_annonce_un_nouvel_enonce_pas_un_rafraichissement(monkeypatch):
    """C'est lui qui périme les corrections tardives et vide la pile de tours.

    Émis à chaque passe, il remettait l'overlay à zéro en plein énoncé — sans
    conséquence tant qu'un redessin complet suivait, faux dès qu'on ne publie
    plus qu'un delta.
    """
    t, _ = _make(monkeypatch, [
        [("bonjour", 0.0, 0.4)],
        [("bonjour", 0.0, 0.4), ("monde", 0.4, 0.8)],
    ])

    t._run_partial(_audio(0.6))
    t._run_partial(_audio(1.0))

    starts = [e for e in _drain(t.display_queue) if e["type"] == "segment_start"]
    assert len(starts) == 1


def test_le_transcripteur_affiche_sans_conserver_avant_laccord(monkeypatch):
    """Le direct ne dépend jamais de l'accord : ce qui change, c'est le disque."""
    t, _ = _make(monkeypatch, [[("bonjour", 0.0, 0.4), ("monde", 0.4, 0.8)]],
                 confirm_before_saving=True)
    t.history = _FakeHistory()
    t.consent = RecordingConsent(t.history)

    t._run_segment(_audio(1.0), is_final=True)

    finals = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"]
    assert finals and finals[0]["text"] == "Bonjour monde"  # affiché
    assert t.history.added == []                            # pas conservé
    assert t.consent.pending_count == 1                     # mais pas perdu

    assert t.consent.arm() == 1
    assert t.history.added == [("Bonjour monde", None)]


def test_un_terme_appris_vaut_pour_la_reunion_en_cours(monkeypatch, tmp_path):
    """Le glossaire est compilé une fois, exprès (le relire par segment mettrait
    le disque sur le chemin chaud). Un terme appris en réunion doit pourtant
    valoir tout de suite : au prochain lancement, c'est trop tard pour la seule
    réunion où l'on parle de ce client-là."""
    from benji.stt import lexicon

    glossaire = tmp_path / "glossary.txt"
    monkeypatch.setattr(lexicon, "glossary_path", lambda: glossaire)

    t, _ = _make(monkeypatch, [[("On", 0.0, 0.2), ("surveille", 0.2, 0.5),
                                ("avec", 0.5, 0.7), ("data", 0.7, 1.0),
                                ("dogue", 1.0, 1.4)]])
    t.history = _FakeHistory()
    t.consent = RecordingConsent(t.history, armed=True)
    assert t._lexicon == []

    assert t.learn_term("Datadog") is True

    t._run_segment(_audio(1.4), is_final=True)
    assert "Datadog" in t.history.added[0][0]


def test_apprendre_un_terme_deja_connu_ne_recompile_pas_pour_rien(monkeypatch, tmp_path):
    from benji.stt import lexicon

    glossaire = tmp_path / "glossary.txt"
    glossaire.write_text("Datadog\n", encoding="utf-8")
    monkeypatch.setattr(lexicon, "glossary_path", lambda: glossaire)

    t, _ = _make(monkeypatch, [[("x", 0.0, 0.1)]])

    assert t.learn_term("datadog") is False


# --- contexte après une coupure forcée ---------------------------------------


def test_drop_context_words_by_midpoint():
    from benji.stt.transcriber import drop_context_words

    words = [
        {"text": "fin", "start": 0.2, "end": 0.6},
        {"text": ",", "start": None, "end": None},  # suit le mot d'avant
        {"text": "à", "start": 0.9, "end": 1.2},     # milieu à 1,05 → gardé
        {"text": "cheval", "start": 1.2, "end": 1.6},
    ]
    assert [w["text"] for w in drop_context_words(words, 1.0)] == ["à", "cheval"]
    assert drop_context_words(words, 0.0) == words


def test_final_with_context_keeps_only_its_own_words(monkeypatch):
    t, backend = _make(monkeypatch, [[
        ("déjà", 0.1, 0.4), ("dit", 0.5, 0.8),   # dans la seconde de contexte
        ("la", 1.1, 1.3), ("suite", 1.4, 1.8),
    ]])
    t._run_segment(_audio(3.0), True, context_s=1.0)
    finals = [e for e in _drain(t.display_queue) if e.get("type") == "final_text"]
    assert [f["text"] for f in finals] == ["La suite"]
    # Le moteur a bien reçu le tampon complet, contexte compris.
    assert backend.calls == [3 * SR]
