"""Backend Parakeet : regroupement des sous-mots et sélection du moteur.

Parakeet rend des morceaux de mots (`" De"`, `" c"`, `"ô"`, `"té"`) ; tout le
reste de Benji — découpage incrémental, overlay, export SRT — raisonne en mots
horodatés. Le regroupement est donc la pièce à ne pas rater : elle est pure et
testée sans charger le moindre modèle.
"""

from dataclasses import dataclass

import pytest

import benji.stt.backend as backend_mod
from benji.stt.backend import build_backend, group_tokens_into_words, words_from_result


@pytest.fixture(autouse=True)
def _chemin_parakeet(monkeypatch):
    """Ce module teste le chemin MLX : la sonde est fixée, pas lue sur la
    machine. Sans ça, sous Windows/Linux, `build_backend` prendrait
    faster-whisper et irait chercher des poids (cf. le miroir dans
    `test_faster_whisper_backend.py`)."""
    monkeypatch.setattr(backend_mod, "_parakeet_available", lambda: True)


@dataclass
class Tok:
    text: str
    start: float | None = None
    end: float | None = None


def test_les_sous_mots_sont_recolles():
    tokens = [Tok(" De", 0.0, 0.16), Tok(" c", 0.32, 0.4), Tok("ô", 0.4, 0.48),
              Tok("té", 0.48, 0.56)]

    assert [w["text"] for w in group_tokens_into_words(tokens)] == ["De", "côté"]


def test_le_mot_herite_du_debut_du_premier_et_de_la_fin_du_dernier():
    tokens = [Tok(" bu", 0.8, 0.96), Tok("ild", 0.96, 1.12)]

    word = group_tokens_into_words(tokens)[0]

    assert word == {"text": "build", "start": 0.8, "end": 1.12}


def test_la_ponctuation_colle_au_mot_precedent():
    tokens = [Tok(" démarre", 0.2, 0.6), Tok(",", 0.6, 0.62), Tok(" on", 0.7, 0.8)]

    assert [w["text"] for w in group_tokens_into_words(tokens)] == ["démarre,", "on"]


def test_un_premier_token_sans_espace_ouvre_quand_meme_un_mot():
    """Le tout premier morceau d'un tampon n'est pas toujours préfixé d'une espace."""
    assert [w["text"] for w in group_tokens_into_words([Tok("Bon", 0.0, 0.3)])] == ["Bon"]


def test_les_tokens_vides_sont_ignores():
    tokens = [Tok(" un", 0.0, 0.2), Tok("  ", 0.2, 0.21), Tok(" deux", 0.3, 0.5)]

    assert [w["text"] for w in group_tokens_into_words(tokens)] == ["un", "deux"]


def test_timestamps_absents_tolérés():
    """Un mot sans borne doit sortir avec None, jamais lever."""
    words = group_tokens_into_words([Tok(" mot"), Tok("s")])

    assert words == [{"text": "mots", "start": None, "end": None}]


def test_liste_vide():
    assert group_tokens_into_words([]) == []


@dataclass
class Sentence:
    tokens: list


@dataclass
class Result:
    sentences: list


def test_deux_phrases_ne_se_recollent_pas():
    """Le premier morceau d'une phrase n'a pas toujours l'espace de tête.

    Sur les tokens aplatis, la fin d'une phrase se recollait au début de la
    suivante : « Apple.Ça » au lieu de « Apple. » puis « Ça ».
    """
    result = Result([
        Sentence([Tok(" Apple", 1.0, 1.4), Tok(".", 1.4, 1.42)]),
        Sentence([Tok("Ça", 1.6, 1.8), Tok(" prend", 1.8, 2.0)]),
    ])

    assert [w["text"] for w in words_from_result(result)] == ["Apple.", "Ça", "prend"]


def test_resultat_sans_phrase():
    assert words_from_result(Result([])) == []
    assert words_from_result(object()) == []


# --- construction du backend ---


def test_build_backend_rend_parakeet(monkeypatch):
    seen = {}

    def _fake(model_id):
        seen["id"] = model_id
        return type("B", (), {"name": "parakeet"})()

    monkeypatch.setattr(backend_mod, "ParakeetBackend", _fake)

    backend = build_backend()

    assert backend.name == "parakeet"
    assert seen["id"] == "mlx-community/parakeet-tdt-0.6b-v3"


def test_build_backend_accepte_dautres_poids(monkeypatch):
    seen = {}

    def _fake(model_id):
        seen["id"] = model_id
        return type("B", (), {"name": "parakeet"})()

    monkeypatch.setattr(backend_mod, "ParakeetBackend", _fake)
    build_backend("mlx-community/parakeet-tdt-0.6b-v2")

    assert seen["id"] == "mlx-community/parakeet-tdt-0.6b-v2"


# --- liaison au thread MLX ---


def test_les_poids_sont_materialises_a_la_construction(monkeypatch):
    """MLX lie les tableaux au stream du thread qui les évalue en premier.

    Sans `mx.eval()` au chargement, la liaison n'a lieu qu'au premier décodage
    réel : toute inférence lancée depuis le thread STT lève alors « There is no
    Stream(gpu, N) in current thread ». On ne peut pas compter sur `warmup()`,
    qui préchauffe sur du silence — Parakeet n'en décode aucun token, donc le
    décodeur ne tourne jamais et rien n'est lié.
    """
    mx = pytest.importorskip("mlx.core")
    parakeet_mlx = pytest.importorskip("parakeet_mlx")

    from benji.stt.backend import ParakeetBackend

    params = {"encoder": "poids"}

    class _FakeModel:
        preprocessor_config = object()

        def parameters(self):
            return params

    evaluated = []
    monkeypatch.setattr(parakeet_mlx, "from_pretrained", lambda model_id: _FakeModel())
    monkeypatch.setattr(mx, "eval", lambda *args: evaluated.append(args))

    ParakeetBackend("mlx-community/parakeet-tdt-0.6b-v3")

    assert evaluated == [(params,)], "les poids doivent être matérialisés au chargement"


# --- routage des passes ---


def test_le_moteur_final_force_le_tout_whisper_sur_demande(monkeypatch):
    """`final_engine="whisper"` = la garantie maximale, payée sur chaque segment.

    Ce n'est plus le défaut (l'hybride l'est), mais ça reste le repli si la
    relecture du texte laissait passer une dérive en réunion.
    """
    from benji.stt.backend import build_final_backend

    seen = {}

    class _FakeWhisper:
        name = "whisper"

        def __init__(self, model_size, language):
            seen["model_size"] = model_size
            seen["language"] = language

    monkeypatch.setattr(backend_mod, "WhisperBackend", _FakeWhisper)
    monkeypatch.setattr(backend_mod, "_whisper_available", lambda: True)

    backend = build_final_backend("whisper", "medium", "fr")

    assert backend.name == "whisper"
    assert seen == {"model_size": "medium", "language": "fr"}


def test_final_en_parakeet_ne_construit_pas_de_second_moteur():
    """`final_engine="parakeet"` = réutiliser le moteur des partielles."""
    from benji.stt.backend import build_final_backend

    assert build_final_backend("parakeet", "medium", "fr") is None


def test_mlx_whisper_absent_ne_bloque_pas_le_demarrage(monkeypatch):
    """Sans Whisper, on transcrit quand même — sans garantie de langue.

    La sonde remplace l'ancien `try: import` : le backend charge désormais ses
    poids paresseusement, donc l'absence du paquet ne se manifesterait qu'au
    premier segment — en pleine réunion.
    """
    from benji.stt.backend import build_final_backend

    monkeypatch.setattr(backend_mod, "_whisper_available", lambda: False)

    assert build_final_backend("whisper", "medium", "fr") is None
    assert build_final_backend("hybrid", "medium", "fr", fast=object()) is None


# --- moteur hybride : Parakeet, rattrapé par Whisper ---


class _Recorder:
    """Faux moteur qui rend un texte fixe et compte ses appels."""

    def __init__(self, name, text):
        self.name = name
        self._text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        for word in self._text.split():
            yield {"text": word, "start": None, "end": None}


def _hybrid(fast_text, slow_text="corrigé par Whisper", language="fr"):
    from benji.stt.backend import HybridFinalBackend

    fast = _Recorder("parakeet", fast_text)
    slow = _Recorder("whisper", slow_text)
    return HybridFinalBackend(fast, slow, language), fast, slow


def test_hybride_garde_parakeet_quand_la_langue_tient():
    """Le cas nominal : Whisper n'est jamais appelé, ses poids restent déchargés."""
    hybrid, fast, slow = _hybrid("On valide la roadmap avec le client demain")

    words = [w["text"] for w in hybrid.transcribe([0.0] * 16000)]

    assert words[0] == "On"
    assert fast.calls == 1
    assert slow.calls == 0


def test_hybride_repasse_par_whisper_quand_la_langue_derive():
    hybrid, fast, slow = _hybrid("the utility devient also the chef")

    words = [w["text"] for w in hybrid.transcribe([0.0] * 16000)]

    assert slow.calls == 1
    assert words == ["corrigé", "par", "Whisper"]
    hybrid.shutdown()


def test_hybride_repasse_par_whisper_quand_parakeet_ne_rend_rien():
    hybrid, fast, slow = _hybrid("")

    words = [w["text"] for w in hybrid.transcribe([0.0] * 16000)]

    assert slow.calls == 1
    assert words == ["corrigé", "par", "Whisper"]
    hybrid.shutdown()


def test_un_whisper_en_panne_ne_perd_pas_le_segment():
    """Le repli est un bonus, jamais une dépendance : si Whisper échoue, on
    garde le texte de Parakeet plutôt que de perdre la phrase."""
    from benji.stt.backend import HybridFinalBackend

    class _Broken:
        name = "whisper"

        def transcribe(self, audio):
            raise RuntimeError("poids introuvables")
            yield  # pragma: no cover

    fast = _Recorder("parakeet", "the utility devient also the chef")
    hybrid = HybridFinalBackend(fast, _Broken(), "fr")

    words = [w["text"] for w in hybrid.transcribe([0.0] * 16000)]

    assert words[0] == "the"
    hybrid.shutdown()


def test_whisper_tourne_sur_un_thread_dedie_et_stable():
    """MLX lie ses tableaux au stream du thread qui les évalue en premier.

    Chargé depuis le thread STT, Whisper deviendrait inutilisable si le
    superviseur relançait ce thread après un incident. Deux segments doivent
    donc être décodés par le **même** thread — et jamais celui de l'appelant.
    """
    import threading

    from benji.stt.backend import HybridFinalBackend

    threads = []

    class _ThreadSpy:
        name = "whisper"

        def transcribe(self, audio):
            threads.append(threading.current_thread().name)
            yield {"text": "ok", "start": None, "end": None}

    fast = _Recorder("parakeet", "the utility devient also the chef")
    hybrid = HybridFinalBackend(fast, _ThreadSpy(), "fr")

    list(hybrid.transcribe([0.0] * 16000))
    list(hybrid.transcribe([0.0] * 16000))

    assert len(threads) == 2
    assert threads[0] == threads[1]
    assert threads[0] != threading.current_thread().name
    hybrid.shutdown()


def test_hybride_construit_par_defaut(monkeypatch):
    from benji.config import STTConfig
    from benji.stt.backend import build_final_backend

    monkeypatch.setattr(backend_mod, "WhisperBackend", lambda *a: _Recorder("whisper", "x"))
    monkeypatch.setattr(backend_mod, "_whisper_available", lambda: True)

    backend = build_final_backend(
        STTConfig().final_engine, "medium", "fr", fast=_Recorder("parakeet", "y")
    )

    assert backend.name == "hybrid"


def test_whisper_ne_charge_pas_ses_poids_a_la_construction(monkeypatch):
    """Le gain mémoire de l'hybride tient entièrement à ce chargement différé."""
    from benji.stt.backend import WhisperBackend

    backend = WhisperBackend("medium", "fr")

    assert backend._mlx is None
    assert backend.eager_warmup is False
