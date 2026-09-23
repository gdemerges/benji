"""Moteurs de transcription : Parakeet pour le direct, Whisper pour l'archive.

Chaque passe a son moteur, parce qu'elles n'ont pas le même cahier des charges.

**Passes partielles → Parakeet TDT.** Le texte y est éphémère : il sera remplacé
par le final. Ce qui compte est la latence, et Parakeet ne paie que l'audio reçu
là où Whisper encode toujours une fenêtre paddée de 30 s — 58 ms contre ~680 ms
sur un tampon de 1,2 s, mesuré sur M4 Pro.

**Passe finale → Parakeet, rattrapé par Whisper.** Le texte y est définitif : il
part dans l'historique, les exports et les résumés. Parakeet fait de la
détection automatique sur 25 langues **sans aucun levier pour la forcer**
(confirmé par la fiche NVIDIA), et bascule en anglais sur des segments
difficiles — au milieu d'une réunion française, on obtient « the utility devient
also the chef d'orchestre ». Whisper accepte `language="fr"` : la dérive devient
impossible, mais il coûte ~5× plus cher sur *tous* les segments.

D'où le moteur hybride (`HybridFinalBackend`, défaut) : décoder avec Parakeet,
puis **relire le texte produit** et ne relancer Whisper que sur les segments qui
ont visiblement dérivé (cf. `benji/stt/language.py`). La garantie est conservée
là où elle se joue ; le coût n'est payé que là où il sert.

**Windows/Linux → faster-whisper.** Sans Apple Silicon, pas de MLX : le moteur
local de l'offre gratuite devient `FasterWhisperBackend` (CTranslate2), détecté
par l'absence de `parakeet_mlx` plutôt que par l'OS — un Mac Intel sans MLX
tombe dans le même cas. Contrairement à Parakeet, faster-whisper accepte
`language=` directement : pas besoin du relais hybride, un seul moteur suffit
pour les deux passes. Mesuré sur CPU (audio français synthétisé, cf.
`benji/stt/CLAUDE.md`) : `tiny` (~120-240 ms) est du même ordre que Parakeet et
sert aux passes partielles ; `small`/`medium` (0,9 à 3,8 s) sont trop lents en
partiel mais corrects en final, où une seule passe est payée par segment.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from benji.stt.language import drifts_from

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

_MLX_WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class STTBackend(Protocol):
    name: str
    # False = ne pas préchauffer au démarrage (le backend charge ses poids
    # paresseusement, et le préchauffage annulerait ce gain). Absent = True.
    eager_warmup: bool

    def transcribe(self, audio) -> Iterator[dict]:
        """Rend des mots `{"text": str, "start": float, "end": float}`.

        `start`/`end` sont en secondes depuis le début du tampon ; l'une ou
        l'autre peut être None si le moteur n'a pas produit d'horodatage.
        """
        ...


def group_tokens_into_words(tokens) -> list[dict]:
    """Regroupe les sous-mots de Parakeet en mots horodatés.

    Le modèle rend des morceaux de mots (`" De"`, `" c"`, `"ô"`, `"té"`) : un
    token qui commence par une espace ouvre un mot, les suivants s'y collent. Le
    mot hérite du début du premier morceau et de la fin du dernier — c'est ce qui
    permet à l'accord entre passes et à l'export SRT de fonctionner.

    Fonction pure : elle prend n'importe quel objet exposant `.text`, `.start` et
    `.end`, donc elle se teste sans charger le modèle.
    """
    words: list[dict] = []
    for token in tokens:
        text = getattr(token, "text", "") or ""
        if not text.strip():
            continue
        start = getattr(token, "start", None)
        end = getattr(token, "end", None)
        if text.startswith(" ") or not words:
            words.append({"text": text.strip(), "start": start, "end": end})
        else:
            # Suite du mot courant : on étend sa borne de fin.
            words[-1]["text"] += text.strip()
            if end is not None:
                words[-1]["end"] = end
    return words


def words_from_result(result) -> list[dict]:
    """Mots horodatés d'un `AlignedResult`, phrase par phrase.

    Le regroupement est fait **par phrase** et non sur les tokens aplatis : le
    premier morceau d'une phrase ne porte pas toujours l'espace de tête, si bien
    qu'aplatir recollait la fin d'une phrase au début de la suivante
    (« Apple.Ça »).
    """
    words: list[dict] = []
    for sentence in getattr(result, "sentences", []) or []:
        words.extend(group_tokens_into_words(getattr(sentence, "tokens", []) or []))
    return words


class ParakeetBackend:
    """Parakeet TDT, alimenté **en mémoire**.

    L'API publique de `parakeet-mlx` ne transcrit que des chemins de fichiers. On
    passe donc par `get_logmel()` + `generate()` : Benji a déjà l'audio en numpy,
    et écrire les tampons d'une réunion dans un fichier temporaire serait une
    régression de confidentialité. Effet de bord heureux : pas de dépendance à
    ffmpeg.
    """

    name = "parakeet"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        import mlx.core as mx

        # `parakeet_mlx.audio` importe librosa pour un unique appel — la matrice
        # de filtres mel — qui vaut ~40 Mo de dépendances dans le DMG. On pose
        # notre équivalent avant l'import ; sans effet si un vrai librosa est
        # installé. Doit rester *avant* la ligne suivante (cf. mel_filters).
        from benji.stt import mel_filters

        mel_filters.install()

        from parakeet_mlx import from_pretrained

        self.model_id = model_id
        log.info("Chargement de Parakeet '%s'...", model_id)
        self.model = from_pretrained(model_id)
        self.preprocess = self.model.preprocessor_config

        # Matérialise les poids **sur ce thread**. MLX charge paresseusement et
        # lie les tableaux au stream du thread qui les évalue en premier ; sans
        # cet appel, la liaison n'a lieu qu'au premier décodage réel, et toute
        # inférence depuis un autre thread lève « There is no Stream(gpu, N) in
        # current thread ». Corollaire : ce constructeur doit être appelé depuis
        # un thread qui vit aussi longtemps que l'app (cf. `benji/app.py`).
        mx.eval(self.model.parameters())
        log.info("Parakeet prêt (16 kHz natif, décodage glouton)")

    def transcribe(self, audio) -> Iterator[dict]:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        if audio is None or len(audio) == 0:
            return
        mel = get_logmel(mx.array(audio), self.preprocess)
        for result in self.model.generate(mel):
            yield from words_from_result(result)


class WhisperBackend:
    """Whisper via mlx-whisper, **langue figée à la construction**.

    La langue ne varie pas d'un segment à l'autre au sein d'une session : la
    fixer ici garde le protocole `transcribe(audio)` à un seul argument, commun
    aux deux moteurs.

    Ne sert que sur la passe finale, d'où la chaîne de repli en température : un
    décodage glouton raté est retenté plus chaud, ce qu'on ne peut pas se
    permettre sur une partielle mais qui vaut le coup sur du définitif.

    **Chargement paresseux.** `mlx_whisper.transcribe` charge et met en cache le
    modèle au premier appel ; construire ce backend ne coûte donc qu'un import,
    et les ~1,5 Go de poids ne sont payés que si un segment en a réellement
    besoin. En moteur hybride, une réunion française entière peut se dérouler
    sans jamais les charger.
    """

    name = "whisper"
    # Le préchauffage forcerait le chargement des poids au démarrage, ce que le
    # chargement paresseux existe précisément pour éviter.
    eager_warmup = False

    def __init__(self, model_size: str = "medium", language: str | None = "fr"):
        self._mlx = None
        self.repo = _whisper_repo(model_size)
        self.language = language
        log.info("Whisper '%s' armé (langue : %s) — poids chargés au 1er usage",
                 self.repo, language or "auto")

    def _engine(self):
        if self._mlx is None:
            import mlx_whisper

            self._mlx = mlx_whisper
        return self._mlx

    def transcribe(self, audio) -> Iterator[dict]:
        if audio is None or len(audio) == 0:
            return
        result = self._engine().transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=(0.0, 0.2, 0.4),
            verbose=None,
        )
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                text = (w.get("word") or "").strip()
                if text:
                    yield {"text": text, "start": w.get("start"), "end": w.get("end")}


def _faster_whisper_device(has_cuda: bool) -> tuple[str, str]:
    """Device et type de calcul — pure, pour se tester sans `ctranslate2`.

    CUDA non mesuré sur ce projet (pas de GPU NVIDIA disponible) : les
    benchmarks connus de la lib suggèrent ×10-20 sur CPU, de quoi rapprocher
    small/medium du niveau Parakeet pour un utilisateur Windows avec carte
    dédiée — à confirmer avant d'en faire un choix d'architecture.
    """
    return ("cuda", "float16") if has_cuda else ("cpu", "int8")


def _words_from_segments(segments) -> Iterator[dict]:
    """Mots horodatés à partir des segments faster-whisper.

    Contrairement à Parakeet, faster-whisper rend déjà des mots entiers (pas de
    sous-mots à recoller) : `word_timestamps=True` suffit. Fonction pure — elle
    prend n'importe quel itérable de `seg.words[].{word,start,end}`, donc elle
    se teste sans charger le moindre modèle.
    """
    for seg in segments:
        for w in seg.words or []:
            text = (getattr(w, "word", "") or "").strip()
            if text:
                yield {"text": text, "start": w.start, "end": w.end}


class FasterWhisperBackend:
    """Whisper via faster-whisper (CTranslate2) — le moteur local hors Mac.

    Sans MLX, pas de Parakeet : `language=` est passé directement à chaque
    appel, donc pas de relais hybride à construire — un seul moteur sert aux
    deux passes, juste dimensionné différemment (cf. `build_backend` /
    `build_final_backend`). CPU par défaut (int8) ; CUDA utilisé
    automatiquement s'il est détecté (float16, non validé sur ce projet).

    Chargé **au constructeur**, pas paresseusement comme `WhisperBackend` : sur
    ce chemin faster-whisper est le moteur de tous les segments, partiels et
    finaux — le charger tard ne ferait que déplacer le coût sur le premier
    segment réel, en pleine réunion.
    """

    name = "faster-whisper"

    def __init__(self, model_size: str = "tiny", language: str | None = "fr"):
        from faster_whisper import WhisperModel

        try:
            import ctranslate2

            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        device, compute_type = _faster_whisper_device(has_cuda)

        self.language = language
        log.info("Chargement de faster-whisper '%s' sur %s (%s)...",
                 model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio) -> Iterator[dict]:
        if audio is None or len(audio) == 0:
            return
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            # Glouton : mesuré comme le meilleur compromis latence/qualité pour
            # les tampons courts de Benji (cf. benji/stt/CLAUDE.md).
            beam_size=1,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        yield from _words_from_segments(segments)


class HybridFinalBackend:
    """Parakeet d'abord, Whisper **seulement si la langue a dérivé**.

    Le compromis précédent était binaire : payer ~800 ms de Whisper sur *chaque*
    segment final pour se prémunir d'une dérive qui ne concerne qu'une poignée
    d'entre eux, ou garder Parakeet partout et laisser passer « the utility
    devient also the chef d'orchestre » dans l'historique.

    Ici l'arbitrage se fait **sur le texte produit** : on décode avec Parakeet
    (~150 ms), on regarde les mots-outils du résultat (cf. `stt/language.py`), et
    on ne relance Whisper que si le segment n'est visiblement pas dans la langue
    attendue — ou si Parakeet n'a rien rendu. Sur une réunion française propre,
    la passe lourde ne se déclenche presque jamais et ses poids ne sont même pas
    chargés.

    Le prix : la passe finale ne streame plus mot à mot, puisqu'il faut avoir lu
    tout le texte de Parakeet pour décider s'il est recevable. C'est sans effet
    visible — les mots de l'énoncé sont déjà à l'écran, posés par les passes
    partielles, et le final les remplace en bloc de toute façon.

    **Whisper tourne sur un thread dédié, créé une fois pour toutes.** MLX lie
    ses tableaux au stream du thread qui les évalue en premier : chargé depuis le
    thread STT, le modèle deviendrait inutilisable si le superviseur relançait ce
    thread après un incident (cf. `benji/app.py`). Un worker qui vit aussi
    longtemps que le backend supprime la question.
    """

    name = "hybrid"
    eager_warmup = False

    def __init__(self, fast: STTBackend, slow: STTBackend, language: str | None):
        self.fast = fast
        self.slow = slow
        self.language = language
        self._pool: ThreadPoolExecutor | None = None

    def _slow_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="STT-whisper"
            )
        return self._pool

    def transcribe(self, audio) -> Iterator[dict]:
        words = list(self.fast.transcribe(audio))
        text = " ".join(w["text"] for w in words)
        if words and not drifts_from(text, self.language):
            yield from words
            return

        reason = "aucun mot rendu" if not words else "langue dérivée"
        log.info("Passe finale reprise par Whisper (%s)", reason)
        try:
            # `list(...)` DANS le worker : le générateur ne doit pas être
            # consommé depuis le thread appelant, sinon l'inférence repartirait
            # sur le mauvais stream MLX.
            rescued = self._slow_pool().submit(lambda: list(self.slow.transcribe(audio))).result()
        except Exception as e:
            log.warning("Repli Whisper impossible (%s) — on garde Parakeet", e)
            yield from words
            return
        yield from (rescued or words)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


DEFAULT_FASTER_WHISPER_PARTIAL_MODEL = "tiny"


def _parakeet_available() -> bool:
    """Présence de Parakeet/MLX — Apple Silicon seulement.

    Sonde plutôt que `platform.system() == "Darwin"` : un Mac Intel sans mlx
    (les deux sont des dépendances dures marquées `sys_platform == 'darwin'`
    dans pyproject.toml, pas `sys_platform == 'darwin' and arm64`) tombe dans
    le même cas qu'un Windows/Linux, et doit prendre le même chemin.
    """
    return importlib.util.find_spec("parakeet_mlx") is not None


def build_backend(model_id: str = DEFAULT_MODEL) -> STTBackend:
    """Moteur des passes partielles."""
    if not _parakeet_available():
        return FasterWhisperBackend(DEFAULT_FASTER_WHISPER_PARTIAL_MODEL)
    return ParakeetBackend(model_id)


def _whisper_repo(model_size: str) -> str:
    return _MLX_WHISPER_MODELS.get(model_size, f"mlx-community/whisper-{model_size}-mlx")


def _whisper_available() -> bool:
    """Présence de mlx-whisper, sans en charger les poids.

    `WhisperBackend` n'importe plus le module à la construction (chargement
    paresseux) : sans cette sonde, l'absence du paquet ne se manifesterait qu'au
    premier segment, en pleine réunion.
    """
    return importlib.util.find_spec("mlx_whisper") is not None


def build_final_backend(
    engine: str, model_size: str, language: str | None, fast: STTBackend | None = None
) -> STTBackend:
    """Moteur de la passe finale — celui dont le texte est conservé.

    - Sans Parakeet/MLX (Windows/Linux) — `FasterWhisperBackend`, seul : il
      accepte `language=` directement, donc aucun relais hybride n'a de sens
      ici. `engine` (qui n'a de signification que pour l'arbitrage MLX) est
      ignoré sur ce chemin.
    - `"hybrid"` (défaut, Mac) — Parakeet, relayé par Whisper sur les seuls
      segments qui ont dérivé. Exige `fast`, le moteur des partielles, qu'il
      réutilise.
    - `"whisper"` — Whisper sur tous les finals : la garantie maximale, au prix
      fort. Le repli si l'hybride déçoit en réunion.
    - `"parakeet"` — renvoie None : l'appelant réutilise le moteur des
      partielles, au prix de la garantie de langue.
    """
    if not _parakeet_available():
        return FasterWhisperBackend(model_size, language)
    if engine == "parakeet":
        return None
    if not _whisper_available():
        log.warning(
            "mlx-whisper absent : la passe finale retombe sur Parakeet, dont la "
            "langue n'est pas garantie."
        )
        return None
    from benji import onboarding

    try:
        # Chargé paresseusement, en pleine réunion : c'est maintenant qu'on
        # vérifie qu'il ne déclenchera pas un téléchargement non accepté.
        onboarding.ensure_allowed(_whisper_repo(model_size))
    except onboarding.ModelNotAllowed:
        log.warning(
            "Whisper non téléchargé et non autorisé : la passe finale reste sur "
            "Parakeet, dont la langue n'est pas garantie."
        )
        return None
    whisper = WhisperBackend(model_size, language)
    if engine == "whisper":
        return whisper
    if fast is None:
        log.warning("Moteur hybride demandé sans moteur rapide — Whisper seul.")
        return whisper
    return HybridFinalBackend(fast, whisper, language)
