"""Speaker labeling backends.

Two implementations:
- `SpeakerTagger` (pitch): F0 autocorrelation + 2-speaker clustering. No deps,
  works offline on Apple Silicon, but unreliable when voices have similar pitch.
- `PyannoteSpeakerTagger`: pyannote.audio speaker embeddings + cosine clustering.
  Real diarization quality, supports >2 speakers. Requires `pyannote.audio` and
  an HF token (env `HF_TOKEN`) for first-time model download.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class DiarizationBackend(Protocol):
    def label(self, audio: np.ndarray, sample_rate: int = 16000) -> str | None: ...


def _estimate_f0(audio: np.ndarray, sample_rate: int = 16000,
                 fmin: float = 70.0, fmax: float = 400.0) -> float | None:
    """Return median F0 estimate in Hz, or None if unvoiced / too short."""
    if len(audio) < sample_rate // 4:  # <250 ms
        return None

    # Take central 1 second to avoid silence at edges
    target_len = min(len(audio), sample_rate)
    start = (len(audio) - target_len) // 2
    segment = audio[start:start + target_len].astype(np.float32)
    segment = segment - segment.mean()
    if np.max(np.abs(segment)) < 1e-3:
        return None

    # Autocorrelation
    corr = np.correlate(segment, segment, mode="full")[len(segment) - 1:]
    min_lag = int(sample_rate / fmax)
    max_lag = int(sample_rate / fmin)
    if max_lag >= len(corr):
        return None
    window = corr[min_lag:max_lag]
    peak = int(np.argmax(window)) + min_lag
    if corr[peak] < 0.3 * corr[0]:  # low periodicity → unvoiced
        return None
    return sample_rate / peak


class SpeakerTagger:
    """Assigns A/B labels based on F0 clustering with a rolling reference."""

    def __init__(self, f0_gap_hz: float = 40.0):
        self.f0_gap_hz = f0_gap_hz
        self._speaker_f0: dict[str, float] = {}
        self._last_label: str | None = None

    def label(self, audio: np.ndarray, sample_rate: int = 16000) -> str | None:
        f0 = _estimate_f0(audio, sample_rate)
        if f0 is None:
            return self._last_label  # fallback to previous speaker

        if not self._speaker_f0:
            self._speaker_f0["A"] = f0
            self._last_label = "A"
            return "A"

        # Find closest existing speaker
        best_label, best_delta = min(
            ((lbl, abs(f0 - ref)) for lbl, ref in self._speaker_f0.items()),
            key=lambda x: x[1],
        )

        if best_delta <= self.f0_gap_hz:
            # Same speaker — update rolling reference (EMA)
            prev = self._speaker_f0[best_label]
            self._speaker_f0[best_label] = 0.8 * prev + 0.2 * f0
            self._last_label = best_label
            return best_label

        # New speaker (cap at 2)
        if len(self._speaker_f0) < 2:
            new_label = "B" if "A" in self._speaker_f0 else "A"
            self._speaker_f0[new_label] = f0
            self._last_label = new_label
            return new_label

        # Already 2 speakers — assign to closest anyway
        self._last_label = best_label
        return best_label


class PyannoteSpeakerTagger:
    """Real speaker labeling using pyannote.audio embeddings + cosine clustering.

    For each segment we compute a 512-d embedding, then assign it to the closest
    existing centroid (cosine sim > threshold) or spawn a new speaker (up to
    `max_speakers`). Centroids update via running mean.
    """

    def __init__(
        self,
        max_speakers: int = 4,
        cosine_threshold: float = 0.55,
        model_id: str = "pyannote/embedding",
        # Commit épinglé du repo HF : un repo compromis ne peut pas substituer
        # des poids modifiés (les checkpoints torch peuvent exécuter du code).
        model_revision: str = "4db4899737a38b2d618bbd74350915aa10293cb2",
    ):
        try:
            from pyannote.audio import Inference, Model
        except ImportError as e:
            raise RuntimeError(
                "pyannote.audio is required for diarization_backend='pyannote'. "
                "Install with: uv sync --extra diarization"
            ) from e

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            log.warning("No HF_TOKEN set — pyannote model download may fail")

        # pyannote 4.x: load the Model (auth via `token`) then wrap it in Inference.
        # `whole` averages embeddings across the full clip — what we want per segment.
        model = Model.from_pretrained(model_id, revision=model_revision, token=token)
        self._inference = Inference(model, window="whole")
        self.max_speakers = max_speakers
        self.cosine_threshold = cosine_threshold
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._next_id = 0
        log.info("pyannote.audio loaded ('%s')", model_id)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def _new_label(self) -> str:
        # A, B, C, ... up to max_speakers, then numeric.
        if self._next_id < 26:
            label = chr(ord("A") + self._next_id)
        else:
            label = f"S{self._next_id}"
        self._next_id += 1
        return label

    def label(self, audio: np.ndarray, sample_rate: int = 16000) -> str | None:
        if len(audio) < sample_rate // 2:  # <500 ms — too short for stable embedding
            return None
        try:
            # pyannote expects torch tensor with shape (channel, samples)
            import torch
            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            emb = self._inference({"waveform": waveform, "sample_rate": sample_rate})
            emb = np.asarray(emb).flatten()
        except Exception as e:
            log.warning("pyannote inference failed: %s", e)
            return None

        if not self._centroids:
            label = self._new_label()
            self._centroids[label] = emb
            self._counts[label] = 1
            return label

        best_label, best_sim = max(
            ((lbl, self._cos(emb, c)) for lbl, c in self._centroids.items()),
            key=lambda x: x[1],
        )

        if best_sim >= self.cosine_threshold or len(self._centroids) >= self.max_speakers:
            n = self._counts[best_label] + 1
            self._centroids[best_label] = self._centroids[best_label] * (n - 1) / n + emb / n
            self._counts[best_label] = n
            return best_label

        label = self._new_label()
        self._centroids[label] = emb
        self._counts[label] = 1
        return label


def build_tagger(backend: str, max_speakers: int = 4) -> DiarizationBackend:
    """Factory: returns a diarization tagger, falling back to pitch on error."""
    if backend == "pyannote":
        from benji import onboarding

        try:
            # pyannote irait chercher ses poids sur le Hub : pas sans accord.
            onboarding.ensure_allowed("pyannote/embedding")
            return PyannoteSpeakerTagger(max_speakers=max_speakers)
        except Exception as e:
            log.warning("pyannote unavailable (%s), falling back to pitch", e)
    return SpeakerTagger()


# --- Découpe d'un segment en tours de parole ---------------------------------
#
# Un segment VAD n'est pas un tour de parole : avec `silence_duration_ms = 600`,
# deux personnes qui s'enchaînent sans vraie pause tiennent dans le même tampon.
# Une étiquette unique par segment fondait alors les deux voix en une seule
# phrase, d'une seule couleur. On étiquette donc **par fenêtres**, puis on
# recoupe les mots (qui sont horodatés) aux frontières de locuteur.
#
# Ces deux fonctions sont volontairement séparées : `label_windows` a besoin du
# tagger (donc du modèle), `split_by_speaker` est **pure** et se teste sans rien
# charger.

SpeakerSpan = tuple[float, float, "str | None"]


def label_windows(
    tagger: DiarizationBackend,
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_s: float = 1.5,
    hop_s: float = 0.75,
) -> list[SpeakerSpan]:
    """Étiquette le tampon par fenêtres glissantes.

    Rend une liste de `(début, fin, locuteur)` en secondes depuis le début du
    tampon. Un tampon plus court qu'une fenêtre donne une seule étendue — le
    comportement d'avant, au coût d'avant.

    Les fenêtres sont soumises au tagger **dans l'ordre chronologique** : son
    clustering est incrémental (centroïdes mis à jour au fil de l'eau), le
    désordre changerait ses décisions.
    """
    duration = len(audio) / sample_rate
    if duration <= 0:
        return []
    if duration <= window_s or hop_s <= 0:
        return [(0.0, duration, tagger.label(audio, sample_rate))]

    spans: list[SpeakerSpan] = []
    start = 0.0
    while True:
        end = min(start + window_s, duration)
        # Une fenêtre résiduelle trop courte ne produit pas d'embedding stable
        # (pyannote rend None sous 500 ms) : la laisser tomber vaut mieux qu'une
        # étiquette tirée au sort, les mots concernés héritent de la voisine.
        if end - start >= 0.5:
            chunk = audio[int(start * sample_rate):int(end * sample_rate)]
            spans.append((start, end, tagger.label(chunk, sample_rate)))
        if end >= duration:
            break
        start += hop_s
    return spans


def _word_time(word: dict) -> float | None:
    """Instant représentatif d'un mot : son milieu, ou ce qu'on a."""
    start, end = word.get("start"), word.get("end")
    if start is not None and end is not None:
        return (start + end) / 2
    return start if start is not None else end


def _span_for(time: float, spans: list[SpeakerSpan]) -> str | None:
    """Étiquette de l'étendue qui couvre `time`, sinon celle du centre le plus proche."""
    covering = [s for s in spans if s[0] <= time < s[1]]
    pool = covering or spans
    return min(pool, key=lambda s: abs((s[0] + s[1]) / 2 - time))[2]


def split_by_speaker(
    words: list[dict],
    spans: list[SpeakerSpan],
    min_turn_words: int = 2,
) -> list[tuple[str | None, list[dict]]]:
    """Découpe une liste de mots horodatés en tours de parole.

    Fonction **pure**. Rend `[(locuteur, mots), ...]` dans l'ordre. Sans étendues
    (diarisation absente ou en échec), ou si toutes portent le même locuteur, on
    rend un unique tour — le chemin d'avant, sans surcoût.

    `min_turn_words` absorbe le bruit du clustering : une fenêtre isolée qui
    change d'avis au milieu d'une phrase produirait un tour d'un mot, c'est-à-dire
    du confetti à l'écran. En dessous du seuil, le tour est refondu dans son
    voisin. Le prix est symétrique et assumé : un vrai « oui » d'une syllabe glissé
    dans la phrase de l'autre reste attribué à l'autre.
    """
    if not words:
        return []
    # Une fenêtre sans étiquette (trop courte, inférence en échec) ne doit pas
    # ouvrir un tour anonyme au milieu de la phrase : on l'ignore et ses mots
    # reviennent à la fenêtre étiquetée la plus proche.
    spans = [s for s in spans if s[2] is not None]
    labels = {s[2] for s in spans}
    if len(labels) <= 1:
        return [(next(iter(labels), None), words)]

    # Un mot sans horodatage hérite du précédent : le moteur en rend parfois sur
    # de la ponctuation recollée, et les laisser tomber trouerait la phrase.
    assigned: list[str | None] = []
    for word in words:
        t = _word_time(word)
        assigned.append(_span_for(t, spans) if t is not None
                        else (assigned[-1] if assigned else spans[0][2]))

    assigned = _snap_to_pauses(words, assigned)

    runs: list[tuple[str | None, list[dict]]] = []
    for label, word in zip(assigned, words):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(word)
        else:
            runs.append((label, [word]))

    return _merge_short_runs(runs, min_turn_words)


def _snap_to_pauses(words: list[dict], assigned: list[str | None],
                    radius: int = 2, min_pause_s: float = 0.15) -> list[str | None]:
    """Recale les changements de locuteur sur le silence le plus proche.

    Les fenêtres se recouvrent : la frontière qu'elles donnent est floue à un
    demi-pas près, soit un mot ou deux mal attribués à chaque tour (« … que oui
    **moi** | non pas du tout »). Mais on sait où chercher mieux : on se relaie
    dans un blanc. On déplace donc la frontière, à `radius` mots près, vers le
    plus grand écart entre deux mots consécutifs — **à condition** qu'il y ait un
    vrai blanc (`min_pause_s`). Sans pause franche à proximité, on ne sait rien
    de mieux que les embeddings et on laisse la frontière où ils l'ont mise :
    déplacer sur un écart nul reviendrait à décider au hasard.

    Fonction pure, tolérante aux mots sans horodatage (ils ne peuvent pas porter
    de frontière, l'écart y est inconnu).
    """
    def gap(i: int) -> float:
        prev_end, start = words[i - 1].get("end"), words[i].get("start")
        if prev_end is None or start is None:
            return float("-inf")
        return start - prev_end

    out = list(assigned)
    floor = 1  # une frontière ne peut pas repasser derrière la précédente
    for i in range(1, len(out)):
        if out[i] == out[i - 1]:
            continue
        lo, hi = max(floor, i - radius), min(len(out) - 1, i + radius)
        best = max(range(lo, hi + 1), key=gap)
        if best != i and gap(best) >= min_pause_s and gap(best) > gap(i):
            new_label, old_label = out[i], out[i - 1]
            for k in range(min(best, i), max(best, i)):
                out[k] = new_label if best < i else old_label
        floor = max(best, i) + 1
    return out


def _merge_short_runs(
    runs: list[tuple[str | None, list[dict]]], min_turn_words: int
) -> list[tuple[str | None, list[dict]]]:
    """Refond les tours trop courts dans leur voisin, jusqu'à stabilité."""
    while len(runs) > 1:
        idx = next((i for i, (_, w) in enumerate(runs) if len(w) < min_turn_words), None)
        if idx is None:
            break
        # Le premier tour n'a pas de précédent : il fusionne vers l'aval.
        target = idx - 1 if idx > 0 else 1
        merged = sorted([idx, target])
        keep, drop = merged[0], merged[1]
        # L'étiquette conservée est celle du tour le plus long : fondre trois
        # mots dans dix ne doit pas donner la voix des trois à l'ensemble.
        label = max((runs[keep], runs[drop]), key=lambda r: len(r[1]))[0]
        runs[keep] = (label, runs[keep][1] + runs[drop][1])
        runs.pop(drop)
        # Une fusion peut mettre deux tours du même locuteur côte à côte (le
        # parasite retiré, ses voisins se rejoignent) : les recoller, sinon on
        # afficherait deux bulles consécutives pour une seule prise de parole.
        runs = _coalesce(runs)
    return runs


def _coalesce(
    runs: list[tuple[str | None, list[dict]]]
) -> list[tuple[str | None, list[dict]]]:
    out: list[tuple[str | None, list[dict]]] = []
    for label, words in runs:
        if out and out[-1][0] == label:
            out[-1][1].extend(words)
        else:
            out.append((label, list(words)))
    return out
