import hashlib
import logging
import os
import threading
from collections import deque
from queue import Empty, Full, Queue

import numpy as np
import onnxruntime as ort

from benji.config import AudioConfig, VADConfig

log = logging.getLogger(__name__)

SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SILERO_ONNX_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
_ALLOWED_HOSTS = {"github.com", "raw.githubusercontent.com", "objects.githubusercontent.com"}
_MAX_REDIRECTS = 3
# Largeur (en morceaux de 32 ms) du lissage de confiance avant une coupure forcée.
_CUT_SMOOTHING_CHUNKS = 3


class SileroVADOnnx:
    """Silero VAD using ONNX runtime (no PyTorch dependency)."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        # Single thread for minimal latency (VAD is very fast ~1-2ms)
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        # Note: graph optimization disabled - incompatible with Silero VAD LSTM layers
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)  # 64 samples context for 16kHz
        self._sr = np.array(16000, dtype=np.int64)

    def reset_state(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def __call__(self, audio_chunk: np.ndarray) -> float:
        # Prepend context to audio chunk
        audio_with_context = np.concatenate([self._context, audio_chunk])
        # Update context for next call
        self._context = audio_chunk[-64:]
        # Run inference
        ort_inputs = {
            "input": audio_with_context[np.newaxis, :].astype(np.float32),
            "state": self._state,
            "sr": self._sr,
        }
        out, new_state = self.session.run(None, ort_inputs)
        self._state = new_state
        return float(out[0][0])


def _verify_sha256(path: str, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _download_model() -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "benji")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "silero_vad.onnx")

    if os.path.exists(model_path):
        if not _verify_sha256(model_path, SILERO_ONNX_SHA256):
            log.warning("Cached model failed integrity check, re-downloading...")
            os.remove(model_path)
        else:
            return model_path

    log.info("Downloading Silero VAD ONNX model...")
    from urllib.parse import urlparse

    import httpx

    def _check_host(url: str) -> None:
        host = urlparse(url).hostname or ""
        if host not in _ALLOWED_HOSTS:
            raise ValueError(f"[VAD] Redirect to untrusted host blocked: {host}")

    with httpx.Client(max_redirects=_MAX_REDIRECTS, event_hooks={
        "request": [lambda req: _check_host(str(req.url))]
    }) as client:
        with client.stream("GET", SILERO_ONNX_URL, follow_redirects=True) as r:
            r.raise_for_status()
            tmp_path = model_path + ".tmp"
            try:
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                if not _verify_sha256(tmp_path, SILERO_ONNX_SHA256):
                    raise ValueError("[VAD] Downloaded model failed integrity check")
                os.replace(tmp_path, model_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    return model_path


class VADProcessor:
    def __init__(
        self,
        audio_queue: Queue,
        transcribe_queue: Queue,
        audio_config: AudioConfig = None,
        vad_config: VADConfig = None,
        display_queue: Queue = None,
        stats=None,
    ):
        self.audio_queue = audio_queue
        self.transcribe_queue = transcribe_queue
        self.display_queue = display_queue
        self.audio_config = audio_config or AudioConfig()
        self.config = vad_config or VADConfig()
        self.sample_rate = self.audio_config.sample_rate
        self.stats = stats

        # Load Silero VAD (ONNX)
        model_path = _download_model()
        self.model = SileroVADOnnx(model_path)
        log.info("Silero VAD loaded (ONNX)")

        # State
        self.is_speaking = False
        # Posé par le thread Qt à la pause micro, traité sur le thread VAD (seul
        # propriétaire de l'état ci-dessous) : cf. request_flush().
        self._flush_requested = threading.Event()
        self.speech_buffer: list[np.ndarray] = []
        # Confiance VAD de chaque morceau de `speech_buffer`, au même indice :
        # c'est là qu'une coupure forcée cherche le moment le plus calme.
        self._speech_conf: list[float] = []
        # Queue du segment précédent, à décoder devant le suivant quand une
        # coupure forcée a tranché une parole continue (cf. `_cut_long_segment`).
        self._context: np.ndarray | None = None
        self.silence_chunks = 0
        self.pre_speech_buffer: list[np.ndarray] = []
        self.samples_since_partial = 0
        self._partial_sample_interval = int(
            self.config.partial_interval_ms / 1000 * self.sample_rate
        )

        # Adaptive threshold: rolling buffer of VAD confidence on non-speech chunks.
        # Effective threshold = max(base, p95(noise) + margin). Robust to a noisy room.
        chunk_ms_est = self.audio_config.chunk_size / self.sample_rate * 1000
        self._noise_window_size = max(
            10, int(self.config.adaptive_window_seconds * 1000 / max(chunk_ms_est, 1))
        )
        self._noise_confidences: deque[float] = deque(maxlen=self._noise_window_size)

    def _chunk_duration_ms(self, chunk: np.ndarray) -> float:
        return len(chunk) / self.sample_rate * 1000

    def _effective_threshold(self) -> float:
        """Lift the speech threshold above the noise floor when the room is noisy.

        Uses the 95th percentile of recent non-speech VAD confidences as the
        noise estimate, then adds a margin. Falls back to the static threshold
        until enough samples have been seen.
        """
        base = self.config.speech_threshold
        if not self.config.adaptive_threshold or len(self._noise_confidences) < 20:
            return base
        noise_p95 = float(np.quantile(self._noise_confidences, 0.95))
        return max(base, min(0.95, noise_p95 + self.config.adaptive_margin))

    def process_chunk(self, chunk: np.ndarray) -> None:
        confidence = self.model(chunk)

        chunk_ms = self._chunk_duration_ms(chunk)
        threshold = self._effective_threshold()
        # Hystérésis : une fois la parole ouverte, il faut descendre nettement
        # sous le seuil pour la clore — sinon les syllabes faibles d'une fin de
        # phrase comptent comme du silence (cf. VADConfig.speech_end_hysteresis).
        if self.is_speaking:
            threshold = max(0.05, threshold - self.config.speech_end_hysteresis)

        if confidence >= threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_buffer = list(self.pre_speech_buffer)
                # Le pré-roll n'est jamais candidat à une coupure forcée.
                self._speech_conf = [1.0] * len(self.speech_buffer)
                self.samples_since_partial = 0
                log.debug("Speech started")
                if self.display_queue:
                    self.display_queue.put({"type": "vad_status", "speaking": True})
            self.speech_buffer.append(chunk)
            self._speech_conf.append(confidence)
            self.samples_since_partial += len(chunk)
            self.silence_chunks = 0
        else:
            if self.is_speaking:
                self.speech_buffer.append(chunk)
                self._speech_conf.append(confidence)
                self.samples_since_partial += len(chunk)
                self.silence_chunks += 1
                silence_ms = self.silence_chunks * chunk_ms

                if silence_ms >= self.config.silence_duration_ms:
                    self._flush_segment(is_final=True)
                    return
            else:
                # Idle: update noise floor estimate and ring-buffer pre-speech audio.
                self._noise_confidences.append(confidence)
                self.pre_speech_buffer.append(chunk)
                max_pre = int(self.config.pre_speech_pad_ms / chunk_ms)
                if len(self.pre_speech_buffer) > max_pre:
                    self.pre_speech_buffer.pop(0)
                return

        # Énoncé trop long : on le coupe, mais au moment le plus calme.
        total_samples = sum(len(c) for c in self.speech_buffer)
        if total_samples / self.sample_rate >= self.config.max_speech_duration_s:
            self._cut_long_segment(chunk_ms)
            return

        # Incremental partial during ongoing speech. A partial re-transcribes the
        # entire growing buffer, so a fixed interval makes total cost quadratic in
        # segment length. Back off as the buffer grows (the final pass redoes the
        # whole segment anyway); this keeps per-segment partial work ~linear while
        # staying responsive early when perceived latency matters most.
        if self.config.partial_interval_ms > 0:
            dynamic_interval = self._partial_sample_interval + int(
                self.config.partial_growth_factor * total_samples
            )
            if self.samples_since_partial >= dynamic_interval:
                self._emit_partial()

    def _message(self, audio: np.ndarray, is_final: bool) -> dict:
        """Message pour `transcribe_queue`, contexte éventuel placé devant.

        `context_s` dit au transcripteur combien de secondes de tête ne sont là
        que pour le contexte : les mots qui y tombent sont déjà dans le segment
        précédent.
        """
        if self._context is None or len(self._context) == 0:
            return {"audio": audio, "is_final": is_final}
        return {
            "audio": np.concatenate([self._context, audio]),
            "is_final": is_final,
            "context_s": len(self._context) / self.sample_rate,
        }

    def _emit_partial(self):
        if not self.speech_buffer:
            return
        audio = np.concatenate(self.speech_buffer)
        min_samples = int(self.config.min_speech_duration_ms / 1000 * self.sample_rate)
        if len(audio) < min_samples:
            return
        self.samples_since_partial = 0
        try:
            self.transcribe_queue.put(self._message(audio, is_final=False), block=False)
        except Full:
            # Transcriber is busy; it'll catch up on next partial or final
            if self.stats is not None:
                self.stats.record_drop("partial_skipped")

    def _cut_long_segment(self, chunk_ms: float) -> None:
        """Coupe un énoncé trop long au morceau le plus calme de sa fin.

        Couper net à `max_speech_duration_s` tranchait un mot en deux, que
        chaque moitié décodait de travers. On cherche plutôt, dans les
        `cut_search_ms` dernières millisecondes, le morceau de confiance VAD
        minimale et d'énergie la plus basse — une respiration, une frontière de
        mot — et on coupe juste après. Ce qui suit n'est pas perdu : il ouvre le tampon suivant, et la
        parole reste ouverte (ni remise à zéro du modèle, ni `vad_status`).
        """
        n = len(self.speech_buffer)
        search = max(1, int(self.config.cut_search_ms / chunk_ms))
        lo = max(0, n - search)
        # Score de « calme » par morceau : la confiance VAD, plus l'énergie
        # relative. La confiance seule ne suffit pas — sur une parole continue
        # Silero sature à 1,0 sur toute la fenêtre, et l'argmin tombait sur le
        # premier morceau venu, en plein mot ; entre deux mots, l'énergie baisse
        # même quand la confiance ne bouge pas.
        rms = np.array([np.sqrt(np.mean(c * c)) for c in self.speech_buffer[lo:]])
        score = np.asarray(self._speech_conf[lo:]) + rms / max(float(rms.max()), 1e-9)
        # Minimum d'une moyenne glissante, pas d'un morceau isolé : l'occlusion
        # d'un « p » ou d'un « t » fait chuter le score le temps d'un morceau, en
        # plein mot (« la pé|riode » décodé « la page »). Une vraie frontière
        # dure plus longtemps.
        k = _CUT_SMOOTHING_CHUNKS
        if len(score) >= k:
            smoothed = np.convolve(score, np.ones(k) / k, mode="valid")
            cut = lo + int(np.argmin(smoothed)) + k // 2
        else:
            cut = lo + int(np.argmin(score))

        head = np.concatenate(self.speech_buffer[:cut + 1])
        tail = self.speech_buffer[cut + 1:]
        self._enqueue_final(head, is_final=True)

        # La parole continue : le segment suivant sera décodé avec la fin de
        # celui-ci devant lui, pour que son premier mot ait du contexte.
        ctx = int(self.config.cut_context_ms / 1000 * self.sample_rate)
        self._context = head[-ctx:] if ctx > 0 else None

        self.speech_buffer = tail
        self._speech_conf = self._speech_conf[cut + 1:]
        self.samples_since_partial = sum(len(c) for c in tail)
        # Le silence en cours, s'il y en a un, se trouve dans ce qui reste.
        self.silence_chunks = min(self.silence_chunks, len(tail))

    def _enqueue_final(self, audio: np.ndarray, is_final: bool) -> None:
        min_samples = int(self.config.min_speech_duration_ms / 1000 * self.sample_rate)

        if len(audio) >= min_samples:
            duration = len(audio) / self.sample_rate
            log.debug("Speech segment: %.1fs (final=%s)", duration, is_final)
            # Finals must not be dropped silently. Block briefly to let the STT
            # catch up; if it's still saturated after the timeout, log loudly
            # and count the drop so a stalled pipeline is visible to the user.
            try:
                self.transcribe_queue.put(self._message(audio, is_final), timeout=2.0)
            except Full:
                log.warning(
                    "transcribe_queue full after 2s; dropping final segment (%.1fs). "
                    "STT can't keep up — consider a smaller model.",
                    duration,
                )
                if self.stats is not None:
                    self.stats.record_drop("transcribe_queue_full")
                if self.display_queue is not None:
                    # Clear any streamed partial words from the dropped segment.
                    self.display_queue.put({"type": "final_text", "text": "", "drop": True})

    def _flush_segment(self, is_final: bool = True):
        audio = np.concatenate(self.speech_buffer) if self.speech_buffer else np.array([], dtype=np.float32)
        self._enqueue_final(audio, is_final)

        # Vraie pause : le prochain énoncé n'a pas besoin de contexte, qui ne
        # serait que du silence.
        self._context = None
        self._speech_conf = []
        self.speech_buffer = []
        self.silence_chunks = 0
        self.is_speaking = False
        self.pre_speech_buffer = []
        self.samples_since_partial = 0
        self.model.reset_state()

        if self.display_queue:
            self.display_queue.put({"type": "vad_status", "speaking": False})

    def request_flush(self) -> None:
        """Clôt l'énoncé en cours — appelé à la pause du micro, depuis le thread Qt.

        La pause coupe le flux : plus aucun morceau n'arrive, donc aucun silence
        ne vient jamais clore la phrase commencée. Elle restait dans le tampon et,
        à la reprise, la phrase suivante lui était recollée — des minutes plus
        tard, décodées comme un seul énoncé. Ce qui a été dit avant la pause part
        en final, comme après un vrai silence.
        """
        self._flush_requested.set()

    def _handle_flush_request(self) -> None:
        if not self._flush_requested.is_set():
            return
        self._flush_requested.clear()
        if self.is_speaking:
            self._flush_segment(is_final=True)

    def run(self):
        log.info("Processing started")
        while True:
            # Attente bornée : une demande de clôture doit être servie même
            # quand plus aucun audio n'arrive — c'est le propre d'une pause.
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except Empty:
                self._handle_flush_request()
                continue
            self._handle_flush_request()
            if chunk is None:
                break
            self.process_chunk(chunk)
        log.info("Processing stopped")
