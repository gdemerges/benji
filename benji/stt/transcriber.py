import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue

import numpy as np

from benji.config import STTConfig

log = logging.getLogger(__name__)
from benji.history import TranscriptionHistory
from benji.recording import RecordingConsent
from benji.stats import SessionStats
from benji.stt.backend import build_backend, build_final_backend
from benji.stt.diarization import build_tagger, label_windows, split_by_speaker
from benji.stt.lexicon import add_term, apply_lexicon, compile_terms, load_terms
from benji.stt.postprocessing import is_hallucination, postprocess_text


def drop_context_words(words: list[dict], context_s: float) -> list[dict]:
    """Retire les mots qui tombent dans le contexte placé en tête du tampon.

    Après une coupure forcée, le VAD décode le segment suivant avec la fin du
    précédent devant lui (cf. `VADConfig.cut_context_ms`) : ces mots-là sont
    déjà dans le segment d'avant. Un mot appartient au contexte si son milieu y
    tombe. Un mot sans horodatage suit le mot horodaté le plus proche avant lui
    (ou, en tête, le premier après) : le moteur en rend parfois sur de la
    ponctuation recollée. Pure.
    """
    if context_s <= 0 or not words:
        return words
    decisions: list[bool | None] = []
    for w in words:
        known = [t for t in (w.get("start"), w.get("end")) if t is not None]
        decisions.append(sum(known) / len(known) >= context_s if known else None)
    first_known = next((d for d in decisions if d is not None), True)
    kept, last = [], first_known
    for w, d in zip(words, decisions):
        if d is not None:
            last = d
        if last:
            kept.append(w)
    return kept


class Transcriber:
    def __init__(
        self,
        transcribe_queue: Queue,
        display_queue: Queue,
        config: STTConfig = None,
        stats: SessionStats | None = None,
        sample_rate: int = 16000,
    ):
        self.transcribe_queue = transcribe_queue
        self.display_queue = display_queue
        self.config = config or STTConfig()
        self.history = TranscriptionHistory()
        # Toute écriture passe par le portillon : le direct s'affiche toujours,
        # le disque attend l'accord de l'utilisateur (cf. benji/recording.py).
        # `self.history` reste le magasin — c'est lui qu'on relit pour l'export.
        self.consent = RecordingConsent(
            self.history, armed=not self.config.confirm_before_saving
        )
        self.stats = stats
        self.sample_rate = sample_rate

        # État de streaming par segment (LocalAgreement-2 sur les passes
        # partielles). On re-décode le tampon **entier** à chaque passe et on
        # fige le préfixe sur lequel deux passes successives sont d'accord.
        #
        # Une version antérieure ne décodait que la queue non confirmée, en
        # réinjectant le préfixe figé via `initial_prompt` : c'était un
        # contournement du coût de Whisper. Parakeet décode 8 s en ~150 ms et
        # n'accepte aucun prompt ; décoder tout est donc à la fois plus simple et
        # plus juste — le modèle voit l'énoncé complet au lieu d'une tranche
        # amputée de son début.
        self._committed_words: list[dict] = []
        self._prev_words_norm: list[str] = []

        # Async LLM correction: raw finals are shown immediately, then corrected
        # off-thread (see _corrector_loop) so the STT loop never blocks on the LLM.
        # Each final carries a monotonic seq so the overlay can replace the right
        # segment (and ignore a correction whose segment is no longer on screen).
        self._segment_seq: int = 0
        self._correction_queue: Queue | None = None
        self._corrector_thread: threading.Thread | None = None

        # Diarisation optionnelle (pitch ou pyannote). Le tagger n'a besoin que
        # de l'audio : on le lance *en parallèle* du décodage final au lieu de
        # l'enchaîner après (cf. _run_segment). Un seul worker — le tagger porte
        # un état de clustering et ne doit jamais tourner sur deux segments à la
        # fois ; le join a lieu avant de rendre la main, donc pas de recouvrement.
        self._diarizer_pool: ThreadPoolExecutor | None = None
        self.tagger = (
            build_tagger(
                self.config.diarization_backend,
                max_speakers=self.config.diarization_max_speakers,
            )
            if self.config.diarization
            else None
        )

        # Deux moteurs, un par type de passe (cf. benji/stt/backend.py) : le
        # direct privilégie la latence, le final la garantie de langue.
        self.backend = build_backend(self.config.model)
        self.final_backend = build_final_backend(
            self.config.final_engine,
            self.config.final_model_size,
            self.config.language,
            fast=self.backend,
        ) or self.backend
        log.info(
            "Moteurs prêts — partielles : %s · finale : %s",
            self.backend.name, self.final_backend.name,
        )

        # Glossaire utilisateur, appliqué au texte final (cf. stt/lexicon.py).
        # Chargé une fois : le fichier est édité depuis les Préférences, et un
        # rechargement à chaud ferait relire le disque à chaque segment.
        self._lexicon = compile_terms(load_terms()) if self.config.glossary else []

    def learn_term(self, term: str) -> bool:
        """Apprend un terme sans relire le disque à chaque segment.

        Le glossaire est compilé une fois au démarrage, exprès : le relire par
        segment mettrait le disque sur le chemin chaud. Un terme appris en
        réunion doit pourtant valoir **tout de suite**, sinon la correction
        n'arrive qu'au prochain lancement — c'est-à-dire trop tard pour la
        réunion en cours, la seule où l'on parle de ce client-là.

        La liste est **rebindée**, pas mutée : le thread STT itère dessus
        pendant ce temps, il verra l'ancienne ou la nouvelle, jamais un état
        intermédiaire.
        """
        if not add_term(term):
            return False
        self._lexicon = compile_terms(load_terms())
        return True

    def warmup(self) -> None:
        """Décodage à blanc pour amortir la compilation des noyaux Metal.

        La liaison du modèle au stream MLX, elle, est faite par le backend au
        chargement (`mx.eval` des poids) : ce préchauffage-ci ne sert plus qu'à
        payer la compilation avant la première vraie phrase, pas après.

        Les backends à chargement paresseux (`eager_warmup = False`, cf. Whisper
        et l'hybride) sont **sautés** : les préchauffer chargerait au démarrage
        les poids que leur paresse existe pour éviter — jusqu'à 1,5 Go qu'une
        réunion française entière peut ne jamais réclamer.
        """
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        for backend in {id(self.backend): self.backend,
                        id(self.final_backend): self.final_backend}.values():
            if not getattr(backend, "eager_warmup", True):
                log.info("Préchauffage de %s différé (chargement paresseux)", backend.name)
                continue
            try:
                t0 = time.monotonic()
                for _ in backend.transcribe(silence):
                    pass
                log.info("Préchauffage de %s en %.0f ms",
                         backend.name, (time.monotonic() - t0) * 1000)
            except Exception as e:
                log.warning("Préchauffage de %s ignoré : %s", backend.name, e)

    def _reset_partial_state(self) -> None:
        self._committed_words = []
        self._prev_words_norm = []

    @staticmethod
    def _norm(text: str) -> str:
        """Normalize a word for cross-partial agreement comparison.

        Le moteur change parfois une capitale ou recolle une ponctuation d'une
        passe à l'autre ; on les ignore pour que l'accord porte sur le contenu
        lexical et pas sur ces variations.
        """
        return text.strip().lower().strip(".,;:!?\"'«»()[]")

    @staticmethod
    def _common_prefix_len(a: list[str], b: list[str]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Normalise en crête les tampons faibles pour présenter un niveau constant.

        Boost-only: only quiet segments are scaled up; loud segments pass through.
        Avoids amplifying near-silence (peak < 0.01) which would just amplify noise.
        """
        target = self.config.agc_target_peak
        if target <= 0.0 or audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak < 0.01 or peak >= self.config.agc_min_peak:
            return audio
        gain = min(target / peak, 8.0)  # Cap gain at 8x to limit noise blow-up
        return (audio * gain).astype(np.float32, copy=False)

    def _run_partial(self, audio: np.ndarray, context_s: float = 0.0) -> None:
        """Re-décode le tampon entier et stabilise l'affichage par LocalAgreement-2.

        Le préfixe sur lequel deux passes successives tombent d'accord est
        considéré comme acquis et ne bougera plus à l'écran ; la queue est
        affichée telle quelle, au titre de « meilleure hypothèse du moment ».
        C'est ce qui évite que le texte déjà lu se réécrive sous les yeux.

        Le figeage est **monotone** : on n'annule jamais un mot déjà acquis, même
        si une passe ultérieure change d'avis — reprendre un mot affiché est plus
        déroutant que de laisser une petite erreur que la passe finale corrigera.
        """
        start_t = time.monotonic()
        audio = self._apply_agc(audio)
        if len(audio) < int(0.3 * self.sample_rate):
            return  # trop court pour valoir une passe

        words = drop_context_words(list(self.backend.transcribe(audio)), context_s)
        if not words:
            return

        # `segment_start` annonce un **nouvel énoncé**, pas un rafraîchissement :
        # c'est lui qui périme les corrections LLM tardives et remet à zéro la
        # pile de tours de parole de l'overlay. Le répéter à chaque passe (ce que
        # faisait le rendu mot à mot) n'avait de sens que parce qu'il précédait
        # un redessin complet.
        first_pass = not self._prev_words_norm and not self._committed_words

        norm = [self._norm(w["text"]) for w in words]
        agreed = self._common_prefix_len(norm, self._prev_words_norm)
        if agreed > len(self._committed_words):
            self._committed_words = words[:agreed]
        self._prev_words_norm = norm

        # Redessine l'instantané : préfixe acquis, puis meilleure hypothèse.
        #
        # **Un seul message par passe, pas un par mot.** `display_queue` est
        # bornée (10) et les `put` sont bloquants : publier mot à mot faisait
        # attendre le thread STT sur le tick de 16 ms de `DisplayBus` — 35 ms
        # mesurés sur 20 mots, 53 sur 37, sur une passe qui n'en coûte que 150 de
        # décodage. Côté Qt, chaque mot déclenchait un `setText` + `adjustSize` +
        # `move` de l'overlay, soit ~10 ms de mise en page pour un état
        # intermédiaire que personne n'a le temps de lire. Le consommateur
        # réaffiche de toute façon la phrase entière : un message suffit.
        if first_pass:
            self.display_queue.put({"type": "segment_start"})
        self.display_queue.put({
            "type": "partial",
            "words": [
                {"text": w["text"], "start": w.get("start"), "end": w.get("end")}
                for w in self._committed_words + words[len(self._committed_words):]
            ],
        })

        if self.stats is not None:
            latency_ms = (time.monotonic() - start_t) * 1000
            self.stats.record_segment(
                len(audio) / self.sample_rate, latency_ms, is_final=False
            )

    def _run_segment(self, audio: np.ndarray, is_final: bool, context_s: float = 0.0):
        if not is_final:
            self._run_partial(audio, context_s)
            return

        start_t = time.monotonic()
        audio = self._apply_agc(audio)

        # Diarisation lancée AVANT le décodage : elle ne dépend que de l'audio.
        # Enchaînée après, son coût (embedding pyannote) s'ajoutait tel quel au
        # délai avant affichage du texte final ; en parallèle, il est absorbé par
        # le décodage qui tourne de toute façon.
        speaker_future = None
        if self.tagger is not None:
            speaker_future = self._ensure_diarizer_pool().submit(
                self._label_spans, audio, self.sample_rate
            )

        self.display_queue.put({"type": "segment_start"})
        # Les horodatages restent relatifs au tampon *avec* contexte : c'est la
        # même base de temps que les étendues de diarisation, calculées sur ce
        # même tampon.
        words = drop_context_words(list(self.final_backend.transcribe(audio)), context_s)
        if words:
            # Le moteur hybride ne streame pas (il faut tout le texte pour juger
            # de sa langue) : il n'y a rien à égrener, et les mots sont déjà à
            # l'écran, posés par les partielles.
            self.display_queue.put({
                "type": "partial",
                "words": [
                    {"text": w["text"], "start": w.get("start"), "end": w.get("end")}
                    for w in words
                ],
            })

        if not words:
            if speaker_future is not None:
                speaker_future.cancel()
            self._reset_partial_state()
            return

        # Étiquettes de locuteur (best-effort), fenêtre par fenêtre. Un segment
        # VAD n'est pas un tour de parole : quand deux personnes s'enchaînent
        # sans vraie pause, elles tiennent dans le même tampon. On recoupe donc
        # les mots aux frontières de locuteur plutôt que de fondre les deux voix
        # dans une phrase unique. Cas courant : une seule étendue, un seul tour,
        # exactement le chemin d'avant.
        turns = split_by_speaker(
            words,
            self._await_spans(speaker_future),
            self.config.diarization_min_turn_words,
        )

        # Chaque tour est post-traité pour lui-même : la ponctuation et le
        # glossaire n'ont pas à voir la phrase de l'autre, et une hallucination
        # localisée ne doit emporter que son tour.
        finals: list[tuple[str | None, str]] = []
        for speaker, turn_words in turns:
            text = postprocess_text(
                " ".join(w["text"] for w in turn_words), language=self.config.language
            )
            # Glossaire : seulement sur le final. Le partiel est jeté de toute façon,
            # et voir un mot se réécrire sous les yeux est plus déroutant qu'une
            # coquille passagère.
            text = apply_lexicon(text, self._lexicon)
            if text and not is_hallucination(text):
                finals.append((speaker, text))

        if not finals:
            # Tell the overlay to drop the streamed (hallucinated) words instead
            # of leaving them on screen.
            self.display_queue.put({"type": "final_text", "text": "", "drop": True})
            self._reset_partial_state()
            return

        for speaker, text in finals:
            # L'étiquette voyage comme champ structuré, jamais collée dans le
            # texte, pour que l'UI puisse la colorer par locuteur.
            if self.config.llm_correction:
                # Show the raw transcription immediately, then correct it off-thread
                # and emit a replacement once ready — the STT loop never blocks on the
                # LLM. History is written by the corrector (stores the corrected text).
                self._segment_seq += 1
                self._emit_final(text, speaker, seq=self._segment_seq)
                self._enqueue_correction(text, speaker, self._segment_seq)
            else:
                # Replace the streamed (raw) overlay text with the post-processed one.
                self._emit_final(text, speaker)
                # DEBUG et pas INFO : le log est persisté sur disque et joint aux
                # rapports de bug — le contenu transcrit ne doit pas y fuiter.
                log.debug('%s"%s"', f"[{speaker}] " if speaker else "", text)
                self.consent.add(text, speaker=speaker)

        # Stats
        if self.stats is not None:
            latency_ms = (time.monotonic() - start_t) * 1000
            audio_seconds = len(audio) / self.sample_rate
            self.stats.record_segment(audio_seconds, latency_ms)

        # Final closes the segment — partial state resets for the next utterance.
        self._reset_partial_state()

    def _ensure_diarizer_pool(self) -> ThreadPoolExecutor:
        """Pool à un worker, créé au premier segment étiqueté.

        Paresseux : le tagger peut être posé après la construction (tests, bascule
        de backend), et un pool n'a aucune raison d'exister sans diarisation.
        """
        if self._diarizer_pool is None:
            self._diarizer_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="STT-diarizer"
            )
        return self._diarizer_pool

    def _label_spans(self, audio, sample_rate: int) -> list:
        """Étendues de locuteur du tampon, best-effort : jamais fatales.

        Le coût est celui d'un embedding par fenêtre au lieu d'un par segment,
        payé sur le pool de diarisation pendant que le décodage tourne — donc
        absorbé, tant qu'il reste sous la durée du décodage final.
        """
        try:
            return label_windows(
                self.tagger, audio, sample_rate,
                window_s=self.config.diarization_window_s,
                hop_s=self.config.diarization_hop_s,
            )
        except Exception as e:
            log.warning("Diarisation ignorée : %s", e)
            return []

    def _await_spans(self, future, timeout: float = 5.0) -> list:
        """Récupère les étendues calculées en parallèle du décodage.

        Le timeout est un garde-fou : un tagger bloqué (modèle qui télécharge,
        backend gelé) ne doit pas figer la boucle STT — on rend le segment sans
        locuteur plutôt que d'arrêter la transcription.
        """
        if future is None:
            return []
        try:
            return future.result(timeout=timeout) or []
        except Exception as e:
            log.warning("Diarisation abandonnée (%s) — segment sans locuteur", e)
            return []

    def _emit_final(self, text: str, speaker: str | None, *, seq: int | None = None,
                    corrected: bool = False) -> None:
        """Push a final_text message to the overlay.

        `seq` tags the segment so an async correction can be matched back to it;
        `corrected` marks the replacement so the overlay only applies it while the
        same segment is still displayed.
        """
        msg: dict = {"type": "final_text", "text": text}
        if speaker:
            msg["speaker"] = speaker
        if seq is not None:
            msg["seq"] = seq
        if corrected:
            msg["corrected"] = True
        self.display_queue.put(msg)

    def _ensure_corrector(self) -> None:
        if self._corrector_thread is not None and self._corrector_thread.is_alive():
            return
        self._correction_queue = Queue(maxsize=8)
        self._corrector_thread = threading.Thread(
            target=self._corrector_loop, daemon=True, name="STT-corrector"
        )
        self._corrector_thread.start()

    def _enqueue_correction(self, text: str, speaker: str | None, seq: int) -> None:
        self._ensure_corrector()
        try:
            self._correction_queue.put_nowait((seq, text, speaker))
        except Full:
            # Corrector saturated: keep the raw text (already displayed) and
            # persist it now so history has exactly one entry for this segment.
            log.warning("LLM corrector saturated; kept raw text")
            self.consent.add(text, speaker=speaker)

    def _corrector_loop(self) -> None:
        """Background worker: correct queued finals and emit replacements.

        Persists the corrected (or unchanged) text to history so there is exactly
        one entry per segment — the STT loop deliberately skips history.add when
        correction is enabled.
        """
        from benji.llm.corrector import correct
        while True:
            item = self._correction_queue.get()
            if item is None:
                break
            seq, text, speaker = item
            try:
                corrected = correct(text, language=self.config.language)
            except Exception as e:
                log.warning("LLM correction skipped: %s", e)
                corrected = text
            self.consent.add(corrected, speaker=speaker)
            log.debug('%s"%s"', f"[{speaker}] " if speaker else "", corrected)
            self._emit_final(corrected, speaker, seq=seq, corrected=True)

    def run(self):
        log.info("Transcription started (incremental streaming)")
        while True:
            item = self.transcribe_queue.get()
            if item is None:
                break
            audio = item["audio"]
            is_final = item["is_final"]
            if not is_final and not self.transcribe_queue.empty():
                continue
            try:
                self._run_segment(audio, is_final, item.get("context_s", 0.0))
            except Exception:
                # A single bad segment should not kill the STT loop.
                log.exception("STT segment failed (final=%s, %.2fs); skipping",
                              is_final, len(audio) / self.sample_rate)
                if self.stats is not None:
                    self.stats.record_drop("stt_error")
                # Ensure the overlay doesn't keep partial words from a failed segment.
                try:
                    self.display_queue.put({"type": "final_text", "text": "", "drop": True})
                except Exception:
                    pass
                # Reset streaming state so the next segment starts clean.
                self._reset_partial_state()
        if self._diarizer_pool is not None:
            self._diarizer_pool.shutdown(wait=False)
        self._stop_corrector()
        log.info("Transcription stopped")

    def _stop_corrector(self) -> None:
        """Arrête le correcteur sans perdre ce qu'il n'a pas encore traité.

        Avec `llm_correction`, c'est le correcteur qui écrit l'historique. Son
        fil n'était jamais arrêté : à la fermeture, les segments en attente de
        correction mouraient avec lui, sans jamais être conservés. On les verse
        tels quels (le texte brut, déjà affiché), puis on arrête le fil — celui
        qu'il corrige en ce moment a encore un court délai pour aboutir.
        """
        if self._correction_queue is None:
            return
        while True:
            try:
                item = self._correction_queue.get_nowait()
            except Empty:
                break
            if item is None:
                continue
            _seq, text, speaker = item
            self.consent.add(text, speaker=speaker)
        with contextlib.suppress(Full):
            self._correction_queue.put_nowait(None)
        if self._corrector_thread is not None:
            self._corrector_thread.join(timeout=2.0)
