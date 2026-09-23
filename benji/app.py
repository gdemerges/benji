"""Composition root de Benji : câble configs → pipeline → threads → UI.

`main.py` reste un point d'entrée mince ; toute l'orchestration du démarrage et
de l'arrêt vit ici, découpée en phases lisibles et pilotables. Chaque phase est
une méthode : on peut en stubber une poignée (Qt, audio, chargement du modèle)
pour tester la composition sans lancer réellement l'app.

IMPORTANT macOS : la politique « accessory » doit être posée AVANT `QApplication`
(cf. main.py) — ce module suppose que c'est déjà fait au moment où `run()` crée
le `QApplication`.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from dataclasses import dataclass, field
from queue import Queue

from PySide6.QtCore import QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from benji.audio.capture import AudioCapture
from benji.audio.vad import VADProcessor
from benji.config import (
    IS_MACOS,
    AudioConfig,
    LLMConfig,
    STTConfig,
    UIConfig,
    VADConfig,
    ensure_secure_backend_url,
)
from benji.launch_mode import launch_mode
from benji.llm.providers import build_summary_provider
from benji.llm.summary_worker import SummaryWorker
from benji.queues import NotifyingQueue
from benji.stats import SessionStats
from benji.stt.transcriber import Transcriber
from benji.ui.display_bus import DisplayBus
from benji.ui.history_window import HistoryWindow
from benji.ui.live_summary_window import LiveSummaryWindow
from benji.ui.main_window import MainWindow
from benji.ui.overlay import SubtitleOverlay
from benji.ui.splash import SplashWindow
from benji.ui.tray import build_tray
from benji.ui.window_controller import WindowController

log = logging.getLogger(__name__)


@dataclass
class AppConfigs:
    """Les cinq dataclasses de config, regroupées pour l'injection/les tests."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


class BenjiApplication:
    """État + orchestration du cycle de vie de l'app (démarrage → exec → arrêt)."""

    def __init__(self, configs: AppConfigs | None = None):
        self.cfg = configs or AppConfigs()

        # Renseignés au fil des phases de run().
        self.user_settings = None
        self.session = None
        self.stats: SessionStats | None = None
        self.session_start = None

        self.audio_queue: Queue | None = None
        self.transcribe_queue: Queue | None = None
        self.display_queue: Queue | None = None
        self.capture: AudioCapture | None = None
        self.vad: VADProcessor | None = None
        self.system_capture = None
        self.mixer = None

        self.app: QApplication | None = None
        self.remote_mode = False

        self.transcriber: Transcriber | None = None
        self.remote_stt = None
        self.history = None

        self.vad_thread: threading.Thread | None = None
        self.stt_supervisor: threading.Thread | None = None
        self.remote_thread: threading.Thread | None = None
        self.stt_stopping = threading.Event()

        self.bus: DisplayBus | None = None
        self.overlay: SubtitleOverlay | None = None
        self.history_window: HistoryWindow | None = None
        self.live_summary_window: LiveSummaryWindow | None = None

        self.mode = "overlay"
        self.main_window: MainWindow | None = None
        self.controller: WindowController | None = None
        self.summary_worker: SummaryWorker | None = None
        self.live_summarizer = None
        self.meeting_titler = None
        self.tray = None
        self._shortcuts: list[QShortcut] = []  # garder les refs (sinon GC)
        self.global_hotkeys = None

    # --- orchestration ---

    def run(self) -> int:
        """Démarre l'app, entre dans la boucle Qt, puis arrête proprement."""
        self._build_configs()
        self._build_account()
        self._build_pipeline()
        self._create_qapp()

        if not self._run_onboarding():
            # L'utilisateur a fermé l'assistant : démarrer quand même donnerait
            # une app sans micro ni modèle, qui ne transcrit rien et n'explique
            # pas pourquoi. On sort avant d'ouvrir quoi que ce soit.
            log.info("Premier lancement abandonné — arrêt.")
            return 0

        splash = self._show_splash()
        try:
            self._load_transcriber(splash)
        except Exception:
            splash.close()
            raise
        self._start_stt()
        if self.mixer is not None:
            self.mixer.start()
        self.capture.start()
        splash.close()

        self._install_excepthooks()
        self._build_display()
        self._build_windows()
        self._build_tray_and_shortcuts()
        self._install_signal_handlers()

        log.info("Ctrl+Shift+H: history · Ctrl+Shift+S: live summary")
        exit_code = self.app.exec()

        self.shutdown()
        return exit_code

    # --- phases ---

    def _build_configs(self) -> None:
        # Préférences persistées (QSettings) : appliquées AVANT le chargement du
        # modèle pour que langue/taille de modèle prennent effet.
        from benji.settings import UserSettings

        self.user_settings = UserSettings()
        self.user_settings.hydrate(
            stt=self.cfg.stt, ui=self.cfg.ui, llm=self.cfg.llm, audio=self.cfg.audio
        )
        # Point de passage unique : session, providers, STT distant et billing
        # lisent tous cfg.llm.backend_url — valider ici couvre tout le monde.
        ensure_secure_backend_url(self.cfg.llm.backend_url)

    def _build_account(self) -> None:
        # Compte Benji : si une session est enregistrée, on injecte son access
        # token pour les appels backend. L'abonnement suit le compte, pas le poste.
        from benji.account import build_session

        self.session = build_session(self.cfg.llm.backend_url)
        token = self.session.access_token()
        if token:
            self.cfg.llm.backend_token = token

    def _build_pipeline(self) -> None:
        # Calculé ici (et pas dans _create_qapp) : le choix local/remote pilote
        # la construction du pipeline, avant même la création du QApplication.
        self.remote_mode = self.cfg.stt.stt_provider == "remote"

        self.stats = SessionStats()
        self.session_start = self.stats.session_start

        self.audio_queue = Queue(maxsize=100)
        self.transcribe_queue = Queue(maxsize=3)
        # Prévient le bus d'affichage au lieu d'être sondée (cf. benji/queues.py).
        self.display_queue = NotifyingQueue(maxsize=10)

        # Audio système : le micro alimente le mixeur, qui publie le mélange
        # dans audio_queue. Désactivé (ou boucle indisponible), le micro écrit
        # directement dans audio_queue — chemin historique, inchangé.
        capture_sink = self.audio_queue
        if self.cfg.audio.system_audio:
            self._build_system_audio()
            if self.mixer is not None:
                capture_sink = self.mixer.mic_queue

        self.capture = AudioCapture(capture_sink, self.cfg.audio, stats=self.stats)
        if not self.remote_mode:
            # En mode remote le VAD n'est jamais démarré : inutile de charger
            # le modèle Silero ONNX (fait dans VADProcessor.__init__).
            self.vad = VADProcessor(
                self.audio_queue, self.transcribe_queue, self.cfg.audio, self.cfg.vad,
                self.display_queue, stats=self.stats,
            )
        log.info("Starting...")

    def _build_system_audio(self) -> None:
        """Arme la capture système + le mixeur, ou renonce proprement.

        Toute défaillance (pilote absent, périphérique refusé) est non fatale :
        Benji retombe sur le micro seul. Un pilote de boucle est installé par
        l'utilisateur, il peut disparaître d'une session à l'autre — ça ne doit
        jamais empêcher l'app de démarrer.
        """
        from benji.audio.loopback import select_loopback
        from benji.audio.system_capture import AudioMixer, SystemAudioCapture

        try:
            import sounddevice as sd

            devices = list(sd.query_devices())
        except Exception as e:
            log.warning("Audio système : énumération des périphériques impossible (%s)", e)
            return

        device = select_loopback(devices, self.cfg.audio.system_audio_device)
        if device is None:
            log.warning(
                "Audio système activé mais aucun périphérique de boucle trouvé — "
                "micro seul. Installer BlackHole puis router la sortie."
            )
            return

        system = SystemAudioCapture(device.name, sample_rate=self.cfg.audio.sample_rate)
        if not system.start():
            return

        self.system_capture = system
        self.mixer = AudioMixer(
            mic_queue=Queue(maxsize=100),
            audio_queue=self.audio_queue,
            system=system,
            system_gain=self.cfg.audio.system_audio_gain,
            stats=self.stats,
        )

    def _create_qapp(self) -> None:
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Benji")

    def _run_onboarding(self) -> bool:
        """Assistant de premier lancement. False = l'utilisateur a renoncé.

        Placé **après** `_create_qapp` (il faut Qt) et **avant** le splash : il
        télécharge les poids que le splash attendrait sinon en silence, et pose
        l'autorisation micro avant que la capture ne l'exige.

        Le mode remote n'a ni modèle local ni assistant à montrer : la
        transcription se fait côté backend.
        """
        from benji import onboarding

        if self.remote_mode or not onboarding.needs_onboarding():
            return True
        from PySide6.QtWidgets import QDialog

        from benji.ui.onboarding_window import OnboardingWindow

        return OnboardingWindow(session=self.session).exec() == QDialog.DialogCode.Accepted

    def _show_splash(self) -> SplashWindow:
        # Charge le modèle sur un thread de fond pour que l'UI reste réactive et
        # que l'utilisateur voie une progression au lieu d'un process figé.
        splash = SplashWindow()
        splash.set_status(self._loading_message())
        splash.show()
        self.app.processEvents()
        return splash

    def _loading_message(self) -> str:
        if self.remote_mode:
            return "Connexion au service de transcription…"
        return "Chargement du modèle Parakeet…"

    def _load_transcriber(self, splash: SplashWindow) -> None:
        """Charge le modèle **sur le thread principal**. Ce n'est pas un détail.

        MLX lie les tableaux au stream du thread qui les évalue en premier. Chargé
        depuis un thread de fond éphémère — ce que faisait Benji du temps de
        Whisper, pour ne pas figer le splash — Parakeet devient définitivement
        inutilisable dès la mort de ce thread : chaque inférence lève « There is
        no Stream(gpu, N) in current thread », depuis n'importe quel thread, et
        aucun `mx.new_stream()` ne le répare.

        Le prix est un blocage de l'UI pendant le chargement, mesuré à ~0,7 s
        depuis le cache disque.
        """
        if self.remote_mode:
            # Transcription côté backend : pas de modèle local, pas de VAD. Le micro
            # est streamé au backend, dont les events alimentent display_queue.
            from benji.history import TranscriptionHistory
            from benji.stt.remote import build_remote_stt_client

            self.history = TranscriptionHistory()
            self.remote_stt = build_remote_stt_client(
                self.audio_queue, self.display_queue, self.history,
                self.cfg.stt, self.cfg.llm, sample_rate=self.cfg.audio.sample_rate,
                # Rafraîchi à chaque (re)connexion : l'access token expire en 15 min.
                token_provider=self.session.access_token if self.session else None,
            )
            splash.set_status("Démarrage de la capture audio…")
            self.app.processEvents()
            return

        self.transcriber = Transcriber(
            self.transcribe_queue, self.display_queue, self.cfg.stt,
            stats=self.stats, sample_rate=self.cfg.audio.sample_rate,
        )
        splash.set_status("Préchauffage du modèle…")
        self.app.processEvents()
        self.transcriber.warmup()
        self.history = self.transcriber.history

        splash.set_status("Démarrage de la capture audio…")
        self.app.processEvents()

    def _start_stt(self) -> None:
        if self.remote_mode:
            self.remote_thread = threading.Thread(
                target=self.remote_stt.run, daemon=True, name="RemoteSTT"
            )
            self.remote_thread.start()
            return

        self.vad_thread = threading.Thread(target=self.vad.run, daemon=True, name="VAD")
        self.stt_supervisor = threading.Thread(
            target=self._stt_supervisor_loop, daemon=True, name="STT-supervisor"
        )
        self.vad_thread.start()
        self.stt_supervisor.start()

    def _stt_supervisor_loop(self) -> None:
        # Supervisor: restart the STT thread if it dies. The transcriber's run loop
        # already catches per-segment errors; this is the last-resort safety net for
        # anything that escapes (e.g. backend crash, OOM in a tokenizer).
        backoff = 1.0
        while not self.stt_stopping.is_set():
            t = threading.Thread(target=self.transcriber.run, daemon=True, name="STT")
            t.start()
            t.join()
            if self.stt_stopping.is_set():
                return
            log.error("STT thread exited unexpectedly; restarting in %.1fs", backoff)
            if self.stats is not None:
                self.stats.record_drop("stt_thread_restart")
            self.stt_stopping.wait(timeout=backoff)
            backoff = min(backoff * 2, 30.0)

    def _install_excepthooks(self) -> None:
        def qt_exception_hook(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                log.error("Uncaught exception: %s: %s", exc_type.__name__, exc_value)
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        def unraisable_hook(unraisable):
            log.error(
                "Unraisable exception: %s: %s",
                unraisable.exc_type.__name__,
                unraisable.exc_value,
            )

        sys.excepthook = qt_exception_hook
        sys.unraisablehook = unraisable_hook

    def _build_display(self) -> None:
        self.bus = DisplayBus(self.display_queue)
        self.bus.start()

        # Mode de lancement : CLI overlay-seul vs .app fenêtre principale.
        self.mode = launch_mode()
        self.overlay = SubtitleOverlay(
            self.bus, self.cfg.ui, interactive=(self.mode == "window")
        )
        log.info("Launch mode: %s", self.mode)

    def _open_preferences(self) -> None:
        from benji.ui.preferences_dialog import PreferencesDialog

        PreferencesDialog(
            self.cfg.stt, self.cfg.ui, self.user_settings,
            on_live_change=self.overlay.apply_config,
            llm_config=self.cfg.llm,
            audio_config=self.cfg.audio,
        ).exec()

    @property
    def _consent(self):
        """Portillon de conservation du transcripteur, s'il y en a un.

        Absent en mode remote (pas de transcripteur local) et quand
        `confirm_before_saving` est désactivé — dans ce cas rien n'est à demander.
        """
        consent = getattr(getattr(self, "transcriber", None), "consent", None)
        if consent is None or consent.armed:
            return None
        return consent

    def _mark_moment(self) -> bool:
        """Marque le moment présent. Faux si rien n'a encore été transcrit.

        Passe par la fenêtre quand elle existe (la marque doit *se voir* sur la
        ligne de temps), sinon écrit directement dans le registre — l'overlay
        seul reste un mode d'usage complet.
        """
        if self.main_window is not None:
            return self.main_window.live_tab.mark_moment()
        from benji import meetings

        return meetings.add_mark() is not None

    def _learn_term(self, term: str) -> bool:
        """Apprend un terme du glossaire et l'arme sur le moteur en cours."""
        if self.transcriber is None:
            return False
        return self.transcriber.learn_term(term)

    @property
    def _has_consent_gate(self) -> bool:
        return getattr(getattr(self, "transcriber", None), "consent", None) is not None

    def _is_saving(self) -> bool:
        consent = getattr(getattr(self, "transcriber", None), "consent", None)
        return bool(consent.armed) if consent is not None else True

    def _on_speaker_named_in_history(self, meeting_id: str, label: str, name: str) -> None:
        """Un nom posé dans la fenêtre Réunions vaut pour le direct, si c'est
        la réunion en cours : sinon le Live et les sous-titres continueraient
        d'afficher « A » pour quelqu'un qu'on vient de nommer."""
        from benji import meetings

        if meeting_id != meetings.current_meeting_id():
            return
        if self.main_window is not None:
            self.main_window.live_tab.set_speaker_name(label, name)  # relaie à l'overlay
        elif self.overlay is not None:
            self.overlay.set_speaker_name(label, name)

    def _on_new_meeting(self) -> None:
        """Nouvelle réunion : l'accord est à redemander, le bandeau revient.

        Les noms de locuteurs repartent de zéro : « A » ne désigne plus
        forcément la même personne.
        """
        if self.main_window is not None:
            self.main_window.live_tab.clear_speaker_names()
        if self.overlay is not None:
            self.overlay.clear_speaker_names()
        consent = getattr(getattr(self, "transcriber", None), "consent", None)
        if consent is None:
            return
        consent.reset(armed=not self.cfg.stt.confirm_before_saving)
        if self.main_window is not None:
            self.main_window.live_tab.set_consent_pending(not consent.armed)

    def _save_meeting(self) -> int:
        """Accorde la conservation de la réunion en cours. Retourne le versé.

        Appelable depuis la fenêtre **ou** le tray : c'est ici que l'état est
        unifié, sinon accorder depuis le menu laisserait le bandeau de la
        fenêtre affirmer le contraire.
        """
        consent = self._consent
        versees = consent.arm() if consent is not None else 0
        if self.main_window is not None:
            self.main_window.live_tab.set_consent_pending(False)
        return versees

    def toggle_pause(self) -> bool:
        """Bascule la pause micro. Retourne le nouvel état (True = en pause).

        La pause ferme réellement le stream d'entrée (l'indicateur micro macOS
        s'éteint). Appelé depuis le tray et la fenêtre principale — thread Qt.

        La capture système suit le micro : mettre en pause doit tout arrêter,
        sinon le voyant micro éteint laisserait croire que plus rien n'est
        transcrit alors que la réunion continuerait de l'être.
        """
        if self.capture.is_paused:
            self.capture.resume()
            if self.system_capture is not None:
                self.system_capture.start()
        else:
            self.capture.pause()
            if self.system_capture is not None:
                self.system_capture.stop()
            # L'utterance en cours ne sera jamais terminée : éteindre
            # l'indicateur « en écoute » de l'UI.
            if self.display_queue is not None:
                self.display_queue.put({"type": "vad_status", "speaking": False})
        paused = self.capture.is_paused
        if self.main_window is not None:
            self.main_window.set_paused(paused)
        return paused

    def _build_windows(self) -> None:
        self.history_window = HistoryWindow(session_start=self.session_start, stats=self.stats)
        self.history_window.hide()

        self.live_summary_window = LiveSummaryWindow()
        self.live_summary_window.hide()
        self.history_window.speaker_named.connect(self._on_speaker_named_in_history)

        if self.mode != "window":
            return

        self.summary_worker = SummaryWorker(provider=build_summary_provider(self.cfg.llm))
        self.summary_worker.start()

        self.main_window = MainWindow(
            bus=self.bus,
            history=self.history,
            session_start=self.session_start,
            summary_worker=self.summary_worker,
            on_minimize=lambda: self.controller.show_overlay() if self.controller else None,
            on_open_preferences=self._open_preferences,
            on_toggle_pause=self.toggle_pause,
            on_save_meeting=self._save_meeting if self._consent is not None else None,
            on_learn_term=self._learn_term if self.transcriber is not None else None,
            session=self.session,
            backend_url=self.cfg.llm.backend_url,
        )
        self.controller = WindowController(
            main_window=self.main_window,
            overlay=self.overlay,
            initial_mode="window",
        )
        # Click sur overlay → revient à la fenêtre.
        self.overlay._on_click = lambda: self.controller.show_window()
        # Un locuteur nommé dans le Live l'est aussi dans les sous-titres.
        self.main_window.live_tab.speaker_named.connect(self.overlay.set_speaker_name)

    def _build_tray_and_shortcuts(self) -> None:
        show_main = (lambda: self.controller.show_window()) if self.controller else None
        self.tray = build_tray(
            self.history_window, self.live_summary_window,
            show_main_window=show_main, session=self.session,
            backend_url=self.cfg.llm.backend_url, open_preferences=self._open_preferences,
            toggle_pause=self.toggle_pause,
            is_paused=lambda: self.capture.is_paused,
            stats=self.stats,
            stt_config=self.cfg.stt,
            mark_moment=self._mark_moment,
            save_meeting=self._save_meeting if self._has_consent_gate else None,
            is_saving=self._is_saving,
            on_new_meeting=self._on_new_meeting,
        )

        # Optional: rolling live summary.
        if self.cfg.stt.live_summary_interval_s > 0:
            from benji.llm.live_summary import LiveSummarizer

            self.live_summarizer = LiveSummarizer(
                interval_seconds=self.cfg.stt.live_summary_interval_s,
                session_start=self.session_start,
                on_summary=self.live_summary_window.on_summary,
                on_summary_start=self.live_summary_window.on_summary_start,
                on_summary_chunk=self.live_summary_window.on_summary_chunk,
            )
            self.live_summarizer.start()

        # Titre automatique de la réunion : le modèle local est déjà chargé pour
        # la correction et le résumé, et « Réunion du 21/08 à 14:32 » ne dit rien
        # de ce qu'on trouvera dedans. En mode remote, aucun modèle local n'est
        # chargé — on n'en réveillerait un que pour ça.
        if self.cfg.stt.auto_title and not self.remote_mode:
            from benji.llm.titler import MeetingTitler

            self.meeting_titler = MeetingTitler(
                on_renamed=self.history_window.meeting_renamed.emit
            )
            self.meeting_titler.start()

        self._install_global_hotkeys()

        self._add_shortcut("Ctrl+Shift+H", lambda: self._toggle(self.history_window))
        self._add_shortcut("Ctrl+Shift+S", lambda: self._toggle(self.live_summary_window))
        if IS_MACOS:
            # Diagnostic: dump current macOS window state on demand.
            self._add_shortcut(
                "Ctrl+Shift+D",
                lambda: self.overlay._apply_macos_window_settings(verbose=True),
            )

    def _install_global_hotkeys(self) -> None:
        """Raccourcis actifs même quand Benji n'a pas le focus.

        C'est le seul chemin praticable pour couper le micro en pleine visio :
        les `QShortcut` ci-dessous sont posés sur l'overlay et ne répondent que
        si Benji est au premier plan — jamais le cas quand Teams est en plein
        écran, c'est-à-dire exactement quand on en a besoin.
        """
        if not (self.cfg.ui.global_hotkey_pause or self.cfg.ui.global_hotkey_mark):
            return
        from benji.hotkeys import build_hotkeys

        self.global_hotkeys = build_hotkeys()
        if self.cfg.ui.global_hotkey_pause:
            self.global_hotkeys.register(
                self.cfg.ui.global_hotkey_pause, self._toggle_pause_notified
            )
        if self.cfg.ui.global_hotkey_mark:
            self.global_hotkeys.register(
                self.cfg.ui.global_hotkey_mark, self._mark_moment_notified
            )

    def _mark_moment_notified(self) -> None:
        """Marque un moment et le **dit** — même règle que la pause : un geste
        posé hors de la vue doit se voir confirmé, sinon on ne sait pas s'il a
        porté et on le refait trois fois."""
        marked = self._mark_moment()
        if self.tray is not None:
            from PySide6.QtWidgets import QSystemTrayIcon

            self.tray.showMessage(
                "Benji",
                "Moment marqué" if marked else "Rien à marquer pour l'instant",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

    def _toggle_pause_notified(self) -> None:
        """Bascule la pause et le **dit** — un raccourci global agit hors de la vue.

        Depuis une visio en plein écran, l'utilisateur ne voit ni le tray ni la
        fenêtre : sans notification, il ne saurait pas si sa frappe a porté, et
        croire le micro coupé alors qu'il ne l'est pas est le pire des états.
        """
        paused = self.toggle_pause()
        if self.tray is not None:
            from PySide6.QtWidgets import QSystemTrayIcon

            self.tray.showMessage(
                "Benji",
                "Micro suspendu" if paused else "Micro rétabli",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

    def _add_shortcut(self, keys: str, slot) -> None:
        sc = QShortcut(QKeySequence(keys), self.overlay)
        sc.activated.connect(slot)
        self._shortcuts.append(sc)

    @staticmethod
    def _toggle(window) -> None:
        window.show() if window.isHidden() else window.hide()

    def _install_signal_handlers(self) -> None:
        self.app.aboutToQuit.connect(
            lambda: self.overlay.cleanup() if not self.overlay._shutting_down else None
        )
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame) -> None:
        log.info("Interrupt received, shutting down...")
        self.overlay.cleanup()
        QTimer.singleShot(0, self.app.quit)

    def shutdown(self) -> None:
        log.info("Shutting down...")
        # Horodate la fin de la réunion en cours (si une transcription a eu lieu)
        # avant de couper les threads qui pourraient encore en écrire.
        from benji import meetings

        meetings.end_current_meeting()
        if self.summary_worker is not None:
            self.summary_worker.shutdown()
        if self.bus is not None:
            self.bus.stop()
        if self.live_summarizer:
            self.live_summarizer.stop()
        if self.meeting_titler is not None:
            self.meeting_titler.stop()
        if self.global_hotkeys is not None:
            self.global_hotkeys.unregister_all()
        if self.capture is not None:
            self.capture.stop()
        # Ordre : micro, puis mixeur, puis capture système — on coupe la source
        # avant le consommateur pour éviter que le mixeur tourne à vide.
        if self.mixer is not None:
            self.mixer.stop()
        if self.system_capture is not None:
            self.system_capture.stop()
        if self.audio_queue is not None:
            self.audio_queue.put(None)
        self.stt_stopping.set()
        if self.remote_stt is not None:
            self.remote_stt.stop()
        if not self.remote_mode and self.transcribe_queue is not None:
            self.transcribe_queue.put(None)
        if self.vad_thread is not None:
            self.vad_thread.join(timeout=2)
        if self.stt_supervisor is not None:
            self.stt_supervisor.join(timeout=3)
        if self.remote_thread is not None:
            self.remote_thread.join(timeout=2)
        # Après les joins : les compteurs (segments, drops) sont figés, on
        # persiste l'instantané final plutôt qu'un état encore en mouvement.
        if self.stats is not None:
            self.stats.save()
