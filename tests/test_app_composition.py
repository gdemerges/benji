"""Composition root : phases non-Qt pilotables + teardown robuste.

On ne lance pas la boucle Qt ni l'audio ici ; on vérifie que le câblage
(configs injectables, injection du token de compte, arrêt idempotent) est
testable en isolation — c'est justement ce que la god-function `main()`
empêchait.
"""

from __future__ import annotations

from benji.app import AppConfigs, BenjiApplication
from benji.config import LLMConfig, STTConfig


def test_configs_are_injectable():
    cfg = AppConfigs(stt=STTConfig(model="tiny"), llm=LLMConfig(backend_url="http://x"))
    app = BenjiApplication(cfg)
    assert app.cfg.stt.model == "tiny"
    assert app.cfg.llm.backend_url == "http://x"


def test_account_token_injected_into_llm_config(monkeypatch):
    class _FakeSession:
        def access_token(self):
            return "acc_123"

    # build_session est importé dans la méthode ; on patche à la source.
    monkeypatch.setattr("benji.account.build_session", lambda url: _FakeSession())

    app = BenjiApplication()
    app._build_account()
    assert app.session is not None
    assert app.cfg.llm.backend_token == "acc_123"


def test_no_token_leaves_backend_token_untouched(monkeypatch):
    class _AnonSession:
        def access_token(self):
            return None

    monkeypatch.setattr("benji.account.build_session", lambda url: _AnonSession())

    app = BenjiApplication()
    assert app.cfg.llm.backend_token is None
    app._build_account()
    assert app.cfg.llm.backend_token is None


def test_build_pipeline_remote_skips_vad():
    # En mode remote, le VAD n'est jamais démarré : on ne doit pas charger
    # le modèle Silero (VADProcessor.__init__).
    app = BenjiApplication(AppConfigs(stt=STTConfig(stt_provider="remote")))
    app._build_pipeline()
    assert app.remote_mode is True
    assert app.vad is None
    # Les drops du callback audio sont comptés dans les stats de session.
    assert app.capture.stats is app.stats


def test_shutdown_is_safe_before_run():
    # Tous les composants sont None sur une instance non démarrée : shutdown()
    # ne doit rien tenter d'invalide (garantit un arrêt propre même si le
    # démarrage a échoué à mi-chemin).
    app = BenjiApplication()
    app.shutdown()  # ne doit pas lever
    assert app.stt_stopping.is_set()


# --- audio système : câblage du mixeur dans le pipeline ---------------------


def _fake_devices(*names):
    return [{"name": n, "max_input_channels": 2, "default_samplerate": 48000} for n in names]


def _stub_qt_free_pipeline(monkeypatch):
    """Neutralise ce qui touche au matériel/ONNX dans _build_pipeline."""
    monkeypatch.setattr("benji.app.AudioCapture", lambda queue, cfg, stats=None: ("capture", queue))
    monkeypatch.setattr(
        "benji.app.VADProcessor",
        lambda *a, **kw: "vad",
    )


def test_system_audio_off_keeps_the_direct_mic_path(monkeypatch):
    """Défaut : aucun mixeur, le micro écrit droit dans audio_queue."""
    from benji.config import AudioConfig

    _stub_qt_free_pipeline(monkeypatch)
    app = BenjiApplication(AppConfigs(audio=AudioConfig(system_audio=False)))
    app._build_pipeline()

    assert app.mixer is None
    assert app.system_capture is None
    assert app.capture[1] is app.audio_queue


def test_system_audio_on_inserts_the_mixer(monkeypatch):
    from benji.config import AudioConfig

    _stub_qt_free_pipeline(monkeypatch)
    monkeypatch.setattr(
        "benji.audio.loopback.select_loopback",
        lambda devices, preferred=None: type(
            "D", (), {"name": "BlackHole 2ch", "is_reliable": True}
        )(),
    )

    started = []

    class FakeSystemCapture:
        def __init__(self, name, sample_rate=16000):
            self.name = name
            self.ring = None

        def start(self):
            started.append(self.name)
            return True

    monkeypatch.setattr("benji.audio.system_capture.SystemAudioCapture", FakeSystemCapture)
    monkeypatch.setattr("sounddevice.query_devices", lambda: _fake_devices("BlackHole 2ch"))

    app = BenjiApplication(AppConfigs(audio=AudioConfig(system_audio=True)))
    app._build_pipeline()

    assert started == ["BlackHole 2ch"]
    assert app.mixer is not None
    # Le micro alimente le mixeur, pas audio_queue directement.
    assert app.capture[1] is app.mixer.mic_queue
    assert app.mixer.audio_queue is app.audio_queue


def test_missing_loopback_falls_back_to_mic_only(monkeypatch):
    """Le pilote de boucle est installé par l'utilisateur : son absence ne doit
    jamais empêcher Benji de démarrer."""
    from benji.config import AudioConfig

    _stub_qt_free_pipeline(monkeypatch)
    monkeypatch.setattr("sounddevice.query_devices", lambda: _fake_devices("MacBook Pro Microphone"))

    app = BenjiApplication(AppConfigs(audio=AudioConfig(system_audio=True)))
    app._build_pipeline()

    assert app.mixer is None
    assert app.capture[1] is app.audio_queue


def test_device_enumeration_failure_falls_back_to_mic_only(monkeypatch):
    from benji.config import AudioConfig

    _stub_qt_free_pipeline(monkeypatch)

    def boom():
        raise OSError("PortAudio indisponible")

    monkeypatch.setattr("sounddevice.query_devices", boom)

    app = BenjiApplication(AppConfigs(audio=AudioConfig(system_audio=True)))
    app._build_pipeline()

    assert app.mixer is None
    assert app.capture[1] is app.audio_queue


def test_system_capture_open_failure_falls_back_to_mic_only(monkeypatch):
    """Le périphérique existe mais refuse de s'ouvrir (déjà pris, reconfiguré)."""
    from benji.config import AudioConfig

    _stub_qt_free_pipeline(monkeypatch)

    class FailingCapture:
        def __init__(self, name, sample_rate=16000):
            pass

        def start(self):
            return False

    monkeypatch.setattr("benji.audio.system_capture.SystemAudioCapture", FailingCapture)
    monkeypatch.setattr("sounddevice.query_devices", lambda: _fake_devices("BlackHole 2ch"))

    app = BenjiApplication(AppConfigs(audio=AudioConfig(system_audio=True)))
    app._build_pipeline()

    assert app.mixer is None
    assert app.system_capture is None
    assert app.capture[1] is app.audio_queue


def test_shutdown_is_idempotent_with_system_audio(monkeypatch):
    """shutdown() est appelé sur un chemin d'erreur comme en sortie normale."""
    from benji.config import AudioConfig

    stopped = []

    class FakeSystemCapture:
        def __init__(self, name, sample_rate=16000):
            pass

        def start(self):
            return True

        def stop(self):
            stopped.append("system")

    _stub_qt_free_pipeline(monkeypatch)
    monkeypatch.setattr("benji.audio.system_capture.SystemAudioCapture", FakeSystemCapture)
    monkeypatch.setattr("sounddevice.query_devices", lambda: _fake_devices("BlackHole 2ch"))

    app = BenjiApplication(AppConfigs(audio=AudioConfig(system_audio=True)))
    app._build_pipeline()
    app.capture = None  # le stub AudioCapture n'a pas de .stop()

    app.shutdown()
    app.shutdown()
    assert stopped == ["system", "system"]


def test_parakeet_est_charge_et_prechauffe_sur_le_thread_principal(monkeypatch):
    """MLX lie le modèle au thread qui le charge **et** le préchauffe.

    Chargé depuis un thread éphémère, Parakeet devient définitivement
    inutilisable dès que ce thread meurt — chaque inférence lève « There is no
    Stream(gpu, N) in current thread », depuis n'importe quel thread, et aucun
    `new_stream` ne le répare. Ce test échoue si le chargement repart en fond.
    """
    import threading

    import benji.app as app_mod

    seen = {}

    class _FakeTranscriber:
        def __init__(self, *args, **kwargs):
            seen["construit"] = threading.current_thread()
            self.history = object()

        def warmup(self):
            seen["prechauffe"] = threading.current_thread()

    class _FakeSplash:
        def set_status(self, _text):
            pass

    monkeypatch.setattr(app_mod, "Transcriber", _FakeTranscriber)

    app = BenjiApplication(AppConfigs(stt=STTConfig(stt_provider="parakeet")))
    app.app = type("QApp", (), {"processEvents": lambda self: None})()
    app.transcribe_queue = app.display_queue = None

    app._load_transcriber(_FakeSplash())

    main = threading.main_thread()
    assert seen["construit"] is main, "Parakeet chargé hors du thread principal"
    assert seen["prechauffe"] is main, "Parakeet préchauffé hors du thread principal"
    assert app.transcriber is not None
    assert app.history is app.transcriber.history


# --- premier lancement ---


def test_l_assistant_est_saute_quand_le_marqueur_existe(monkeypatch):
    from benji import onboarding

    monkeypatch.setattr(onboarding, "needs_onboarding", lambda *a: False)
    monkeypatch.setattr(onboarding, "missing_models", lambda *a: [])

    app = BenjiApplication()
    # Rien à instancier : la fenêtre n'est importée que si l'assistant tourne.
    assert app._run_onboarding() is True


def test_moteur_local_absent_et_jamais_accepte_repose_la_question(monkeypatch):
    """Assistant terminé sans télécharger : on ne va pas chercher 2 Go en
    douce au démarrage, on redemande."""
    from benji import onboarding

    monkeypatch.setattr(onboarding, "needs_onboarding", lambda *a: False)
    monkeypatch.setattr(onboarding, "missing_models", lambda *a: [("r", "l", 1)])
    monkeypatch.setattr(onboarding, "local_models_allowed", lambda *a: False)
    assert BenjiApplication()._onboarding_required() is True

    # Accord donné (un téléchargement a été lancé puis interrompu) : Benji reprend.
    monkeypatch.setattr(onboarding, "local_models_allowed", lambda *a: True)
    assert BenjiApplication()._onboarding_required() is False


def test_le_mode_remote_na_pas_d_assistant(monkeypatch):
    """Ni modèle local à télécharger, ni écran à montrer : la transcription se
    fait côté backend."""
    from benji import onboarding

    monkeypatch.setattr(onboarding, "needs_onboarding", lambda *a: True)

    app = BenjiApplication()
    app.remote_mode = True

    assert app._run_onboarding() is True


def test_le_raccourci_global_est_optionnel():
    """Un raccourci vide ne doit pas armer Carbon pour rien."""
    from benji.config import UIConfig

    app = BenjiApplication(AppConfigs(
        ui=UIConfig(global_hotkey_pause="", global_hotkey_mark="")
    ))
    app._install_global_hotkeys()

    assert app.global_hotkeys is None


def test_l_arret_est_idempotent_sans_raccourci_ni_titreur():
    app = BenjiApplication()
    app.shutdown()
    app.shutdown()


class _Recorder:
    def __init__(self):
        self.calls = []

    def set_speaker_name(self, label, name):
        self.calls.append(("name", label, name))

    def clear_speaker_names(self):
        self.calls.append(("clear",))


def test_un_nom_pose_dans_les_reunions_atteint_le_direct(monkeypatch):
    from types import SimpleNamespace

    from benji import meetings

    monkeypatch.setattr(meetings, "current_meeting_id", lambda: "m-en-cours")
    app = BenjiApplication()
    live, overlay = _Recorder(), _Recorder()
    app.main_window = SimpleNamespace(live_tab=live)
    app.overlay = overlay

    app._on_speaker_named_in_history("m-ancienne", "A", "Paul")
    assert live.calls == []  # une autre réunion : le direct n'est pas concerné

    app._on_speaker_named_in_history("m-en-cours", "A", "Alice")
    # Le Live relaie lui-même à l'overlay (signal) : pas de double application.
    assert live.calls == [("name", "A", "Alice")]
    assert overlay.calls == []


def test_sans_fenetre_le_nom_va_droit_aux_sous_titres(monkeypatch):
    from benji import meetings

    monkeypatch.setattr(meetings, "current_meeting_id", lambda: "m1")
    app = BenjiApplication()
    overlay = _Recorder()
    app.main_window, app.overlay = None, overlay

    app._on_speaker_named_in_history("m1", "A", "Alice")
    assert overlay.calls == [("name", "A", "Alice")]


def test_une_nouvelle_reunion_oublie_les_noms_partout():
    from types import SimpleNamespace

    app = BenjiApplication()
    live, overlay = _Recorder(), _Recorder()
    app.main_window = SimpleNamespace(live_tab=live)
    app.overlay = overlay

    app._on_new_meeting()

    assert live.calls == [("clear",)]
    assert overlay.calls == [("clear",)]


def test_en_mode_remote_rien_n_est_conserve_sans_accord(monkeypatch):
    """Le client distant écrivait droit dans l'historique : en mode cloud, tout
    partait sur disque sans l'accord que le mode local exige."""
    import benji.stt.remote as remote_mod
    from benji.recording import RecordingConsent

    seen = {}

    def fake_build(audio_q, display_q, history, *a, **kw):
        seen["history"] = history
        return object()

    monkeypatch.setattr(remote_mod, "build_remote_stt_client", fake_build)

    class _FakeSplash:
        def set_status(self, _text):
            pass

    app = BenjiApplication(AppConfigs(stt=STTConfig(stt_provider="remote")))
    app.remote_mode = True
    app.app = type("QApp", (), {"processEvents": lambda self: None})()
    app._load_transcriber(_FakeSplash())

    assert isinstance(seen["history"], RecordingConsent)
    assert seen["history"] is app.consent
    assert not app.consent.armed
    assert app._has_consent_gate and not app._is_saving()

    # Nouvelle réunion : l'accord est redemandé, en remote aussi.
    app.consent.arm()
    app._on_new_meeting()
    assert not app.consent.armed


def test_la_pause_ne_bloque_jamais_le_thread_qt():
    """Le thread Qt vide lui-même display_queue : un `put` bloquant sur une
    file pleine le figeait pour de bon."""
    from queue import Queue

    app = BenjiApplication()
    app.display_queue = Queue(maxsize=1)
    app.display_queue.put("plein")
    flushed = []
    app.vad = type("V", (), {"request_flush": lambda self: flushed.append(True)})()
    app.capture = type("C", (), {
        "is_paused": False,
        "pause": lambda self: setattr(self, "is_paused", True),
    })()

    assert app.toggle_pause() is True  # rend la main malgré la file pleine
    assert flushed == [True]


def test_a_l_arret_display_queue_ne_bloque_plus_les_producteurs():
    from benji.queues import NotifyingQueue

    app = BenjiApplication()
    app.display_queue = NotifyingQueue(maxsize=1)
    app.display_queue.put("plein")

    app._discard_display()

    app.display_queue.put("a", timeout=0.5)  # ne lève pas Full
    app.display_queue.put("b", timeout=0.5)
