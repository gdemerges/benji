import platform
from dataclasses import dataclass, field

IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


def _default_font() -> str:
    return ".AppleSystemUIFont" if IS_MACOS else "Segoe UI"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512  # Silero VAD ONNX requires 512 samples (32ms @ 16kHz)
    dtype: str = "float32"
    # Capture de l'audio système (son des autres participants en visio), mixé
    # avec le micro avant le VAD. Nécessite un pilote de boucle installé par
    # l'utilisateur — cf. benji/audio/loopback.py. Désactivé => chemin micro
    # seul strictement inchangé.
    system_audio: bool = False
    # Sous-chaîne du nom du périphérique de boucle. None = auto-détection.
    system_audio_device: str | None = None
    # Gain appliqué au flux système avant sommation. 1.0 convient quand la
    # sortie est à un niveau normal ; baisser si la visio sature le mixage.
    system_audio_gain: float = 1.0


@dataclass
class VADConfig:
    speech_threshold: float = 0.5
    # Hystérésis : la parole *commence* au-dessus de `speech_threshold`, mais ne
    # *s'arrête* que sous `speech_threshold - speech_end_hysteresis` (la valeur
    # recommandée par Silero). Sans elle, les syllabes faibles d'une fin de
    # phrase comptaient comme du silence — surtout quand le seuil adaptatif
    # monte dans une pièce bruyante — et l'énoncé était coupé trop tôt.
    speech_end_hysteresis: float = 0.15
    silence_duration_ms: int = 600  # Wait longer before cutting, reduces fragmentation
    min_speech_duration_ms: int = 300  # Keep short interjections ("oui", "ok", "non")
    max_speech_duration_s: float = 8.0  # Force flush sooner for long utterances
    # Une coupure forcée (énoncé plus long que `max_speech_duration_s`) tombait
    # net, au milieu d'un mot, que chaque moitié décodait ensuite de travers. On
    # coupe désormais au point le plus calme (confiance VAD + énergie, lissées)
    # parmi les `cut_search_ms` dernières millisecondes, et la suite repart dans
    # le tampon suivant au lieu d'être tranchée (cf. audio/vad.py).
    cut_search_ms: int = 1500
    # Après une coupure forcée, la parole continue : le segment suivant est
    # décodé avec cette queue du précédent devant lui, pour que son premier mot
    # ait du contexte. Les mots qui tombent dans ce contexte sont retirés du
    # texte (cf. stt/transcriber.py) — ils appartiennent déjà au segment d'avant.
    # 0 = pas de contexte. Sans objet après une vraie pause, où le contexte ne
    # serait que du silence.
    cut_context_ms: int = 1000
    pre_speech_pad_ms: int = 200  # Less pre-context = smaller audio buffer = faster inference
    partial_interval_ms: int = 400  # Re-transcribe partial audio every N ms (0 = disabled)
    # Espacement progressif des passes partielles à mesure que le tampon grandit :
    # intervalle effectif = partial_interval_ms + growth_factor * durée_tampon_ms.
    # Hérité de Whisper, dont une passe sur 8 s coûtait ~800 ms : sans ce frein, le
    # coût total devenait intenable. Parakeet décode le même tampon en ~150 ms, soit
    # ~38 % d'occupation à cadence fixe — le frein ne protège plus rien et rendait le
    # direct poussif (jusqu'à 4,4 s entre deux rafraîchissements en fin d'énoncé).
    # Remis à 0 = cadence fixe. Le mécanisme reste là si un moteur plus lourd revient.
    partial_growth_factor: float = 0.0
    # Adaptive threshold: lifts speech_threshold above the noise floor in noisy rooms.
    # Effective threshold = max(speech_threshold, p95(non_speech_conf) + adaptive_margin).
    adaptive_threshold: bool = True
    adaptive_margin: float = 0.10
    adaptive_window_seconds: float = 5.0  # Rolling window for noise-floor estimation


@dataclass
class STTConfig:
    # "parakeet" : Parakeet TDT sur le Mac (défaut). "remote" : transcription via
    # le backend Benji (cf. docs/api-contract.md ; coordonnées dans LLMConfig).
    stt_provider: str = "parakeet"
    # Poids du moteur des **passes partielles** (le texte vivant, éphémère).
    model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    # Moteur de la **passe finale** — celle dont le texte part dans l'historique,
    # les exports et les résumés.
    #   "hybrid"   — Parakeet, rattrapé par Whisper (défaut). Le segment est
    #                décodé par Parakeet puis *relu* : on ne relance Whisper que
    #                si le texte n'est visiblement pas dans `language` (cf.
    #                stt/language.py). Garantie conservée là où elle se joue,
    #                coût payé seulement là où il sert — et les poids de Whisper
    #                ne sont chargés que si un segment en a besoin.
    #   "whisper"  — Whisper sur *tous* les finals : ~800 ms par segment, la
    #                garantie maximale. Le repli si l'hybride déçoit en réunion.
    #   "parakeet" — réutilise le moteur des partielles : ~5× plus rapide sur le
    #                final, mais la langue n'est plus garantie du tout.
    # Sans Parakeet/MLX (Windows/Linux), ce champ dimensionne directement
    # FasterWhisperBackend et `final_engine` est ignoré (cf. stt/backend.py).
    final_engine: str = "hybrid"
    # "medium" coûte ~150 ms sur MLX (rattrapage rare, hybride) mais 2,5-3,8 s
    # sur CPU faster-whisper où c'est le seul moteur de *chaque* final (mesuré,
    # cf. benji/stt/CLAUDE.md) — "small" (~1 s) reste utilisable en réunion.
    final_model_size: str = "medium" if IS_MACOS else "small"
    # Glossaire utilisateur (noms propres, jargon maison) appliqué au texte
    # final, cf. stt/lexicon.py. Le fichier vit dans les données utilisateur ;
    # ce drapeau ne fait qu'activer sa lecture.
    glossary: bool = True
    # Langue imposée à la passe finale, et langue du post-traitement (nombres,
    # interjections) et de la correction LLM. None = détection automatique.
    language: str | None = "fr"
    diarization: bool = True  # Enable speaker labeling
    # "pitch" (built-in F0 clustering, no extra deps) or "pyannote" (real embeddings,
    # requires `uv sync --extra diarization` and HF token via env HF_TOKEN).
    diarization_backend: str = "pyannote"
    diarization_max_speakers: int = 4  # Cap for pyannote clustering (pitch is hard-capped at 2)
    # Découpe d'un segment en tours de parole (cf. stt/diarization.py). Un segment
    # VAD tient souvent deux locuteurs qui s'enchaînent sans pause franche : on
    # étiquette par fenêtres glissantes, puis on recoupe les mots (horodatés) aux
    # frontières. Fenêtre : assez longue pour un embedding stable (pyannote rend
    # None sous 500 ms), assez courte pour serrer une frontière.
    diarization_window_s: float = 1.5
    diarization_hop_s: float = 0.75  # 0 = pas de découpe, une étiquette par segment
    # Un tour plus court que ça est refondu dans son voisin : une fenêtre isolée
    # qui change d'avis au milieu d'une phrase ferait du confetti à l'écran.
    diarization_min_turn_words: int = 2
    llm_correction: bool = False  # Post-hoc grammar/punctuation fix via MLX-LM
    live_summary_interval_s: int = 0  # 0 = disabled; e.g. 300 = every 5 min
    # Nomme la réunion en cours à partir de ses premières phrases, via le modèle
    # local déjà chargé (cf. benji/llm/titler.py). Un titre choisi à la main
    # n'est jamais écrasé. False = les réunions gardent leur horodatage.
    auto_title: bool = True
    # Écouter et garder ne sont pas le même geste. Benji transcrit dès le
    # lancement (on n'a jamais « oublié de lancer l'enregistrement »), mais rien
    # n'est écrit sur disque tant que l'utilisateur ne l'a pas accordé — ce qui a
    # déjà été dit est versé à ce moment-là, pas perdu (cf. benji/recording.py).
    # False = tout est conservé d'office, le comportement d'avant.
    confirm_before_saving: bool = True
    # Audio gain control before STT: peak-normalize quiet segments to this target.
    # 0.0 disables. Useful for low-gain microphones.
    agc_target_peak: float = 0.7
    agc_min_peak: float = 0.3  # Only boost when current peak is below this


@dataclass
class LLMConfig:
    # Choix du moteur de résumé :
    #   "local"  — mlx-lm, 100 % sur le Mac (défaut)
    #   "cloud"  — API Claude en direct (clé sur le poste ; pour dev/test)
    #   "remote" — via le backend Benji (clé côté serveur ; chemin production)
    summary_provider: str = "local"
    # --- mode "cloud" (Claude direct) ---
    # Modèle Claude. Haiku 4.5 : rapide et peu coûteux, suffisant pour du résumé
    # (cf. docs/cloud-architecture.md). Sonnet/Opus pour plus de qualité.
    cloud_model: str = "claude-haiku-4-5"
    # None → l'SDK anthropic lit la clé depuis l'environnement (ANTHROPIC_API_KEY).
    # Ne jamais committer une clé en clair ici.
    anthropic_api_key: str | None = None
    cloud_max_tokens: int = 2048
    # --- mode "remote" (via backend) ---
    backend_url: str = "http://127.0.0.1:8000"
    backend_token: str | None = None          # jeton Bearer du backend
    summary_model_alias: str = "haiku"        # alias logique envoyé au backend


_LOCAL_HOSTS = {"localhost", "::1"}


def ensure_secure_backend_url(url: str) -> str:
    """Valide que l'URL backend est en HTTPS dès qu'elle sort du poste.

    Par ce canal transitent identifiants, jetons Bearer et transcriptions :
    en clair (http/ws), une URL de prod mal saisie exposerait tout. Seul le
    loopback (dev local) est exempté. Lève ValueError sinon — on échoue au
    démarrage plutôt que de fuiter silencieusement.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL backend invalide (schéma {parsed.scheme!r}) : {url}")
    host = parsed.hostname or ""
    is_local = host in _LOCAL_HOSTS or host.startswith("127.")
    if parsed.scheme == "http" and not is_local:
        raise ValueError(
            f"URL backend non locale en HTTP refusée (jetons et transcriptions "
            f"transiteraient en clair) : {url} — utilise https://"
        )
    return url


@dataclass
class UIConfig:
    font_family: str = field(default_factory=_default_font)
    font_size: int = 28
    bg_opacity: int = 160
    display_duration_ms: int = 8000
    fade_duration_ms: int = 1000
    window_width_ratio: float = 0.6
    bottom_margin: int = 80
    streaming_display: bool = True  # Display words progressively
    # Multi-monitor: anchor the overlay on the screen under the cursor (the
    # user's active display), re-evaluated between utterances. False = primary.
    follow_active_screen: bool = True
    # Raccourci **global** (actif même quand Benji n'a pas le focus) pour couper
    # et rendre le micro : en réunion, le focus est sur Teams ou Zoom, donc les
    # QShortcut posés sur l'overlay ne répondent pas. Combinaison à quatre
    # modificateurs pour ne rien prendre à personne. "" = désactivé.
    # cf. benji/hotkeys.py.
    global_hotkey_pause: str = "Ctrl+Alt+Cmd+B"
    # Marquer un moment (« là, c'est important ») sans quitter la visio : c'est
    # le geste qu'on fait vraiment en réunion, et il n'a de valeur que s'il est
    # à portée pendant qu'elle a lieu. "" = désactivé.
    global_hotkey_mark: str = "Ctrl+Alt+Cmd+M"
    # Diagnostic only: verbose macOS window-state dump every 5s (off in prod).
    # Same info is available on demand via Ctrl+Shift+D.
    debug_macos_window: bool = False
