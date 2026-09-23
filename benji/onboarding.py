"""Premier lancement : permission micro et téléchargement des modèles.

Le trou le plus large entre le code et un produit installable était là. Un
utilisateur qui ouvrait le DMG tombait sur une fenêtre figée : Benji chargeait
Parakeet — plus d'un gigaoctet à télécharger — sans barre de progression, et
demandait l'accès au micro au moment où le flux s'ouvre, c'est-à-dire sans avoir
rien expliqué. En cas de refus, plus rien ne se passait jamais, sans un mot.

Ce module porte la logique de ce premier lancement, **sans Qt** : l'état de la
permission micro, la présence des poids sur le disque, l'avancement d'un
téléchargement. `benji/ui/onboarding_window.py` ne fait que le mettre en scène.

Le marqueur de fin vit dans les données utilisateur et pas dans QSettings :
supprimer le dossier de données doit remettre Benji dans l'état d'un premier
lancement, ce qui est aussi la façon la plus simple de rejouer l'assistant.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

MARKER_NAME = "onboarded.json"

# Les poids nécessaires à une transcription complète, avec une taille estimée
# pour afficher une progression avant même de savoir ce que le dépôt pèse.
#
# Whisper est là alors que le moteur hybride ne l'appelle que sur les segments
# qui dérivent : justement, ce jour-là on est **en réunion**. Télécharger 1,5 Go
# au milieu d'une phrase serait pire que de le faire maintenant.
#
# Le modèle de langue (résumés, titres de réunion) y est aussi : le titreur
# l'appelle dès les 300 premiers caractères d'une réunion, il était donc
# téléchargé en douce — 800 Mo que personne n'avait acceptés.
REQUIRED_MODELS: tuple[tuple[str, str, int], ...] = (
    ("mlx-community/parakeet-tdt-0.6b-v3", "Moteur de transcription", 2_500_000_000),
    ("mlx-community/whisper-medium-mlx", "Garantie de langue", 1_600_000_000),
    ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "Résumés et titres", 900_000_000),
)


# --- marqueur de premier lancement ------------------------------------------


def marker_path() -> Path:
    from benji.paths import data_dir

    return data_dir() / MARKER_NAME


def needs_onboarding(path: Path | None = None) -> bool:
    return not (path or marker_path()).exists()


# --- accord pour les modèles locaux -------------------------------------------
#
# Règle produit : **aucun poids n'est téléchargé sans l'accord de l'utilisateur**.
# Il peut préférer le cloud Benji (abonnement) et ne jamais vouloir 4 Go sur son
# disque. L'accord se donne dans l'assistant (bouton « Télécharger ») et vit dans
# le marqueur ; chaque chargement de modèle local le vérifie avant de toucher au
# réseau (`ensure_allowed`). Ce qui est déjà sur le disque se charge toujours :
# l'accord porte sur le téléchargement, pas sur l'usage.


class ModelNotAllowed(RuntimeError):
    """Un modèle manque et l'utilisateur n'a pas accepté de le télécharger."""


def local_models_allowed(path: Path | None = None) -> bool:
    """L'utilisateur a-t-il accepté de télécharger les modèles locaux ?

    Marqueur sans la clé (posé par une version antérieure de l'assistant) :
    l'accord est réputé donné si le moteur est déjà là — il n'a pu arriver que
    par le bouton de l'assistant ou un lancement en local délibéré.
    """
    try:
        payload = json.loads((path or marker_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if "local_models" in payload:
        return bool(payload["local_models"])
    return is_downloaded(REQUIRED_MODELS[0][0])


def ensure_allowed(repo_id: str, marker: Path | None = None,
                   cache_root: Path | None = None) -> None:
    """Lève `ModelNotAllowed` si charger `repo_id` déclencherait un
    téléchargement que l'utilisateur n'a pas accepté."""
    if is_downloaded(repo_id, cache_root) or local_models_allowed(marker):
        return
    raise ModelNotAllowed(
        "Ce modèle local n'est pas téléchargé, et vous n'avez pas accepté de le "
        "télécharger. Autorisez les modèles locaux, ou passez par le cloud Benji."
    )


def mark_done(path: Path | None = None, **details) -> None:
    """Pose le marqueur. Toute défaillance est non fatale : au pire, l'assistant
    se represente une fois de trop — bien moins grave que de bloquer l'app."""
    path = path or marker_path()
    payload = {"version": 1, **details}
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as e:
        log.warning("Marqueur de premier lancement non écrit (%s)", e)


# --- permission micro -------------------------------------------------------

GRANTED = "granted"
DENIED = "denied"
UNDETERMINED = "undetermined"
UNKNOWN = "unknown"

# AVAuthorizationStatus (AVFoundation).
_AV_STATUS = {0: UNDETERMINED, 1: DENIED, 2: DENIED, 3: GRANTED}


def microphone_status() -> str:
    """État de l'autorisation micro, sans jamais lever.

    `UNKNOWN` quand AVFoundation n'est pas disponible (pyobjc partiel, machine
    non-macOS) : l'assistant propose alors un test au lieu d'un état — mieux
    vaut ne rien affirmer que d'affirmer faux.
    """
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        raw = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
    except Exception:
        return UNKNOWN
    return _AV_STATUS.get(int(raw), UNKNOWN)


def request_microphone_access(on_result: Callable[[bool], None]) -> bool:
    """Déclenche la demande système. Renvoie False si l'API est indisponible.

    `on_result` est appelé depuis un thread système, pas depuis Qt : à
    l'appelant de marshaller vers l'interface.
    """
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, lambda granted: on_result(bool(granted))
        )
        return True
    except Exception as e:
        log.warning("Demande d'accès micro impossible (%s)", e)
        return False


def open_privacy_settings() -> str:
    """URL du volet Confidentialité › Microphone des Réglages Système.

    Une fois l'accès refusé, macOS ne redemande **plus jamais** : le seul chemin
    de réparation passe par les Réglages, et l'utilisateur ne le trouve pas
    seul.
    """
    return "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"


# --- présence et téléchargement des modèles ---------------------------------


def repo_cache_name(repo_id: str) -> str:
    """Nom du dossier d'un dépôt dans le cache Hugging Face."""
    return "models--" + repo_id.replace("/", "--")


def hf_cache_root() -> Path:
    """Racine du cache HF, en respectant les variables d'environnement.

    Résolue **à l'appel** : figée à l'import, elle capturerait le HOME du moment
    (cf. la même règle dans `benji/paths.py`).
    """
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def downloaded_bytes(repo_id: str, cache_root: Path | None = None) -> int:
    """Octets déjà sur le disque pour ce dépôt, fichiers partiels compris.

    Les `.incomplete` comptent : ce sont précisément les octets en train
    d'arriver, et c'est ce qui fait avancer la barre pendant un téléchargement.
    """
    root = (cache_root or hf_cache_root()) / repo_cache_name(repo_id)
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def is_downloaded(repo_id: str, cache_root: Path | None = None) -> bool:
    """Vrai si le dépôt a un instantané complet (aucun fichier partiel)."""
    root = (cache_root or hf_cache_root()) / repo_cache_name(repo_id)
    snapshots = root / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return False
    return not any(root.rglob("*.incomplete"))


def missing_models(cache_root: Path | None = None) -> list[tuple[str, str, int]]:
    return [m for m in REQUIRED_MODELS if not is_downloaded(m[0], cache_root)]


def format_size(num_bytes: float) -> str:
    """« 1,2 Go » — l'utilisateur veut un ordre de grandeur, pas des octets."""
    if num_bytes < 1_000_000:
        return f"{num_bytes / 1000:.0f} ko"
    if num_bytes < 1_000_000_000:
        return f"{num_bytes / 1_000_000:.0f} Mo"
    return f"{num_bytes / 1_000_000_000:.1f} Go".replace(".", ",")


def progress_fraction(done: int, total: int) -> float:
    """Avancement borné à [0, 1] — les tailles annoncées sont des estimations,
    et une barre qui dépasse 100 % est plus inquiétante qu'informative."""
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, done / total))


class ModelDownloader:
    """Télécharge les poids manquants sur un fil de fond, avec progression.

    L'avancement est **observé sur le disque** plutôt que demandé à
    `huggingface_hub` : sa barre de progression passe par tqdm, dont l'API de
    substitution n'est pas stable d'une version à l'autre. Compter les octets du
    dossier de cache marche quelle que soit la version, et compte aussi ce qui
    était déjà là.
    """

    POLL_SECONDS = 0.5

    def __init__(
        self,
        models=REQUIRED_MODELS,
        on_progress: Callable[[float, str], None] | None = None,
        on_done: Callable[[str | None], None] | None = None,
        cache_root: Path | None = None,
    ):
        self.models = list(models)
        self.on_progress = on_progress
        self.on_done = on_done
        self.cache_root = cache_root
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ModelDownload"
        )
        self._thread.start()

    def cancel(self) -> None:
        """Demande l'arrêt entre deux modèles.

        Un téléchargement en cours n'est pas interrompu au milieu : les octets
        déjà obtenus restent en cache et la reprise repart de là.
        """
        self._stop.set()

    def _run(self) -> None:
        error: str | None = None
        try:
            for repo_id, label, expected in self.models:
                if self._stop.is_set():
                    break
                if is_downloaded(repo_id, self.cache_root):
                    continue
                self._download_one(repo_id, label, expected)
        except Exception as e:
            log.warning("Téléchargement des modèles interrompu (%s)", e)
            error = str(e)
        if self.on_done is not None:
            self.on_done(error)

    def _download_one(self, repo_id: str, label: str, expected: int) -> None:
        from huggingface_hub import snapshot_download

        done = threading.Event()
        watcher = threading.Thread(
            target=self._watch, args=(repo_id, label, expected, done), daemon=True
        )
        watcher.start()
        try:
            snapshot_download(
                repo_id,
                cache_dir=str(self.cache_root) if self.cache_root else None,
            )
        finally:
            done.set()
            watcher.join(timeout=1)

    def _watch(self, repo_id: str, label: str, expected: int, done: threading.Event) -> None:
        while not done.wait(self.POLL_SECONDS):
            if self.on_progress is None:
                continue
            got = downloaded_bytes(repo_id, self.cache_root)
            self.on_progress(
                progress_fraction(got, expected),
                f"{label} — {format_size(got)} sur ~{format_size(expected)}",
            )
