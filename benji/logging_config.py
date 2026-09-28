"""Centralized logging setup.

Each module gets its logger via `logging.getLogger(__name__)`. The console
handler emits a short, colorized-friendly format that mirrors the legacy
`[Tag] message` style by using the logger's *short* name as the tag.

Verbosity is controlled by env var `BENJI_LOG_LEVEL` (default INFO). Set to
DEBUG for verbose troubleshooting, WARNING to silence routine messages.

Deux handlers : stderr (dev) et un fichier tournant dans `log_dir()`. Lancée
depuis le Finder, l'app n'a pas de terminal attaché — stderr part au néant, et
le fichier est alors le *seul* canal de diagnostic (cf. « Révéler les logs »
dans le menu tray). Il porte donc un format plus riche : horodatage + niveau.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_TAG_BY_MODULE = {
    "benji.main": "Benji",
    "benji.audio.capture": "Audio",
    "benji.audio.vad": "VAD",
    "benji.stt.backend": "STT",
    "benji.stt.transcriber": "STT",
    "benji.stt.diarization": "Diarization",
    "benji.llm.summarizer": "Summary",
    "benji.llm.live_summary": "LiveSummary",
    "benji.llm.corrector": "LLM",
    "benji.ui.overlay": "UI",
    "benji.ui.splash": "UI",
}


class _TagFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        tag = _TAG_BY_MODULE.get(record.name, record.name.split(".")[-1])
        record.tag = tag
        return super().format(record)


_configured = False

_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3


def log_dir() -> Path:
    """Dossier des logs : `~/Library/Logs/Benji` (macOS), `%LOCALAPPDATA%` (Windows), sinon XDG."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "Benji"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "Benji" / "Logs"
    base = os.environ.get("XDG_STATE_HOME")
    return (Path(base) if base else Path.home() / ".local" / "state") / "benji"


def log_file_path() -> Path:
    return log_dir() / "benji.log"


def _build_file_handler() -> logging.Handler | None:
    """RotatingFileHandler sur `log_file_path()`, ou None si le disque refuse.

    Un log qu'on n'arrive pas à écrire ne doit jamais empêcher l'app de démarrer.
    """
    try:
        path = log_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setFormatter(
        _TagFormatter("%(asctime)s %(levelname)-7s [%(tag)s] %(message)s")
    )
    return handler


def setup_logging(level: str | None = None) -> None:
    """Configure the root `benji` logger. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    resolved = (level or os.environ.get("BENJI_LOG_LEVEL") or "INFO").upper()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_TagFormatter("[%(tag)s] %(message)s"))

    root = logging.getLogger("benji")
    root.setLevel(resolved)
    root.addHandler(handler)

    file_handler = _build_file_handler()
    if file_handler is not None:
        root.addHandler(file_handler)

    root.propagate = False

    if file_handler is None:
        root.warning("Could not open log file at %s — stderr only", log_file_path())
