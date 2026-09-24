"""Provider STT temps réel : fabrique de session selon la config serveur.

`STT_BACKEND` (env) : "deepgram" (défaut), "grok" ou "fake" (tests/dev hors-ligne).
Deepgram exige `DEEPGRAM_API_KEY` ; Grok exige `XAI_API_KEY`.
"""

from __future__ import annotations

import os

from app.stt.base import BaseSTTSession, STTSession
from app.stt.deepgram import DeepgramSTTSession
from app.stt.fake import FakeSTTSession
from app.stt.grok import GrokSTTSession

__all__ = [
    "STTSession",
    "BaseSTTSession",
    "DeepgramSTTSession",
    "GrokSTTSession",
    "FakeSTTSession",
    "make_session",
]


def make_session(start: dict) -> STTSession:
    """Construit une session STT depuis le message `start` du client."""
    audio = start.get("audio") or {}
    sample_rate = int(audio.get("sample_rate", 16000)) or 16000
    # Absent → "fr" (défaut du contrat) ; `null` explicite → détection automatique.
    language = start.get("language", "fr") or None
    diarization = bool(start.get("diarization", True))

    backend = os.environ.get("STT_BACKEND", "deepgram").lower()
    if backend == "fake":
        return FakeSTTSession()
    if backend == "grok":
        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY non configurée côté backend.")
        return GrokSTTSession(
            key, sample_rate=sample_rate, language=language, diarization=diarization
        )

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY non configurée côté backend.")
    return DeepgramSTTSession(
        key, sample_rate=sample_rate, language=language, diarization=diarization
    )
