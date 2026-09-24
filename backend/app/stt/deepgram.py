"""Session STT Deepgram (streaming temps réel).

Traduit le flux Deepgram vers les events du contrat Benji. Le branchement réel
nécessite DEEPGRAM_API_KEY et une validation en conditions réelles (non couvert
par les tests hermétiques, qui utilisent FakeSTTSession).

Mapping :
  SpeechStarted  → vad_status(speaking=true) + segment_start
  UtteranceEnd   → vad_status(speaking=false)
  Results interim→ `word` (deltas par rapport au partiel précédent)
  Results final  → final_text, un par tour de parole (+ speaker si diarisation)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from urllib.parse import urlencode

from app.stt.base import BaseSTTSession
from app.stt.turns import split_turns

log = logging.getLogger(__name__)

_DG_URL = "wss://api.deepgram.com/v1/listen"
# Deepgram ferme la connexion (NET-0001) après 10 s sans audio ni KeepAlive —
# ce qui arrive dès que l'utilisateur met le micro en pause (le client ferme son
# flux audio). Doc : 3 à 5 s entre deux KeepAlive.
_KEEPALIVE_S = 4.0


class DeepgramSTTSession(BaseSTTSession):
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        language: str | None = "fr",
        diarization: bool = True,
        model: str = "nova-3",
    ):
        super().__init__()
        self._api_key = api_key
        self._params = {
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": "1",
            # None = détection automatique côté client : nova-3 le fait en
            # streaming via "multi" (sans quoi l'URL portait `language=None`).
            "language": language or "multi",
            "model": model,
            "interim_results": "true",
            "punctuate": "true",
            "vad_events": "true",
            # Sans ce paramètre, Deepgram n'envoie jamais `UtteranceEnd` : le
            # vad_status(speaking=false) ne partait pas et l'indicateur de parole
            # restait allumé.
            "utterance_end_ms": "1000",
            "diarize": "true" if diarization else "false",
        }
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._keepalive: asyncio.Task | None = None
        self._last_send = 0.0
        self._partial_words: list[str] = []

    async def open(self) -> None:
        import websockets

        url = f"{_DG_URL}?{urlencode(self._params)}"
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )
        self._reader = asyncio.create_task(self._read_loop())
        self._last_send = time.monotonic()
        self._keepalive = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        """Garde la connexion ouverte pendant une pause du micro."""
        while True:
            await asyncio.sleep(_KEEPALIVE_S)
            if time.monotonic() - self._last_send < _KEEPALIVE_S:
                continue
            try:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                return  # connexion fermée : _read_loop s'en charge
            self._last_send = time.monotonic()

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._translate(msg)
        except Exception as e:  # fermeture / réseau
            log.warning("Deepgram read loop ended: %s", e)
        finally:
            await self._emit_done()

    async def _translate(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "SpeechStarted":
            await self._emit({"type": "vad_status", "speaking": True})
            await self._emit({"type": "segment_start"})
            self._partial_words = []
            return
        if mtype == "UtteranceEnd":
            await self._emit({"type": "vad_status", "speaking": False})
            return
        if mtype != "Results":
            return

        alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
        transcript = (alt.get("transcript") or "").strip()
        if not transcript:
            return

        if not msg.get("is_final"):
            words = transcript.split()
            for w in words[len(self._partial_words):]:
                await self._emit({"type": "word", "text": w})
            self._partial_words = words
            return

        # Segment finalisé : un final_text par tour de parole (cf. turns.py).
        words = alt.get("words") or []
        for text, spk in split_turns(transcript, words, ("punctuated_word", "word")):
            out: dict = {"type": "final_text", "text": text}
            if spk:
                out["speaker"] = spk
            await self._emit(out)
        self._partial_words = []

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is not None:
            await self._ws.send(chunk)
            self._last_send = time.monotonic()

    async def finish(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
        # _read_loop émettra _emit_done à la fermeture du flux Deepgram.

    async def close(self) -> None:
        if self._keepalive is not None:
            self._keepalive.cancel()
        if self._reader is not None:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        await self._emit_done()
