"""Session STT xAI Grok (streaming temps réel).

Doc : https://docs.x.ai/developers/model-capabilities/audio/speech-to-text
Endpoint WebSocket : wss://api.x.ai/v1/stt (config par query params, auth Bearer,
frames audio binaires, fin via {"type":"audio.done"}).

Traduit le flux Grok vers les events du contrat Benji (cf. api-contract §3).
Comme pour Deepgram, le branchement réel demande XAI_API_KEY + une validation en
conditions réelles ; les tests couvrent la traduction des messages.

Mapping :
  transcript.partial (interim) → segment_start (1re fois) + `word` (deltas)
  transcript.done   (final)    → final_text par tour (+ speaker si diarisation) + vad off
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlencode

from app.stt.base import BaseSTTSession
from app.stt.turns import split_turns

log = logging.getLogger(__name__)

_XAI_URL = "wss://api.x.ai/v1/stt"


class GrokSTTSession(BaseSTTSession):
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        language: str | None = "fr",
        diarization: bool = True,
    ):
        super().__init__()
        self._api_key = api_key
        self._params = {
            "encoding": "pcm",
            "sample_rate": str(sample_rate),
            "diarize": "true" if diarization else "false",
            "interim_results": "true",
        }
        # None = détection automatique : on laisse Grok décider plutôt que
        # d'envoyer `language=None` dans l'URL.
        if language:
            self._params["language"] = language
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._partial_words: list[str] = []
        self._in_segment = False

    async def open(self) -> None:
        import websockets

        url = f"{_XAI_URL}?{urlencode(self._params)}"
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                await self._translate(msg)
        except Exception as e:
            log.warning("Grok read loop ended: %s", e)
        finally:
            await self._emit_done()

    async def _translate(self, msg: dict) -> None:
        mtype = msg.get("type")

        # Interim (xAI a expédié `transcript.partial` ; on tolère `transcript.part`).
        if mtype in ("transcript.partial", "transcript.part"):
            transcript = (msg.get("text") or "").strip()
            if not transcript:
                return
            if not self._in_segment:
                self._in_segment = True
                await self._emit({"type": "vad_status", "speaking": True})
                await self._emit({"type": "segment_start"})
            words = transcript.split()
            for w in words[len(self._partial_words):]:
                await self._emit({"type": "word", "text": w})
            self._partial_words = words
            return

        if mtype == "transcript.done":
            transcript = (msg.get("text") or "").strip()
            if transcript:
                # Un final_text par tour de parole (cf. turns.py).
                words = msg.get("words") or []
                for text, spk in split_turns(transcript, words, ("text",)):
                    out: dict = {"type": "final_text", "text": text}
                    if spk:
                        out["speaker"] = spk
                    await self._emit(out)
            await self._emit({"type": "vad_status", "speaking": False})
            self._partial_words = []
            self._in_segment = False
            return

        # transcript.created et autres : rien à relayer.

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is not None:
            await self._ws.send(chunk)

    async def finish(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "audio.done"}))
            except Exception:
                pass
        # _read_loop émettra _emit_done à la fermeture du flux Grok.

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        await self._emit_done()
