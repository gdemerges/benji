"""WebSocket /v1/transcribe — transcription temps réel (cf. api-contract §3).

Handshake `start`/`ready` + auth, puis deux flux concurrents :
- *feeder* : frames audio binaires du client → session STT (et métering)
- *forwarder* : events de la session (vad/segment_start/word/final_text) → client

Le provider STT concret est choisi par `app.stt.make_session` (Deepgram en prod,
Fake en test).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.auth import authenticate
from app.deps import get_db
from app.stt import make_session

log = logging.getLogger(__name__)
router = APIRouter()

# Codes de fermeture WS (cf. api-contract §6).
WS_UNAUTHENTICATED = 4401
WS_FORBIDDEN = 4403
WS_QUOTA = 4429

_BYTES_PER_SAMPLE = 2  # pcm_s16le
# Fréquences acceptées : le métering se calcule dessus, une valeur arbitraire
# fausserait la conso facturée (et les providers la refuseraient de toute façon).
_SAMPLE_RATES = {8000, 16000, 24000, 44100, 48000}


async def _send(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def _forward(session, ws: WebSocket) -> None:
    async for event in session.events():
        await _send(ws, event)


@router.websocket("/v1/transcribe")
async def transcribe(ws: WebSocket) -> None:
    await ws.accept()

    # 1) Premier message : `start` (auth + config audio).
    try:
        start = json.loads(await ws.receive_text())
    except (WebSocketDisconnect, json.JSONDecodeError):
        await ws.close(code=WS_UNAUTHENTICATED)
        return
    if start.get("type") != "start":
        await ws.close(code=WS_UNAUTHENTICATED)
        return

    db = get_db()
    user = authenticate(start.get("token"), db)
    if user is None:
        await ws.close(code=WS_UNAUTHENTICATED)
        return
    if not user.cloud_stt:
        await ws.close(code=WS_FORBIDDEN)
        return

    # Quota STT (le poste facturable) : refus si le plafond du plan est atteint.
    limit = user.stt_seconds_limit
    used_before = db.get_usage(user.user_id)
    if limit is not None and used_before >= limit:
        await _send(ws, {"type": "error", "code": "quota_exceeded",
                         "message": "Quota STT atteint."})
        await ws.close(code=WS_QUOTA)
        return

    audio = start.get("audio") or {}
    sample_rate = audio.get("sample_rate", 16000)
    if sample_rate not in _SAMPLE_RATES:
        await _send(ws, {"type": "error", "code": "bad_request",
                         "message": "sample_rate non pris en charge."})
        await ws.close()
        return
    bytes_per_second = sample_rate * _BYTES_PER_SAMPLE

    # 2) Ouverture de la session STT.
    try:
        session = make_session(start)
        await session.open()
    except Exception as e:
        log.warning("STT session open failed: %s", e)
        await _send(ws, {"type": "error", "code": "upstream_error", "message": str(e)})
        await ws.close()
        return

    await _send(ws, {"type": "ready"})
    forwarder = asyncio.create_task(_forward(session, ws))

    # 3) Feeder : frames audio + contrôle.
    total_bytes = 0
    # Erreur à signaler au client en fin de session : (code, message, code WS).
    failure: tuple[str, str, int] | None = None
    # `finish` appelé : le provider doit encore rendre ses derniers finals.
    finished = False
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                total_bytes += len(data)
                await session.send_audio(data)
                # Le quota vaut aussi en cours de session : sans ce contrôle, une
                # session ouverte sous le plafond pouvait le dépasser sans limite.
                if limit is not None and used_before + total_bytes / bytes_per_second >= limit:
                    await session.finish()
                    finished = True
                    failure = ("quota_exceeded", "Quota STT atteint.", WS_QUOTA)
                    break
                continue
            if (text := msg.get("text")) is not None:
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if ctrl.get("type") == "stop":
                    await session.finish()
                    finished = True
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Provider tombé en cours de route (réseau, fermeture côté Deepgram/Grok) :
        # on sort proprement pour que l'audio déjà transcrit soit bien compté.
        log.warning("STT session failed: %s", e)
        failure = ("upstream_error", "Le service de transcription a interrompu la session.", 1011)
    finally:
        # Arrêt demandé : le provider rend encore le final du segment en cours
        # après `finish` — le laisser finir avant `close`, qui coupe sa lecture
        # (sinon la dernière phrase de chaque session était perdue). Sur
        # déconnexion ou panne, rien à attendre : on ferme d'abord.
        if finished:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(forwarder), timeout=5)
        with contextlib.suppress(Exception):
            await session.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(forwarder, timeout=5)  # draine les derniers events
        if not forwarder.done():
            forwarder.cancel()

        # 4) Clôture : conso facturable (secondes d'audio reçues) → métering.
        # Dans le `finally` : aucune sortie, même sur exception, ne doit laisser
        # de l'audio transcrit hors du compteur.
        stt_seconds = round(total_bytes / bytes_per_second, 1)
        if stt_seconds > 0:
            try:
                db.add_usage(user.user_id, stt_seconds)
            except Exception:
                log.exception("Métering STT perdu (%.1f s)", stt_seconds)

    # Client déjà parti : rien à lui dire (RuntimeError ou WebSocketDisconnect
    # selon la version de Starlette).
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        if failure is not None:
            await _send(ws, {"type": "error", "code": failure[0], "message": failure[1]})
        await _send(ws, {"type": "closed", "stt_seconds": stt_seconds})
        await ws.close(code=failure[2] if failure else 1000)
