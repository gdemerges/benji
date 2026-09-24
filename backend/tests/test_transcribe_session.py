"""WS /v1/transcribe : robustesse de la session (panne provider, quota en cours
de route, validation) et paramètres des providers — sans réseau."""

import asyncio
import json

import pytest
from app import deps
from app.routers import transcribe as transcribe_router
from app.stt import deepgram as deepgram_mod
from app.stt import make_session
from app.stt.base import BaseSTTSession
from app.stt.deepgram import DeepgramSTTSession
from app.stt.grok import GrokSTTSession
from starlette.websockets import WebSocketDisconnect

ONE_SECOND = b"\x00" * 32000  # 16 kHz, pcm_s16le


def _start(token, sample_rate=16000):
    return json.dumps({
        "type": "start", "token": token,
        "audio": {"encoding": "pcm_s16le", "sample_rate": sample_rate, "channels": 1},
    })


def _usage(email):
    db = deps.get_db()
    return db.get_usage(db.get_user_by_email(email)["id"])


class _FailingSession(BaseSTTSession):
    """Provider qui tombe au 2e chunk, comme Deepgram après une coupure."""

    def __init__(self):
        super().__init__()
        self._chunks = 0

    async def send_audio(self, chunk: bytes) -> None:
        self._chunks += 1
        if self._chunks >= 2:
            raise ConnectionError("upstream closed")


class _SlowFinalSession(BaseSTTSession):
    """Comme Deepgram : le final arrive *après* `finish`, et `close` coupe la
    lecture sans attendre."""

    async def finish(self) -> None:
        async def later():
            await asyncio.sleep(0.05)
            await self._emit({"type": "final_text", "text": "Dernière phrase"})
            await self._emit_done()
        self._pending = asyncio.create_task(later())


def test_upstream_failure_still_meters(client, pro_token, monkeypatch):
    monkeypatch.setattr(transcribe_router, "make_session", lambda start: _FailingSession())
    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(_start(pro_token))
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(ONE_SECOND)
        ws.send_bytes(ONE_SECOND)  # le provider tombe ici
        assert ws.receive_json()["code"] == "upstream_error"
        closed = ws.receive_json()
        assert closed["type"] == "closed" and closed["stt_seconds"] == pytest.approx(2.0)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1011
    assert _usage("pro@b.c") == pytest.approx(2.0)


def test_quota_enforced_mid_session(client, pro_token, monkeypatch):
    monkeypatch.setenv("STT_BACKEND", "fake")
    db = deps.get_db()
    uid = db.get_user_by_email("pro@b.c")["id"]
    db.add_usage(uid, 36000 - 1.5)  # plafond pro = 36000 s, reste 1,5 s

    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(_start(pro_token))
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(ONE_SECOND)
        ws.send_bytes(ONE_SECOND)  # franchit le plafond
        events = []
        with pytest.raises(WebSocketDisconnect) as exc:
            while True:
                events.append(ws.receive_json())
    assert exc.value.code == 4429
    assert {"type": "error", "code": "quota_exceeded", "message": "Quota STT atteint."} in events
    # Le final du segment en cours est quand même rendu, et la conso comptée.
    assert any(e["type"] == "final_text" for e in events)
    assert _usage("pro@b.c") == pytest.approx(36000 + 0.5)


@pytest.mark.parametrize("rate", [0, 12345, "16000", 10**9])
def test_invalid_sample_rate_rejected(client, pro_token, monkeypatch, rate):
    monkeypatch.setenv("STT_BACKEND", "fake")
    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(_start(pro_token, sample_rate=rate))
        assert ws.receive_json()["code"] == "bad_request"


def test_stop_waits_for_last_final(client, pro_token, monkeypatch):
    monkeypatch.setattr(transcribe_router, "make_session", lambda start: _SlowFinalSession())
    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(_start(pro_token))
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(ONE_SECOND)
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "final_text", "text": "Dernière phrase"}
        assert ws.receive_json()["type"] == "closed"


# --- paramètres des providers ---

def test_deepgram_requests_utterance_end():
    assert DeepgramSTTSession("k")._params["utterance_end_ms"] == "1000"


def test_auto_language_is_not_sent_as_none():
    assert DeepgramSTTSession("k", language=None)._params["language"] == "multi"
    assert "language" not in GrokSTTSession("k", language=None)._params
    assert GrokSTTSession("k")._params["language"] == "fr"


def test_make_session_language(monkeypatch):
    monkeypatch.setenv("STT_BACKEND", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    assert make_session({"language": None})._params["language"] == "multi"
    assert make_session({})._params["language"] == "fr"


def test_deepgram_keepalive_during_silence(monkeypatch):
    monkeypatch.setattr(deepgram_mod, "_KEEPALIVE_S", 0.02)

    class _WS:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    async def run():
        s = DeepgramSTTSession("k")
        s._ws = _WS()
        task = asyncio.create_task(s._keepalive_loop())
        await asyncio.sleep(0.1)  # aucun audio : pause du micro
        task.cancel()
        return s._ws.sent

    sent = asyncio.run(run())
    assert sent and all(json.loads(p) == {"type": "KeepAlive"} for p in sent)


def test_deepgram_no_keepalive_while_audio_flows(monkeypatch):
    monkeypatch.setattr(deepgram_mod, "_KEEPALIVE_S", 0.05)

    class _WS:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    async def run():
        s = DeepgramSTTSession("k")
        s._ws = _WS()
        task = asyncio.create_task(s._keepalive_loop())
        for _ in range(10):
            await s.send_audio(b"\x00\x00")
            await asyncio.sleep(0.01)
        task.cancel()
        return s._ws.sent

    sent = asyncio.run(run())
    assert not any(isinstance(p, str) for p in sent)  # que de l'audio binaire
