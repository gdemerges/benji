import json

import pytest
from app.routers import summary as summary_router

LONG_ENTRIES = [{"text": "Bonjour, ceci est une transcription suffisamment longue pour être résumée."}]


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_me_requires_auth(client):
    r = client.get("/v1/me")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


def test_me_ok(client, auth):
    r = client.get("/v1/me", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "free"
    assert body["entitlements"]["cloud_summary"] is True
    assert body["entitlements"]["cloud_stt"] is False  # gratuit : pas de STT cloud
    assert body["quota"]["stt_seconds_used"] == 0


def test_summary_requires_auth(client):
    r = client.post("/v1/summary", json={"entries": LONG_ENTRIES})
    assert r.status_code == 401


def test_summary_rejects_short_transcription(client, auth, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = client.post("/v1/summary", json={"entries": [{"text": "court"}]}, headers=auth)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_summary_missing_key_is_500(client, auth, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/v1/summary", json={"entries": LONG_ENTRIES}, headers=auth)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal"


def test_summary_streams_sse(client, auth, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    async def fake_stream(text, model):
        assert model == "claude-haiku-4-5"
        yield "event: token\ndata: {\"text\": \"Voici \"}\n\n"
        yield "event: done\ndata: {\"summary_id\": \"sum_x\"}\n\n"

    monkeypatch.setattr(summary_router, "_stream_summary", fake_stream)

    with client.stream("POST", "/v1/summary",
                       json={"entries": LONG_ENTRIES, "model": "haiku"},
                       headers=auth) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "event: token" in body and "event: done" in body


def test_transcribe_streams_events_and_meters(client, pro_token, monkeypatch):
    monkeypatch.setenv("STT_BACKEND", "fake")
    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(json.dumps({
            "type": "start", "token": pro_token,
            "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        }))
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(b"\x00" * 32000)  # 1.0 s
        assert ws.receive_json() == {"type": "vad_status", "speaking": True}
        assert ws.receive_json() == {"type": "segment_start"}
        for w in ("bonjour", "le", "monde"):
            assert ws.receive_json() == {"type": "word", "text": w}
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "final_text", "text": "Bonjour le monde"}
        assert ws.receive_json() == {"type": "vad_status", "speaking": False}
        closed = ws.receive_json()
        assert closed["type"] == "closed" and closed["stt_seconds"] == pytest.approx(1.0)

    # La conso a été persistée et est visible dans /me.
    me = client.get("/v1/me", headers={"Authorization": f"Bearer {pro_token}"}).json()
    assert me["quota"]["stt_seconds_used"] == 1


def test_transcribe_free_plan_forbidden(client, token):
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(json.dumps({"type": "start", "token": token}))  # plan free
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4403


def test_transcribe_rejects_unauthenticated(client):
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect("/v1/transcribe") as ws:
        ws.send_text(json.dumps({"type": "start"}))  # pas de token
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4401


def test_summary_prompt_carries_speakers():
    from app import prompts

    text = prompts.prepare_transcription([
        {"text": LONG_ENTRIES[0]["text"], "speaker": "Marie"},
        {"text": "D'accord.", "speaker": None},
    ])
    assert text.splitlines()[0].startswith("Marie : Bonjour")
    assert text.splitlines()[1] == "D'accord."
    assert "locuteur" in prompts.build_user_prompt(text)
