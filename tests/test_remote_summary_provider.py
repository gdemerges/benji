"""RemoteSummaryProvider : parsing SSE du backend, sans réseau (httpx mocké)."""

import httpx
import pytest

from benji.llm.providers import RemoteSummaryProvider, build_summary_provider

LONG = [{"text": "Bonjour, ceci est une transcription assez longue pour être résumée."}]

_SSE_OK = (
    "event: token\n"
    'data: {"text": "Voici "}\n'
    "\n"
    "event: token\n"
    'data: {"text": "un résumé."}\n'
    "\n"
    "event: done\n"
    'data: {"summary_id": "sum_x"}\n'
    "\n"
)


def _transport(handler):
    return httpx.MockTransport(handler)


def test_streams_tokens_and_returns_text():
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200, text=_SSE_OK, headers={"content-type": "text/event-stream"}
        )

    provider = RemoteSummaryProvider(
        base_url="http://test", token="tok", model_alias="haiku",
        transport=_transport(handler),
    )
    tokens = []
    result = provider.summarize(LONG, on_token=tokens.append)

    assert result == "Voici un résumé."
    assert tokens == ["Voici ", "un résumé."]
    # Le provider a bien tapé l'endpoint avec l'alias + le Bearer.
    req = seen_requests[0]
    assert req.url.path == "/v1/summary"
    assert req.headers["authorization"] == "Bearer tok"


def test_error_event_raises():
    def handler(request):
        return httpx.Response(
            200,
            text='event: error\ndata: {"code": "upstream_error", "message": "boom"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    provider = RemoteSummaryProvider("http://test", transport=_transport(handler))
    with pytest.raises(RuntimeError, match="boom"):
        provider.summarize(LONG)


def test_non_200_raises():
    def handler(request):
        return httpx.Response(403, json={"error": {"code": "forbidden", "message": "no"}})

    provider = RemoteSummaryProvider("http://test", transport=_transport(handler))
    with pytest.raises(RuntimeError, match="403"):
        provider.summarize(LONG)


def test_short_transcription_short_circuits_without_network():
    called = False

    def handler(request):
        nonlocal called
        called = True
        return httpx.Response(200, text="")

    provider = RemoteSummaryProvider("http://test", transport=_transport(handler))
    assert provider.summarize([{"text": "court"}]) is None
    assert called is False  # aucun appel réseau pour une transcription trop courte


def test_factory_builds_remote():
    from benji.config import LLMConfig

    p = build_summary_provider(LLMConfig(summary_provider="remote",
                                         backend_url="http://x", summary_model_alias="sonnet"))
    assert isinstance(p, RemoteSummaryProvider)
    assert p.name == "remote"


def test_token_provider_is_called_per_summary():
    """Un jeton figé au lancement expirait en 15 min : chaque résumé redemande
    un access token valide à la session."""
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, text=_SSE_OK, headers={"content-type": "text/event-stream"})

    tokens = iter(["frais-1", "frais-2"])
    provider = RemoteSummaryProvider(
        "http://test", token="perime", transport=_transport(handler),
        token_provider=lambda: next(tokens),
    )
    provider.summarize(LONG)
    provider.summarize(LONG)
    assert seen == ["Bearer frais-1", "Bearer frais-2"]


def test_factory_passes_token_provider():
    from benji.config import LLMConfig

    p = build_summary_provider(
        LLMConfig(summary_provider="remote", backend_url="http://x"),
        token_provider=lambda: "tok",
    )
    assert p._token_provider() == "tok"
