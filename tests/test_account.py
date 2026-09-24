"""Session/AuthClient : login, refresh auto, persistance — sans réseau (httpx mocké)."""

import base64
import json
import time

import httpx
import pytest

from benji.account import AuthClient, AuthError, CredentialStore, Session


def _jwt(exp: int) -> str:
    """Forge un JWT non signé avec un `exp` donné (lecture locale seulement)."""
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{b64({'alg': 'HS256'})}.{b64({'exp': exp})}.sig"


def _transport(handler):
    return httpx.MockTransport(handler)


def _store(tmp_path):
    # use_keyring=False : les tests restent hermétiques (fichier temporaire),
    # sans toucher au trousseau système réel.
    return CredentialStore(tmp_path / "creds.json", use_keyring=False)


def test_login_persists_tokens(tmp_path):
    access, refresh = _jwt(int(time.time()) + 900), _jwt(int(time.time()) + 99999)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/auth/login"
        assert json.loads(request.content) == {"email": "u@b.c", "password": "pw"}
        return httpx.Response(200, json={
            "access_token": access, "refresh_token": refresh,
            "expires_in": 900, "token_type": "Bearer",
        })

    session = Session(AuthClient("http://test", transport=_transport(handler)),
                      store=_store(tmp_path))
    assert not session.is_authenticated
    session.login("u@b.c", "pw")
    assert session.is_authenticated
    assert session.email == "u@b.c"
    assert session.access_token() == access

    # Persisté : une nouvelle session relit les jetons du disque.
    reloaded = Session(AuthClient("http://test"), store=_store(tmp_path))
    assert reloaded.is_authenticated
    assert reloaded.email == "u@b.c"


def test_bad_credentials_raise(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "unauthenticated",
                                                   "message": "Identifiants invalides."}})

    session = Session(AuthClient("http://test", transport=_transport(handler)),
                      store=_store(tmp_path))
    with pytest.raises(AuthError, match="Identifiants invalides"):
        session.login("u@b.c", "bad")
    assert not session.is_authenticated


def test_expired_access_token_triggers_refresh(tmp_path):
    expired = _jwt(int(time.time()) - 10)
    fresh = _jwt(int(time.time()) + 900)
    refresh_tok = _jwt(int(time.time()) + 99999)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/auth/login":
            return httpx.Response(200, json={"access_token": expired,
                                             "refresh_token": refresh_tok, "expires_in": 900})
        if request.url.path == "/v1/auth/refresh":
            assert json.loads(request.content) == {"refresh_token": refresh_tok}
            return httpx.Response(200, json={"access_token": fresh,
                                             "refresh_token": refresh_tok, "expires_in": 900})
        raise AssertionError(request.url.path)

    session = Session(AuthClient("http://test", transport=_transport(handler)),
                      store=_store(tmp_path))
    session.login("u@b.c", "pw")
    # access expiré → access_token() rafraîchit et renvoie le nouveau.
    assert session.access_token() == fresh
    assert "/v1/auth/refresh" in calls


def test_dead_refresh_logs_out(tmp_path):
    expired = _jwt(int(time.time()) - 10)
    refresh_tok = _jwt(int(time.time()) + 99999)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/auth/login":
            return httpx.Response(200, json={"access_token": expired,
                                             "refresh_token": refresh_tok, "expires_in": 900})
        return httpx.Response(401, json={"error": {"message": "refresh expiré"}})

    store = _store(tmp_path)
    session = Session(AuthClient("http://test", transport=_transport(handler)), store=store)
    session.login("u@b.c", "pw")
    # Refresh refusé → session nettoyée, jetons effacés du disque.
    assert session.access_token() is None
    assert not session.is_authenticated
    assert store.load() is None


def test_logout_clears_store(tmp_path):
    access = _jwt(int(time.time()) + 900)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": access,
                                         "refresh_token": _jwt(int(time.time()) + 99999),
                                         "expires_in": 900})

    store = _store(tmp_path)
    session = Session(AuthClient("http://test", transport=_transport(handler)), store=store)
    session.login("u@b.c", "pw")
    session.logout()
    assert not session.is_authenticated
    assert store.load() is None


def _session_with_expired_access(tmp_path, refresh_handler):
    """Session connectée dont l'access token est expiré : le prochain
    `access_token()` passe par le refresh, servi par `refresh_handler`."""
    expired, refresh_tok = _jwt(int(time.time()) - 10), _jwt(int(time.time()) + 99999)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/auth/login":
            return httpx.Response(200, json={"access_token": expired,
                                             "refresh_token": refresh_tok, "expires_in": 900})
        return refresh_handler(request)

    session = Session(AuthClient("http://test", transport=_transport(handler)),
                      store=_store(tmp_path))
    session.login("u@b.c", "pw")
    return session


def test_network_error_keeps_session(tmp_path):
    """Réveil de veille sans Wi-Fi : le refresh échoue faute de réseau. La
    session ne doit pas être jetée — elle déconnectait l'utilisateur."""
    def handler(request):
        raise httpx.ConnectError("offline")

    session = _session_with_expired_access(tmp_path, handler)
    assert session.access_token() is None
    assert session.is_authenticated
    assert Session(AuthClient("http://test"), store=_store(tmp_path)).is_authenticated


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_server_error_keeps_session(tmp_path, status):
    session = _session_with_expired_access(
        tmp_path, lambda request: httpx.Response(status, json={})
    )
    assert session.access_token() is None
    assert session.is_authenticated


def test_concurrent_refresh_presents_refresh_token_once(tmp_path):
    """Refresh rotatif + détection de réutilisation côté backend : deux threads
    qui rafraîchissent ensemble ne doivent présenter le jeton qu'une fois."""
    import threading

    fresh_access = _jwt(int(time.time()) + 900)
    presented = []
    gate = threading.Event()

    def handler(request):
        presented.append(json.loads(request.content)["refresh_token"])
        gate.wait(0.2)  # élargit la fenêtre de concurrence
        return httpx.Response(200, json={"access_token": fresh_access,
                                         "refresh_token": _jwt(int(time.time()) + 99999),
                                         "expires_in": 900})

    session = _session_with_expired_access(tmp_path, handler)
    results = []
    threads = [threading.Thread(target=lambda: results.append(session.access_token()))
               for _ in range(4)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    assert len(presented) == 1
    assert results == [fresh_access] * 4
