"""Compte Benji côté app : connexion/inscription + persistance des jetons.

L'abonnement est lié au **compte** (email/mot de passe), pas au poste : se
connecter avec les mêmes identifiants sur n'importe quelle plateforme retrouve
le même plan — le backend rattache le plan à l'utilisateur (cf.
docs/api-contract.md §1-2, docs/cloud-architecture.md).

Les jetons sont stockés dans le trousseau du système via `keyring` (Keychain
macOS / Credential Locker Windows / Secret Service Linux). Si le trousseau est
indisponible, on retombe sur `~/.cache/benji/credentials.json` (chmod 600), même
convention que l'historique — et un fichier hérité est migré vers le trousseau
au premier accès. L'access token (15 min) est rafraîchi à la volée via le refresh
token (30 j) tant que ce dernier est valide.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

def _cred_path() -> Path:
    """Chemin du repli fichier des identifiants (migré depuis `~/.cache/benji`)."""
    from benji.paths import user_path

    return user_path("credentials.json")
_KEYRING_SERVICE = "benji"
_KEYRING_USER = "credentials"


class AuthError(RuntimeError):
    """Échec d'authentification (identifiants, réseau, backend).

    `status` : code HTTP de la réponse, `None` quand le backend n'a pas été
    joint. C'est ce qui sépare un refus (session morte) d'une coupure réseau
    (session intacte, à retenter).
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _jwt_exp(token: str) -> int | None:
    """Lit le champ `exp` d'un JWT sans vérifier la signature (info locale)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # padding base64url
        data = json.loads(base64.urlsafe_b64decode(payload))
        return int(data["exp"])
    except Exception:
        return None


def _error_message(resp) -> str:
    try:
        return resp.json().get("error", {}).get("message") or f"HTTP {resp.status_code}"
    except Exception:
        return f"HTTP {resp.status_code}"


class CredentialStore:
    """Persistance des jetons : trousseau système (keyring) + fallback fichier 0600.

    `use_keyring=False` force le mode fichier (utilisé par les tests pour rester
    hermétiques, sans toucher au vrai trousseau).
    """

    def __init__(self, path: Path | None = None, *, use_keyring: bool = True):
        # Résolu à l'instanciation (et non à l'import) : le chemin dépend du HOME
        # courant, que les tests réécrivent.
        self._path = path or _cred_path()
        self._use_keyring = use_keyring

    def _keyring(self):
        """Module keyring si un backend utilisable est disponible, sinon None."""
        if not self._use_keyring:
            return None
        try:
            import keyring
            from keyring.backends.fail import Keyring as _FailKeyring
            if isinstance(keyring.get_keyring(), _FailKeyring):
                return None  # pas de backend réel (ex. CI headless) → fichier
            return keyring
        except Exception:
            return None

    def load(self) -> dict | None:
        kr = self._keyring()
        if kr is not None:
            try:
                raw = kr.get_password(_KEYRING_SERVICE, _KEYRING_USER)
            except Exception as e:
                log.debug("keyring load échoué, fallback fichier : %s", e)
                raw = None
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
            # Trousseau vide : migrer un éventuel fichier hérité puis l'effacer.
            legacy = self._load_file()
            if legacy is not None:
                try:
                    kr.set_password(_KEYRING_SERVICE, _KEYRING_USER, json.dumps(legacy))
                    self._path.unlink(missing_ok=True)
                    log.info("Jetons migrés du fichier vers le trousseau système.")
                except Exception as e:
                    log.debug("Migration vers keyring échouée : %s", e)
            return legacy
        return self._load_file()

    def save(self, data: dict) -> None:
        kr = self._keyring()
        if kr is not None:
            try:
                kr.set_password(_KEYRING_SERVICE, _KEYRING_USER, json.dumps(data))
                # Pas de copie en clair sur le disque une fois dans le trousseau.
                self._path.unlink(missing_ok=True)
                return
            except Exception as e:
                log.warning("keyring save échoué, fallback fichier : %s", e)
        self._save_file(data)

    def clear(self) -> None:
        kr = self._keyring()
        if kr is not None:
            try:
                kr.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
            except Exception:
                pass
        self._path.unlink(missing_ok=True)

    def _load_file(self) -> dict | None:
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_file(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Mode 0600 dès la création : un write-puis-chmod laisserait le jeton
        # lisible par tous entre les deux appels (umask par défaut).
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data))


class AuthClient:
    """Appels HTTP vers /v1/auth/* et /v1/me (httpx, transport injectable)."""

    def __init__(self, base_url: str, *, timeout: float = 30.0, transport=None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    def _client(self):
        import httpx
        return httpx.Client(timeout=self._timeout, transport=self._transport)

    def _post(self, path: str, payload: dict) -> dict:
        import httpx
        try:
            with self._client() as c:
                resp = c.post(f"{self._base_url}{path}", json=payload)
        except httpx.HTTPError as e:
            raise AuthError(f"Connexion au backend impossible : {e}") from e
        if resp.status_code != 200:
            raise AuthError(_error_message(resp), status=resp.status_code)
        return resp.json()

    def register(self, email: str, password: str) -> dict:
        return self._post("/v1/auth/register", {"email": email, "password": password})

    def login(self, email: str, password: str) -> dict:
        return self._post("/v1/auth/login", {"email": email, "password": password})

    def refresh(self, refresh_token: str) -> dict:
        return self._post("/v1/auth/refresh", {"refresh_token": refresh_token})

    def me(self, access_token: str) -> dict:
        import httpx
        try:
            with self._client() as c:
                resp = c.get(
                    f"{self._base_url}/v1/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
        except httpx.HTTPError as e:
            raise AuthError(f"Connexion au backend impossible : {e}") from e
        if resp.status_code != 200:
            raise AuthError(_error_message(resp), status=resp.status_code)
        return resp.json()


class Session:
    """État de connexion : orchestre store + client, rafraîchit l'access token."""

    _REFRESH_MARGIN_S = 60  # rafraîchit un peu avant l'expiration réelle

    def __init__(self, client: AuthClient, store: CredentialStore | None = None):
        self._client = client
        self._store = store or CredentialStore()
        self._creds = self._store.load()
        # Un seul rafraîchissement à la fois. Les refresh tokens sont **rotatifs**
        # côté backend, avec détection de réutilisation : deux threads (STT
        # distant, facturation, résumé) qui rafraîchissaient en même temps
        # présentaient le même jeton deux fois — le backend y voit un vol et
        # révoque toute la famille, l'utilisateur est déconnecté.
        self._lock = threading.RLock()

    @property
    def is_authenticated(self) -> bool:
        creds = self._creds
        return bool(creds and creds.get("refresh_token"))

    @property
    def email(self) -> str | None:
        return (self._creds or {}).get("email")

    def login(self, email: str, password: str) -> None:
        tokens = self._client.login(email, password)
        with self._lock:
            self._persist(email, tokens)

    def register(self, email: str, password: str) -> None:
        tokens = self._client.register(email, password)
        with self._lock:
            self._persist(email, tokens)

    def logout(self) -> None:
        with self._lock:
            self._creds = None
            self._store.clear()

    def access_token(self) -> str | None:
        """Access token valide (rafraîchi si expiré/proche), ou None si déconnecté.

        None ne veut pas toujours dire « déconnecté » : backend injoignable, la
        session est gardée et l'appelant retentera (cf. ci-dessous).
        """
        with self._lock:
            if not self._creds:
                return None
            access = self._creds.get("access_token")
            exp = _jwt_exp(access) if access else None
            if access and exp and exp - time.time() > self._REFRESH_MARGIN_S:
                return access
            refresh = self._creds.get("refresh_token")
            if not refresh:
                return None
            try:
                tokens = self._client.refresh(refresh)
            except AuthError as e:
                if e.status is None or e.status >= 500 or e.status == 429:
                    # Réseau coupé (réveil de veille avant le Wi-Fi), backend en
                    # panne ou limité : la session n'est pas en cause. La jeter
                    # obligeait à se reconnecter après chaque coupure.
                    log.warning("Rafraîchissement de session impossible pour l'instant : %s", e)
                    return None
                # Refus explicite (expiré, révoqué) → session morte, on nettoie.
                log.info("Session expirée — reconnexion requise.")
                self.logout()
                return None
            self._persist(self._creds.get("email"), tokens)
            return self._creds.get("access_token")

    def _persist(self, email: str | None, tokens: dict) -> None:
        self._creds = {
            "email": email,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
        }
        self._store.save(self._creds)


def build_session(base_url: str, store: CredentialStore | None = None) -> Session:
    return Session(AuthClient(base_url), store=store)
