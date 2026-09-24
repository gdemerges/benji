"""Abstraction de provider pour le résumé : moteur local (mlx-lm) ou cloud (Claude).

Le `SummaryWorker` ne connaît que l'interface `SummaryProvider` ; le choix
concret vient de `LLMConfig` (cf. `benji/config.py`). Le mode local reste le
défaut — rien ne sort du Mac tant que l'utilisateur n'active pas le cloud.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from benji.llm import summarizer

log = logging.getLogger(__name__)

OnToken = Callable[[str], None]


@runtime_checkable
class SummaryProvider(Protocol):
    name: str

    def summarize(
        self, entries: list[dict], on_token: OnToken | None = None
    ) -> str | None: ...


class LocalSummaryProvider:
    """Résumé via mlx-lm sur Apple Silicon (défaut, 100 % local)."""

    name = "local"

    def summarize(
        self, entries: list[dict], on_token: OnToken | None = None
    ) -> str | None:
        return summarizer.summarize(entries, on_token=on_token)


class CloudSummaryProvider:
    """Résumé via l'API Claude (Anthropic), en streaming.

    La transcription (texte uniquement) quitte le poste — usage opt-in.
    """

    name = "cloud"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ):
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = None  # construit paresseusement (1er résumé cloud)

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "Le résumé cloud nécessite le paquet 'anthropic'. "
                    "Installe-le : uv sync --extra cloud"
                ) from e
            # api_key=None → l'SDK résout ANTHROPIC_API_KEY depuis l'environnement.
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    def summarize(
        self, entries: list[dict], on_token: OnToken | None = None
    ) -> str | None:
        transcription_text = summarizer.prepare_transcription(entries)
        if transcription_text is None:
            return None

        client = self._get_client()
        log.info("Génération du résumé via Claude (%s)…", self._model)

        chunks: list[str] = []
        with client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=summarizer.SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": summarizer.build_user_prompt(transcription_text),
                }
            ],
        ) as stream:
            for delta in stream.text_stream:
                if not delta:
                    continue
                chunks.append(delta)
                if on_token is not None:
                    try:
                        on_token(delta)
                    except Exception as e:
                        log.warning("on_token callback failed: %s", e)

        return "".join(chunks).strip() or None


class RemoteSummaryProvider:
    """Résumé via le backend Benji (POST /v1/summary, SSE).

    La transcription part vers le backend (jamais la clé Anthropic, qui reste
    côté serveur). Le backend résout l'alias de modèle. Cf. docs/api-contract.md.
    """

    name = "remote"

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        model_alias: str = "haiku",
        timeout: float = 120.0,
        transport=None,  # httpx transport injectable (tests)
        token_provider: Callable[[], str | None] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        # Appelé à chaque résumé : l'access token expire en 15 min, et un jeton
        # figé au lancement faisait échouer (401) tout résumé demandé en fin de
        # réunion. Le jeton statique ne sert plus que de repli (tests, dev).
        self._token_provider = token_provider
        self._model_alias = model_alias
        self._timeout = timeout
        self._transport = transport

    def summarize(
        self, entries: list[dict], on_token: OnToken | None = None
    ) -> str | None:
        # Court-circuit local : inutile d'appeler le réseau pour une session vide
        # ou trop courte (le backend renverrait 400 de toute façon).
        if summarizer.prepare_transcription(entries) is None:
            return None

        import httpx

        url = f"{self._base_url}/v1/summary"
        token = self._token_provider() if self._token_provider is not None else None
        token = token or self._token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = {
            "entries": [
                {
                    "text": e.get("text", ""),
                    "timestamp": e.get("timestamp"),
                    "speaker": e.get("speaker"),
                }
                # Les noms donnés aux locuteurs voyagent à la place des
                # étiquettes : le serveur n'a pas accès au registre local.
                for e in summarizer.with_speaker_names(entries)
            ],
            "model": self._model_alias,
        }

        chunks: list[str] = []
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise RuntimeError(
                        f"Backend a répondu {resp.status_code}: {resp.text}"
                    )
                event: str | None = None
                for line in resp.iter_lines():
                    if not line:
                        event = None
                        continue
                    if line.startswith("event:"):
                        event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[len("data:"):].strip())
                        if event == "token":
                            piece = data.get("text", "")
                            if piece:
                                chunks.append(piece)
                                if on_token is not None:
                                    try:
                                        on_token(piece)
                                    except Exception as e:
                                        log.warning("on_token callback failed: %s", e)
                        elif event == "error":
                            raise RuntimeError(
                                data.get("message", "Erreur backend inconnue.")
                            )
                        # event == "done" : rien à faire de plus

        return "".join(chunks).strip() or None


def build_summary_provider(cfg, token_provider=None) -> SummaryProvider:
    """Construit le provider de résumé d'après `LLMConfig`.

    `token_provider` rend un access token valide (`Session.access_token`) ; il
    est rappelé à chaque résumé distant.
    """
    provider = getattr(cfg, "summary_provider", "local")
    if provider == "cloud":
        return CloudSummaryProvider(
            model=cfg.cloud_model,
            api_key=cfg.anthropic_api_key,
            max_tokens=cfg.cloud_max_tokens,
        )
    if provider == "remote":
        return RemoteSummaryProvider(
            base_url=cfg.backend_url,
            token=cfg.backend_token,
            model_alias=cfg.summary_model_alias,
            token_provider=token_provider,
        )
    return LocalSummaryProvider()
