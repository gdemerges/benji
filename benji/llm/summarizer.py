"""Session summarizer using MLX-LM (Apple Silicon)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MAX_TOKENS = 512

def _get_model():
    """Modèle partagé avec le correcteur (cf. benji/llm/model_cache.py) : même
    MODEL_ID, donc mêmes poids en mémoire au lieu de deux copies."""
    from benji.llm import model_cache

    return model_cache.load(MODEL_ID)


# Source unique de vérité du prompt, partagée par le provider local (mlx-lm) et
# le provider cloud (Claude) — voir benji/llm/providers.py.
SYSTEM_PROMPT = (
    "Tu es un assistant qui résume des conversations. "
    "Réponds uniquement en français, de façon concise et structurée."
)


def build_user_prompt(transcription_text: str) -> str:
    return (
        "Voici la transcription d'une conversation. "
        "Génère un résumé structuré avec :\n"
        "- **Sujets abordés** : les thèmes principaux\n"
        "- **Points clés** : les informations importantes\n"
        "- **Décisions / Actions** : les décisions prises ou actions à faire (si applicable)\n\n"
        "Chaque réplique peut être précédée de son locuteur (« Marie : … ») ; "
        "une lettre seule (« A : … ») est un locuteur anonyme. Attribue les "
        "propositions, décisions et actions à la personne concernée quand "
        "c'est clair.\n\n"
        "Sois factuel et concis. "
        "Si la transcription est trop courte pour être résumée, dis-le simplement.\n\n"
        "Transcription :\n<transcription>\n"
        f"{transcription_text}"
        "\n</transcription>"
    )


def with_speaker_names(entries: list[dict]) -> list[dict]:
    """Copies des entrées où l'étiquette du locuteur est remplacée par son nom.

    Les noms vivent dans le registre des réunions (`benji/meetings.py`), un jeu
    par réunion : on les lit une fois par réunion présente dans `entries`. Un
    registre illisible n'empêche pas de résumer — on garde les étiquettes.
    """
    from benji import meetings

    names_by_meeting: dict[str, dict[str, str]] = {}
    out = []
    for entry in entries:
        speaker, meeting_id = entry.get("speaker"), entry.get("meeting")
        if speaker and meeting_id:
            if meeting_id not in names_by_meeting:
                try:
                    names_by_meeting[meeting_id] = meetings.speaker_names(meeting_id)
                except Exception:
                    log.exception("Noms de locuteurs illisibles — étiquettes gardées")
                    names_by_meeting[meeting_id] = {}
            name = names_by_meeting[meeting_id].get(speaker)
            if name:
                entry = {**entry, "speaker": name}
        out.append(entry)
    return out


def prepare_transcription(entries: list[dict]) -> str | None:
    """Concatène les utterances et écarte les sessions trop courtes.

    Chaque réplique est précédée de son locuteur — nommé si l'utilisateur l'a
    fait — pour que le résumé puisse dire *qui* a proposé ou décidé quoi.
    Retourne le texte prêt à résumer, ou None si rien d'exploitable.
    """
    if not entries:
        log.info("Aucune transcription à résumer.")
        return None
    # Le seuil porte sur ce qui a été dit, pas sur les préfixes de locuteur.
    if len("\n".join(e["text"] for e in entries).strip()) < 50:
        log.info("Transcription trop courte pour être résumée.")
        return None
    return "\n".join(
        f"{e['speaker']} : {e['text']}" if e.get("speaker") else e["text"]
        for e in with_speaker_names(entries)
    )


def _build_prompt(tokenizer, transcription_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(transcription_text)},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for tokenizers without chat template
    return f"Résume cette conversation en français :\n\n{transcription_text}\n\nRésumé :"


def summarize(
    entries: list[dict],
    on_token: Callable[[str], None] | None = None,
) -> str | None:
    """Generate a summary of a transcription session using MLX-LM.

    If `on_token` is provided, streams the output token-by-token via
    `mlx_lm.stream_generate` and calls the callback for each chunk.
    Returns the full summary text once generation completes.
    """
    transcription_text = prepare_transcription(entries)
    if transcription_text is None:
        return None

    try:
        import mlx_lm  # noqa: F401 — présence seule ; l'usage est sur le fil MLX
    except ImportError:
        log.warning("mlx-lm non installé. Exécute : uv sync")
        return None

    from benji.llm import mlx_runner

    model, tokenizer = _get_model()

    log.info("Génération du résumé...")
    prompt = _build_prompt(tokenizer, transcription_text)

    if on_token is None:
        return mlx_runner.generate(model, tokenizer, prompt, MAX_TOKENS).strip()

    # La boucle entière part sur le fil MLX : `stream_generate` ouvre le stream
    # de mlx-lm à chaque itération, pas seulement au premier appel. `on_token`
    # est donc appelé depuis ce fil — les appelants n'en font qu'un signal Qt.
    return mlx_runner.run(_stream, model, tokenizer, prompt, on_token)


def _stream(model, tokenizer, prompt: str, on_token: Callable[[str], None]) -> str:
    """Exécutée *sur* le fil MLX (cf. benji/llm/mlx_runner.py)."""
    from mlx_lm import stream_generate

    chunks: list[str] = []
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS):
        # mlx_lm.stream_generate yields a `GenerationResponse` with a `.text` field
        # (incremental text since the previous yield). Le dernier porte le jeton
        # de fin et un texte vide : un `or str(response)` versait alors toute la
        # représentation de l'objet (logprobs, tokens/s…) à la fin du résumé.
        piece = response if isinstance(response, str) else getattr(response, "text", "")
        if piece:
            chunks.append(piece)
            try:
                on_token(piece)
            except Exception as e:
                log.warning("on_token callback failed: %s", e)
    return "".join(chunks).strip()


# Qui résume quoi : nom de fichier → réunions dont il contient des phrases. Un
# index à côté plutôt qu'un nom de fichier enrichi, que l'onglet Résumés
# reconnaît par motif (`summary_AAAAMMJJ_HHMMSS.md`) et ignore ce fichier-ci.
_SUMMARY_INDEX = ".meetings.json"


def _summaries_dir() -> Path:
    from benji.paths import user_path

    return user_path("summaries")


def _write_private(path: Path, text: str) -> None:
    # Mode 0600 dès la création (un write-puis-chmod laisserait le contenu
    # lisible par tous entre les deux appels).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def _read_index(directory: Path) -> dict[str, list[str]]:
    import json

    try:
        data = json.loads((directory / _SUMMARY_INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(m) for m in v] for k, v in data.items() if isinstance(v, list)}


def _write_index(directory: Path, index: dict[str, list[str]]) -> None:
    import json

    tmp = directory / (_SUMMARY_INDEX + ".tmp")
    _write_private(tmp, json.dumps(index, ensure_ascii=False))
    os.replace(tmp, directory / _SUMMARY_INDEX)


def save_summary(summary: str, entries: list[dict] | None = None) -> Path:
    """Save the summary to a timestamped markdown file.

    `entries` : les phrases résumées. Leurs réunions sont notées dans l'index,
    pour qu'effacer une réunion emporte aussi ses résumés (`delete_summaries`).
    """
    from benji import meetings

    cache_dir = _summaries_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    stem = f"summary_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    filename = f"{stem}.md"
    # Deux résumés dans la même seconde (réunion + direct) : le second écrasait
    # le premier. L'onglet Résumés date un nom suffixé par sa mtime.
    n = 2
    while (cache_dir / filename).exists():
        filename = f"{stem}_{n}.md"
        n += 1
    file_path = cache_dir / filename

    _write_private(
        file_path,
        f"# Résumé de session — {timestamp.strftime('%d/%m/%Y %H:%M')}\n\n{summary}\n",
    )

    meeting_ids = sorted({e.get("meeting") or meetings.LEGACY_ID for e in entries or []})
    if meeting_ids:
        index = _read_index(cache_dir)
        index[filename] = meeting_ids
        _write_index(cache_dir, index)

    return file_path


def delete_summaries(meeting_id: str) -> int:
    """Efface les résumés qui contiennent des phrases de `meeting_id`.

    Un résumé qui couvre plusieurs réunions part aussi : il cite la réunion
    effacée. Les résumés d'avant l'index ne sont rattachés à rien et restent —
    on ne devine pas leur réunion. Retourne le nombre de fichiers effacés.
    """
    directory = _summaries_dir()
    index = _read_index(directory)
    doomed = [name for name, ids in index.items() if meeting_id in ids]
    if not doomed:
        return 0
    for name in doomed:
        (directory / name).unlink(missing_ok=True)
        del index[name]
    _write_index(directory, index)
    return len(doomed)
