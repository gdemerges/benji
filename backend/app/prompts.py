"""Prompt de résumé côté serveur.

Miroir de `benji/llm/summarizer.py` (provider local). Les deux runtimes — mlx
local et Claude cloud — partagent volontairement le même prompt ; garder les
deux fichiers en phase si l'un change. (Quand le client desktop appellera le
backend pour le résumé cloud, cette duplication disparaîtra.)
"""

from __future__ import annotations

from app.config import MIN_TRANSCRIPTION_CHARS

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


def prepare_transcription(entries: list[dict]) -> str | None:
    """Concatène les utterances, chacune précédée de son locuteur s'il est
    connu ; None si rien d'exploitable (trop court — le seuil porte sur le
    texte dit, pas sur les préfixes)."""
    if not entries:
        return None
    text = "\n".join(e.get("text", "") for e in entries)
    if len(text.strip()) < MIN_TRANSCRIPTION_CHARS:
        return None
    return "\n".join(
        f"{e['speaker']} : {e.get('text', '')}" if e.get("speaker") else e.get("text", "")
        for e in entries
    )
