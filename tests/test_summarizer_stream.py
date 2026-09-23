"""Le flux du résumé ne doit verser que du texte, jamais la représentation d'un objet."""

import sys
import types
from dataclasses import dataclass

from benji.llm import summarizer


@dataclass
class _Response:
    text: str
    token: int = 0


def test_final_empty_response_is_not_stringified(monkeypatch):
    """Le dernier `GenerationResponse` de mlx-lm porte le jeton de fin et un
    texte vide : il ne doit pas finir dans le résumé sous forme de `repr`."""
    fake = types.ModuleType("mlx_lm")
    fake.stream_generate = lambda *a, **k: iter(
        [_Response("Sujets"), _Response(" abordés"), _Response("", token=151645)]
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    pieces: list[str] = []
    result = summarizer._stream(None, None, "prompt", pieces.append)

    assert result == "Sujets abordés"
    assert pieces == ["Sujets", " abordés"]
