from pathlib import Path
from unittest.mock import patch

import pytest

from benji.llm.summary_worker import SummaryWorker


class FakeProvider:
    """Provider de test injecté dans SummaryWorker (cf. SummaryProvider)."""

    name = "fake"

    def __init__(self, summarize):
        self._summarize = summarize

    def summarize(self, entries, on_token=None):
        return self._summarize(entries, on_token=on_token)


@pytest.fixture
def fake_summarize():
    def _summarize(entries, on_token=None):
        for chunk in ["Voici ", "un ", "résumé."]:
            if on_token:
                on_token(chunk)
        return "Voici un résumé."
    return _summarize


def test_full_lifecycle(qtbot, tmp_path, fake_summarize):
    saved_files: list[Path] = []
    def fake_save(text, entries=None):
        p = tmp_path / "summary_1.md"
        p.write_text(text)
        saved_files.append(p)
        return p

    started, chunks, finished, failed = [], [], [], []
    worker = SummaryWorker(provider=FakeProvider(fake_summarize))
    worker.started.connect(lambda sid: started.append(sid))
    worker.chunk.connect(lambda sid, c: chunks.append((sid, c)))
    worker.finished.connect(lambda sid, path: finished.append((sid, path)))
    worker.failed.connect(lambda sid, err: failed.append((sid, err)))

    with patch("benji.llm.summary_worker.save_summary", side_effect=fake_save):
        worker.start()
        worker.request(entries=[{"text": "Texte source"}], summary_id="abc")
        qtbot.waitUntil(lambda: len(finished) == 1, timeout=2000)

    assert started == ["abc"]
    assert chunks == [("abc", "Voici "), ("abc", "un "), ("abc", "résumé.")]
    assert finished == [("abc", saved_files[0])]
    assert failed == []
    worker.shutdown()


def test_failure_emits_failed(qtbot):
    def bad(*a, **kw):
        raise RuntimeError("model crashed")

    failed = []
    worker = SummaryWorker(provider=FakeProvider(bad))
    worker.failed.connect(lambda sid, err: failed.append((sid, err)))

    worker.start()
    worker.request(entries=[{"text": "x"}], summary_id="z")
    qtbot.waitUntil(lambda: len(failed) == 1, timeout=2000)

    assert failed[0][0] == "z"
    assert "model crashed" in failed[0][1]
    worker.shutdown()


def test_empty_summary_emits_failed(qtbot):
    def empty_summarize(entries, on_token=None):
        return None

    failed = []
    worker = SummaryWorker(provider=FakeProvider(empty_summarize))
    worker.failed.connect(lambda sid, err: failed.append((sid, err)))

    worker.start()
    worker.request(entries=[], summary_id="empty")
    qtbot.waitUntil(lambda: len(failed) == 1, timeout=2000)

    assert failed[0][0] == "empty"
    worker.shutdown()
