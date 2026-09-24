"""QThread async qui exécute summarize() + save_summary() sans bloquer l'UI."""

from __future__ import annotations

import logging
from queue import Queue

from PySide6.QtCore import QThread, Signal

from benji.llm.providers import LocalSummaryProvider, SummaryProvider
from benji.llm.summarizer import save_summary

log = logging.getLogger(__name__)

_STOP_SENTINEL = object()


class SummaryWorker(QThread):
    started = Signal(str)               # summary_id
    chunk = Signal(str, str)            # summary_id, token chunk
    finished = Signal(str, object)      # summary_id, Path
    failed = Signal(str, str)           # summary_id, error message

    def __init__(self, provider: SummaryProvider | None = None, parent=None):
        super().__init__(parent)
        self._provider: SummaryProvider = provider or LocalSummaryProvider()
        self._queue: Queue = Queue()
        self.setObjectName("SummaryWorker")

    def request(self, entries: list[dict], summary_id: str) -> None:
        """Thread-safe : enqueue une demande de résumé.

        entries: liste de dicts (au format TranscriptionHistory) avec au moins
        une clé 'text'. summarize() concatène et alimente le LLM.
        """
        self._queue.put((summary_id, entries))

    def shutdown(self) -> None:
        self._queue.put(_STOP_SENTINEL)
        self.wait(5000)

    def run(self) -> None:
        log.info("SummaryWorker started")
        while True:
            item = self._queue.get()
            if item is _STOP_SENTINEL:
                break
            sid, entries = item
            self.started.emit(sid)
            try:
                full = self._provider.summarize(
                    entries,
                    on_token=lambda c, _sid=sid: self.chunk.emit(_sid, c),
                )
                if not full:
                    self.failed.emit(sid, "Le résumé est vide (aucune transcription).")
                    continue
                path = save_summary(full, entries)
                self.finished.emit(sid, path)
            except Exception as e:
                log.exception("Summary failed for %s", sid)
                self.failed.emit(sid, str(e))
        log.info("SummaryWorker stopped")
