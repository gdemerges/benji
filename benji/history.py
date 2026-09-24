"""Historique des transcriptions : un JSONL append-only, découpé en réunions.

Chaque entrée porte l'identifiant de la réunion dans laquelle elle a été dite
(cf. `benji/meetings.py`) ; les entrées écrites avant l'existence des réunions
n'en ont pas et sont regroupées sous `meetings.LEGACY_ID`.

Deux contraintes ont façonné ce module :

- **Confidentialité** — le fichier contient le contenu des réunions. Il est créé
  en 0600 dès l'`os.open` (un write-puis-chmod laisserait une fenêtre où il est
  lisible par tous) et vit dans les données utilisateur, pas dans `~/.cache`.
- **Chemin chaud** — `add()` est appelé pour *chaque* segment final, depuis le
  thread STT ou le thread correcteur. Il ne doit donc rien faire de proportionnel
  à la taille du fichier : la troncature est amortie via un compteur de lignes
  tenu en mémoire, et non une relecture intégrale à chaque ajout.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from benji import meetings
from benji.paths import user_path

# Au-delà du plafond, on ne tronque qu'une fois ce surplus accumulé : la
# réécriture du fichier coûte O(n), l'amortir la rend négligeable par segment.
_TRIM_SLACK = 500


class TranscriptionHistory:
    def __init__(self, max_entries: int = 20000, path: Path | None = None):
        self.max_entries = max_entries
        self.history_file = path or user_path("history.jsonl")
        self._lock = threading.Lock()
        # None = jamais compté. Le comptage initial est fait au premier ajout,
        # une seule fois pour la durée du process.
        self._line_count: int | None = None

    # --- écriture ---

    def add(self, text: str, speaker: str | None = None, meeting_id: str | None = None,
            timestamp: datetime | None = None):
        """Ajoute une transcription (optionnellement taguée d'un locuteur).

        `timestamp` : quand la phrase a été dite, si ce n'est pas maintenant —
        le versement de ce qui précédait l'accord de conservation
        (`benji/recording.py`). Une réunion ouverte par cette entrée commence
        alors à cet instant, pas à celui du clic.
        """
        said_at = timestamp or datetime.now()
        entry = {
            "timestamp": said_at.isoformat(),
            "text": text,
            "meeting": meeting_id or meetings.current_meeting(started_at=said_at).id,
        }
        if speaker:
            entry["speaker"] = speaker
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        with self._lock:
            fd = os.open(self.history_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)
            if self._line_count is None:
                self._line_count = self._count_lines()
            else:
                self._line_count += 1
            if self._line_count > self.max_entries + _TRIM_SLACK:
                self._trim()

    def _count_lines(self) -> int:
        try:
            with open(self.history_file, encoding="utf-8") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def _trim(self) -> None:
        """Ne garde que les `max_entries` dernières entrées. Lock déjà tenu."""
        try:
            with open(self.history_file, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        kept = lines[-self.max_entries:]
        tmp = self.history_file.with_suffix(".jsonl.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp, self.history_file)
        self._line_count = len(kept)

    # --- lecture ---

    def _iter_entries(self):
        try:
            with open(self.history_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict):
                        yield entry
        except OSError:
            return

    def get_recent(self, n: int = 50) -> list[dict]:
        """Les n transcriptions les plus récentes, la plus récente en premier."""
        from collections import deque

        return list(reversed(deque(self._iter_entries(), maxlen=n)))

    def get_since(self, since: datetime) -> list[dict]:
        """Toutes les transcriptions enregistrées depuis un instant donné."""
        entries = []
        for entry in self._iter_entries():
            try:
                if datetime.fromisoformat(entry["timestamp"]) >= since:
                    entries.append(entry)
            except (KeyError, TypeError, ValueError):
                continue
        return entries

    def get_for_meeting(self, meeting_id: str) -> list[dict]:
        """Transcriptions d'une réunion, dans l'ordre chronologique d'écriture.

        `meetings.LEGACY_ID` renvoie les entrées antérieures aux réunions
        (celles sans champ `meeting`).
        """
        if meeting_id == meetings.LEGACY_ID:
            return [e for e in self._iter_entries() if not e.get("meeting")]
        return [e for e in self._iter_entries() if e.get("meeting") == meeting_id]

    def group_by_meeting(self) -> dict[str, list[dict]]:
        """Toutes les entrées, indexées par réunion, **en une seule lecture**.

        La fenêtre Réunions a besoin du contenu de chaque réunion à la fois (un
        compteur par ligne, et désormais une recherche qui les traverse toutes).
        Appeler `get_for_meeting()` en boucle relisait le fichier entier une fois
        par réunion : à cinquante réunions, cinquante lectures complètes à chaque
        rafraîchissement de la liste.

        Les entrées antérieures à la notion de réunion sont regroupées sous
        `meetings.LEGACY_ID`.
        """
        grouped: dict[str, list[dict]] = {}
        for entry in self._iter_entries():
            grouped.setdefault(entry.get("meeting") or meetings.LEGACY_ID, []).append(entry)
        return grouped

    def has_legacy_entries(self) -> bool:
        return any(not e.get("meeting") for e in self._iter_entries())

    # --- suppression ---

    def clear(self, meeting_id: str | None = None):
        """Efface tout l'historique, ou seulement celui d'une réunion."""
        with self._lock:
            if meeting_id is None:
                if self.history_file.exists():
                    self.history_file.unlink()
                self._line_count = 0
                return
            if meeting_id == meetings.LEGACY_ID:
                kept = [e for e in self._iter_entries() if e.get("meeting")]
            else:
                kept = [e for e in self._iter_entries() if e.get("meeting") != meeting_id]
            tmp = self.history_file.with_suffix(".jsonl.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for entry in kept:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp, self.history_file)
            self._line_count = len(kept)
