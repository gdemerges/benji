"""Rien n'est conservé tant que l'utilisateur ne l'a pas dit.

Benji écoute et transcrit dès le lancement — c'est ce qui fait qu'on n'a jamais
« oublié de lancer l'enregistrement ». Mais **écouter et garder ne sont pas le
même geste** : sans cette distinction, une conversation de couloir, un appel
perso ou un déjeuner près du Mac finissaient dans `history.jsonl`, sur un
produit dont l'argument est précisément la maîtrise de ses propres réunions.

Le direct reste donc toujours affiché ; l'écriture disque, elle, attend un
accord. Ce qui a déjà été dit **n'est pas perdu** dans l'intervalle : les
entrées sont gardées en mémoire et versées à l'historique au moment de l'accord.
C'est tout l'intérêt par rapport à un bouton « démarrer » — on peut décider de
garder la réunion trois minutes après qu'elle a commencé.

Module pur : ni Qt, ni disque. Le magasin réel est injecté.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime

log = logging.getLogger(__name__)

# Ce qu'on garde en mémoire en attendant l'accord. Au-delà, les plus anciennes
# sont oubliées : un Benji laissé ouvert toute la journée sans accord ne doit pas
# accumuler indéfiniment — et les entrées perdues n'ont, par construction, jamais
# été promises à personne.
_MAX_PENDING = 2000


class RecordingConsent:
    """Portillon d'écriture entre le transcripteur et l'historique.

    Expose la même signature `add(text, speaker=None)` que
    `TranscriptionHistory` : le transcripteur ne sait pas qu'il est filtré.
    """

    def __init__(self, history, armed: bool = False):
        self._history = history
        self._armed = armed
        # (texte, locuteur, instant où c'est dit). L'instant est pris **ici** et
        # non au versement : sans lui, tout ce qui précédait l'accord partageait
        # l'heure du clic — la ligne de temps s'effondrait en un point, les
        # sous-titres SRT duraient zéro seconde et les marques s'accrochaient
        # de travers.
        self._pending: list[tuple[str, str | None, datetime]] = []
        # `add` vient du thread STT (et du correcteur), `arm`/`reset` du thread
        # Qt. Sans verrou, une phrase ajoutée pendant le versement tombait dans
        # la liste qu'on venait de vider, portillon ouvert : jamais écrite.
        # Le versement se fait verrou tenu, pour qu'une phrase dite pendant ce
        # temps s'écrive *après* celles qui l'ont précédée.
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add(self, text: str, speaker: str | None = None) -> None:
        with self._lock:
            if self._armed:
                self._history.add(text, speaker=speaker)
                return
            self._pending.append((text, speaker, datetime.now()))
            if len(self._pending) > _MAX_PENDING:
                del self._pending[: len(self._pending) - _MAX_PENDING]

    def arm(self) -> int:
        """Accorde la conservation et verse l'attente. Retourne le nombre versé.

        Idempotent : réarmer une conservation déjà accordée ne réécrit rien.
        """
        with self._lock:
            if self._armed:
                return 0
            # Armer **avant** de verser : si une écriture échoue, on ne repart
            # pas avec un portillon fermé et un historique à moitié rempli.
            self._armed = True
            pending, self._pending = self._pending, []
            for text, speaker, said_at in pending:
                self._history.add(text, speaker=speaker, timestamp=said_at)
        # Le compte, jamais le contenu : ce log part dans les rapports de bug.
        log.info("Conservation accordée — %d entrée(s) versée(s)", len(pending))
        return len(pending)

    def reset(self, armed: bool = False) -> None:
        """Nouvelle réunion : l'accord ne se reporte pas d'une réunion à l'autre.

        Ce qui restait en attente est abandonné — il appartenait à la réunion
        qu'on vient de quitter, et personne n'a demandé à le garder.
        """
        with self._lock:
            self._armed = armed
            self._pending = []
