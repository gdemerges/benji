"""Reconnaître quelqu'un qui se présente : « Bonjour, je m'appelle Guillaume ».

Pur, sans Qt. En début de réunion, les gens se présentent : c'est le moment où
« A » devient une personne, et le seul où l'app peut l'apprendre sans qu'on
clique. On ne cherche que la **première personne** — « il s'appelle Marc »
nomme quelqu'un d'autre que celui qui parle, et la diarisation n'a aucun moyen
de savoir qui.

Prudence plutôt que rappel : un faux nom collé sur un locuteur pendant toute
une réunion est pire qu'un « A » qu'on renomme d'un clic. D'où trois garde-fous :
le nom doit porter une majuscule (Parakeet capitalise les noms propres ; « moi
c'est bon » ne nomme personne), il ne peut pas être un mot courant capitalisé en
début de phrase, et « je suis » n'est pas un déclencheur (« je suis Désolé »,
« je suis Là » après une mauvaise casse).
"""

from __future__ import annotations

import re

# Un prénom : majuscule, lettres (accents compris), éventuellement composé
# (Jean-Pierre, Marie-Ève). Jusqu'à deux mots pour « Guillaume Demergès ».
_WORD = r"[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][a-zà-ÿ]+(?:-[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][a-zà-ÿ]+)?"
_NAME = rf"(?P<name>{_WORD}(?:\s+{_WORD})?)"

_TRIGGERS = (
    r"je\s+m['’]\s*appelle",
    r"je\s+me\s+(?:nomme|prénomme|prenomme)",
    r"mon\s+(?:pré)?nom\s+(?:c['’]\s*)?est",
    r"moi\s+c['’]\s*est",
    r"my\s+name\s+is",
    r"(?<!\w)i\s+am\s+called",
)
_PATTERN = re.compile(
    rf"(?:^|[\s,.;:!?«\"])(?:{'|'.join(_TRIGGERS)})\s+{_NAME}",
    re.IGNORECASE,
)

# Mots qu'une casse de début de phrase peut faire passer pour un prénom.
_NOT_NAMES = {
    "bien", "bon", "pas", "rien", "tout", "juste", "vraiment", "comme",
    "donc", "alors", "aussi", "enfin", "bref", "voilà", "euh", "ben",
    "le", "la", "les", "un", "une", "des", "du", "de", "moi", "lui",
    "the", "not", "just", "also",
}


def detect_self_name(text: str) -> str | None:
    """Le nom que se donne celui qui parle, ou None.

    >>> detect_self_name("Bonjour, je m'appelle Guillaume et je suis dev.")
    'Guillaume'
    """
    if not text:
        return None
    match = _PATTERN.search(text)
    if match is None:
        return None
    # L'IGNORECASE du déclencheur vaut aussi pour le nom : on revérifie la
    # majuscule à la main, mot par mot, et on s'arrête au premier qui n'en a pas.
    words: list[str] = []
    for word in match.group("name").split():
        if not word[0].isupper() or word.lower() in _NOT_NAMES:
            break
        words.append(word)
    return " ".join(words) or None
