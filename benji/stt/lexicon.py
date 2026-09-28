"""Glossaire utilisateur, appliqué **après** le décodage.

Le point faible d'une transcription de réunion n'est pas la grammaire : ce sont
les noms propres et le jargon maison. « Kubernetes » ressort en « couvre
Nettes », un nom de client en trois mots inventés. Le levier classique — un
`initial_prompt` glissé au moteur — n'existe plus depuis le passage à Parakeet,
qui n'accepte aucun prompt (cf. `benji/stt/CLAUDE.md`).

Ce module rétablit l'essentiel du bénéfice **à un autre étage** : on ne souffle
rien au modèle, on relit sa sortie. Un terme du glossaire remplace un mot (ou
une suite de mots) du texte quand les deux **sonnent pareil**. La comparaison
est phonétique, pas orthographique : c'est précisément parce que le moteur a
écrit autre chose que la comparaison littérale ne trouverait rien.

Deux garde-fous portent tout le reste :

- **Seule la passe finale est concernée.** Le texte partiel est jeté de toute
  façon, et remplacer des mots sous les yeux du lecteur est plus déroutant
  qu'une coquille passagère.
- **Rien ne sort de la machine.** Le glossaire contient des noms de clients et
  de projets : il vit dans les données utilisateur en 0600, n'est jamais loggué,
  et `benji/report.py` / `benji/monitoring.py` ne le joignent à rien.

Module pur (ni Qt, ni modèle) : le remplacement se teste sur des chaînes.
"""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

GLOSSARY_NAME = "glossary.txt"

# Un terme trop court ne peut pas être apparié sans risque : « API » et « happy »
# partagent une clé plausible, et une erreur sur un mot de deux lettres se voit
# plus qu'elle ne répare.
MIN_TERM_LENGTH = 4
# Nombre maximum de mots d'un terme apparié en un bloc (« Crédit Agricole »).
MAX_TERM_WORDS = 4

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


# --- clé phonétique ---------------------------------------------------------


def _strip_accents(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def phonetic_key(word: str) -> str:
    """Clé phonétique approximative, orientée français.

    Volontairement grossière : elle doit rapprocher « Kubernetes » de « couvre
    Nettes », pas transcrire l'API phonétique. Les règles sont appliquées dans
    l'ordre — « ch » avant « c », « ph » avant « h » — parce qu'une règle courte
    appliquée trop tôt mange le digramme que la suivante attendait.
    """
    word = (word or "").lower().replace("ç", "s")
    word = _strip_accents(word)
    word = "".join(c for c in word if c.isalpha())
    if not word:
        return ""

    # Digrammes d'abord : ils disparaissent avant que les règles à une lettre
    # ne puissent les casser.
    for src, dst in (
        ("ph", "f"), ("ch", "x"), ("sch", "x"), ("th", "t"),
        ("qu", "k"), ("gu", "g"), ("gn", "n"),
        ("eau", "o"), ("au", "o"), ("ou", "u"),
        ("ai", "e"), ("ei", "e"),
    ):
        word = word.replace(src, dst)

    out: list[str] = []
    for i, c in enumerate(word):
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if c == "c":
            out.append("s" if nxt in "eiy" else "k")
        elif c == "g":
            out.append("j" if nxt in "eiy" else "g")
        elif c == "q":
            out.append("k")
        elif c == "x":
            out.append("ks")
        elif c in "zs":
            out.append("s")
        elif c == "w":
            out.append("v")
        elif c == "y":
            out.append("i")
        elif c == "h":
            continue  # muet en français, et déjà consommé par ph/ch/th
        else:
            out.append(c)
    key = "".join(out)

    # Fins muettes, retirées jusqu'à stabilité : le français ne prononce ni le
    # 'e' final, ni la consonne qui le précède, ni la marque du pluriel. Sans
    # cette boucle « Kubernetes » et « kubernétesse » ne tomberaient pas sur la
    # même clé — or c'est exactement le genre d'écart que le glossaire existe
    # pour rattraper.
    while key and key[-1] in "etdspkxz":
        key = key[:-1]
    # Doublons : « Nettes » / « netes ».
    key = re.sub(r"(.)\1+", r"\1", key)
    return key


def _edit_distance(a: str, b: str, *, cap: int = 2) -> int:
    """Distance de Levenshtein, abandonnée dès qu'elle dépasse `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def keys_match(candidate: str, target: str) -> bool:
    """Vrai si deux clés phonétiques désignent vraisemblablement la même chose.

    Tolérance graduée : égalité stricte pour les clés courtes, une lettre
    d'écart à partir de six caractères, deux à partir de neuf. Plus la clé est
    longue, moins une collision fortuite est probable — et plus le moteur a eu
    d'occasions de se tromper sur un mot qu'il ne connaît pas.
    """
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    shortest = min(len(candidate), len(target))
    if shortest >= 9:
        return _edit_distance(candidate, target, cap=2) <= 2
    if shortest >= 6:
        return _edit_distance(candidate, target, cap=1) <= 1
    return False


# --- chargement du glossaire ------------------------------------------------


def parse_glossary(raw: str) -> list[str]:
    """Un terme par ligne ; `#` commente, les lignes vides sont ignorées."""
    terms: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        term = line.split("#", 1)[0].strip()
        if not term or len(term) < MIN_TERM_LENGTH:
            continue
        if len(_WORD_RE.findall(term)) > MAX_TERM_WORDS:
            continue
        if term.lower() in seen:
            continue
        seen.add(term.lower())
        terms.append(term)
    return terms


def load_terms(path=None) -> list[str]:
    """Termes du glossaire utilisateur. Fichier absent = liste vide.

    Le contenu n'est jamais loggué : seul son *nombre de termes* l'est, parce
    qu'un glossaire liste des clients et des projets.
    """
    # Même source de chemin que l'écriture : deux résolutions distinctes du
    # même fichier finissent par diverger.
    path = path or glossary_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    terms = parse_glossary(raw)
    if terms:
        log.info("Glossaire : %d terme(s) chargé(s)", len(terms))
    return terms


# --- application ------------------------------------------------------------


def _term_keys(term: str) -> list[str]:
    return [phonetic_key(w) for w in _WORD_RE.findall(term)]


def compile_terms(terms) -> list[tuple[str, list[str]]]:
    """Pré-calcule les clés, les termes longs d'abord.

    L'ordre compte : « Crédit Agricole » doit être essayé avant « Crédit »,
    sinon le premier mot est consommé seul et le second reste faux.
    """
    compiled = []
    for term in terms or []:
        keys = _term_keys(term)
        if keys and all(keys):
            compiled.append((term, keys))
    compiled.sort(key=lambda item: -len(item[1]))
    return compiled


def apply_lexicon(text: str, compiled) -> str:
    """Remplace dans *text* les suites de mots qui sonnent comme un terme.

    `compiled` vient de `compile_terms`. La comparaison porte sur les clés
    **concaténées** de la fenêtre, pas mot à mot : c'est ce qui permet de
    rattraper un terme que le moteur a découpé (« data dogue » → « Datadog »)
    ou recollé, alors qu'un appariement mot à mot exigerait qu'il se soit
    trompé en gardant le bon nombre de mots.

    Le texte est reconstruit à partir de ses fragments d'origine : la
    ponctuation, les espaces et tout ce qui n'a pas été apparié ressortent **à
    l'identique**.
    """
    if not text or not compiled:
        return text

    # Découpe alternée séparateurs / mots : le groupe capturant place les mots
    # aux indices impairs. On ne réécrit que ceux-là.
    tokens = re.split(r"([^\W\d_]+)", text, flags=re.UNICODE)
    word_positions = [i for i in range(1, len(tokens), 2)]
    keys = {i: phonetic_key(tokens[i]) for i in word_positions}

    out = list(tokens)
    consumed: set[int] = set()
    for term, term_keys in compiled:
        target = "".join(term_keys)
        # Le moteur peut avoir éclaté le terme en plus de mots qu'il n'en a :
        # on regarde jusqu'à deux fragments de plus.
        for span in range(1, len(term_keys) + 3):
            for start in range(len(word_positions) - span + 1):
                window = word_positions[start:start + span]
                if any(i in consumed for i in window):
                    continue
                # La fenêtre ne doit enjamber que des espaces : « Crédit,
                # agricole » est une énumération, pas le nom de la banque.
                if any(tokens[i].strip(" ") for i in range(window[0] + 1, window[-1])):
                    continue
                original = "".join(tokens[i] for i in range(window[0], window[-1] + 1))
                if original == term:
                    consumed.update(window)
                    continue
                if not keys_match("".join(keys[i] for i in window), target):
                    continue
                out[window[0]] = term
                for i in window[1:]:
                    out[i] = ""
                    out[i - 1] = ""  # l'espace qui précédait le fragment absorbé
                consumed.update(window)
    return "".join(out)


# --- édition (Préférences) --------------------------------------------------


def glossary_path():
    from benji.paths import user_path

    return user_path(GLOSSARY_NAME)


def read_raw(path=None) -> str:
    """Contenu brut du glossaire, tel que l'utilisateur l'a écrit."""
    path = path or glossary_path()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def save_glossary(raw: str, path=None) -> None:
    """Écrit le glossaire en 0600 **dès la création**.

    Il liste des noms de clients et de projets, au même titre que l'historique :
    un write-puis-chmod le laisserait lisible par tous entre les deux appels.

    Le `fchmod` n'est pas redondant avec le mode passé à `os.open` : celui-ci ne
    s'applique qu'à la **création**. Un glossaire déjà sur disque en 0644 —
    écrit par une version antérieure, ou par l'utilisateur à la main — resterait
    lisible par tous à chaque enregistrement. Il porte sur un descripteur déjà
    ouvert et avant toute écriture : pas de fenêtre.
    """
    import os

    path = path or glossary_path()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # absent sous Windows, où les modes POSIX ne s'appliquent pas
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(raw if raw.endswith("\n") else raw + "\n")


def add_term(term: str, path=None) -> bool:
    """Ajoute un terme au glossaire de l'utilisateur. Faux s'il y était déjà.

    Le glossaire est le seul levier qui reste sur les noms propres — le point
    faible n°1 d'une transcription de réunion — mais il fallait aller le saisir
    dans les Préférences, c'est-à-dire au moment où on ne pense pas à lui. Le
    nourrir depuis le transcript, quand on vient de *voir* la faute, est ce qui
    le rend vivant.

    Le terme n'est pas loggué : il nomme un client ou un projet.
    """
    term = (term or "").strip()
    if not term:
        return False
    existing = load_terms(path)
    if any(t.lower() == term.lower() for t in existing):
        return False
    save_glossary("\n".join([*existing, term]), path)
    log.info("Glossaire : 1 terme ajouté (%d au total)", len(existing) + 1)
    return True
