"""Découpe d'un segment finalisé en tours de parole (pur, sans réseau).

Deepgram et Grok étiquettent **chaque mot** avec un locuteur, mais un segment
final peut couvrir deux voix qui s'enchaînent sans pause franche. Étiqueter le
segment entier avec le locuteur de son premier mot fondait les deux répliques en
une seule, d'une seule couleur — le défaut que `benji/stt/diarization.py` règle
côté local. On regroupe donc les mots consécutifs d'un même locuteur, et le
segment rend un `final_text` par tour.

Garde-fou, parce que le clustering du provider hésite : un tour plus court que
`min_turn_words` est refondu dans son voisin plutôt que d'ouvrir une réplique
d'un mot (même règle que `diarization_min_turn_words` en local).
"""

from __future__ import annotations

MIN_TURN_WORDS = 2


def speaker_label(n) -> str | None:
    """Index de locuteur du provider → label du contrat (0 → "A", 1 → "B", …)."""
    if n is None:
        return None
    n = int(n)
    return chr(ord("A") + n) if 0 <= n < 26 else f"S{n}"


def split_turns(
    transcript: str,
    words: list[dict],
    text_keys: tuple[str, ...],
    min_turn_words: int = MIN_TURN_WORDS,
) -> list[tuple[str, str | None]]:
    """Renvoie `[(texte, locuteur)]`, un élément par tour de parole.

    `text_keys` : champs à essayer, dans l'ordre, pour le texte d'un mot
    (Deepgram : `punctuated_word` puis `word` ; Grok : `text`).

    Un seul locuteur (ou aucune étiquette) → le transcript d'origine, intact :
    le chemin d'avant, sans reconstruction du texte depuis les mots.
    """
    tokens: list[tuple[str, str | None]] = []
    for w in words:
        text = next((w[k] for k in text_keys if w.get(k)), None)
        if text:
            tokens.append((str(text), speaker_label(w.get("speaker"))))

    # Un mot sans étiquette prolonge le tour en cours.
    runs: list[list] = []  # [[speaker, [mots]]]
    for text, spk in tokens:
        if runs and (spk is None or spk == runs[-1][0]):
            runs[-1][1].append(text)
        elif not runs and spk is None:
            runs.append([None, [text]])
        elif runs and runs[-1][0] is None:
            runs[-1][0] = spk
            runs[-1][1].append(text)
        else:
            runs.append([spk, [text]])

    # Tours trop courts refondus dans le voisin (le précédent, sinon le suivant).
    merged = True
    while merged and len(runs) > 1:
        merged = False
        for i, (_, ws) in enumerate(runs):
            if len(ws) < min_turn_words:
                if i > 0:
                    runs[i - 1][1].extend(ws)
                else:
                    runs[1][1][:0] = ws
                del runs[i]
                merged = True
                break
        # Deux voisins du même locuteur après refonte → un seul tour.
        i = 1
        while i < len(runs):
            if runs[i][0] == runs[i - 1][0]:
                runs[i - 1][1].extend(runs.pop(i)[1])
            else:
                i += 1

    if len(runs) <= 1:
        spk = runs[0][0] if runs else None
        return [(transcript, spk)]
    return [(" ".join(ws), spk) for spk, ws in runs]
