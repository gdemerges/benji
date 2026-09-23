"""Nettoyage appliqué au texte sorti du moteur, après la passe finale."""

import re

# Rebuts classiques des modèles entraînés sur des corpus de sous-titres, émis
# sur du silence ou un signal faible. C'était le mode d'échec de Whisper ; le
# filtre reste en place pour Parakeet — il ne coûte qu'une recherche de
# sous-chaîne, et le jour où un moteur recrache ce boilerplate, il est déjà là.
# Comparé en minuscules, points de fin retirés.
#
# Deux familles, parce qu'elles ne se jugent pas pareil :
#
# - les **crédits** de sous-titrage, que personne ne prononce en réunion : leur
#   seule présence suffit, et ils traînent un nom derrière eux (« … par la
#   société Radio-Canada ») ;
HALLUCINATION_CREDITS = (
    "sous-titres réalisés par",
    "sous-titres fait par",
    "sous-titrage st'",
    "sous-titrage société radio",
    "sous-titres faits par",
    "❤️ par sous-titres",
    "amara.org",
)
# - les **formules** de fin de vidéo, qui sont aussi de vraies phrases de
#   réunion. « Merci à tous d'être venus, on commence » est une ouverture, pas
#   un rebut : jeter tout le tour dès que la formule apparaît faisait disparaître
#   précisément les phrases d'ouverture et de clôture. Des formules ne sont un
#   rebut que si elles font **tout** le texte, à `_FORMULA_SLACK_WORDS` mots de
#   liaison près (« merci d'avoir regardé **et** à la prochaine »).
HALLUCINATION_FORMULAS = (
    "merci d'avoir regardé",
    "merci de votre attention",
    "merci à tous",
    "abonnez-vous",
    "n'oubliez pas de vous abonner",
    "à la prochaine",
    "thanks for watching",
    "thank you for watching",
    "subscribe to",
    "please subscribe",
    "like and subscribe",
)
_FORMULA_SLACK_WORDS = 1
# Un moteur qui déraille boucle longuement sur le même mot ; un humain dit « non
# non non non » et continue sa phrase. Le seuil laisse passer le second.
_MAX_HUMAN_REPEATS = 5


def is_hallucination(text: str) -> bool:
    """Vrai si *text* ressemble à un rebut connu ou à une répétition dégénérée
    (même mot plus de `_MAX_HUMAN_REPEATS` fois de suite)."""
    if not text:
        return True
    normalized = text.lower().strip().rstrip(".!?")
    if any(pattern in normalized for pattern in HALLUCINATION_CREDITS):
        return True
    rest = normalized
    for pattern in HALLUCINATION_FORMULAS:
        rest = rest.replace(pattern, " ")
    if rest != normalized and len(re.findall(r"\w+", rest)) <= _FORMULA_SLACK_WORDS:
        return True
    # Sur du bruit, un modèle qui déraille boucle sur le même token.
    if re.search(rf"\b(\w{{2,}})\b(?:\W+\1\b){{{_MAX_HUMAN_REPEATS},}}",
                 normalized, flags=re.IGNORECASE):
        return True
    return False


def postprocess_text(text: str, language: str = None) -> str:
    """
    Enhance transcription with better punctuation and capitalization.

    Le moteur ponctue déjà ; on améliore :
    - Capitalization after periods
    - Removal of hesitations (uh, um, etc.)
    - Proper spacing around punctuation
    - Number formatting
    """
    if not text or not text.strip():
        return text

    # Remove common hesitations/fillers.
    # NB: pas de « eh/ah/oh » ici — mots légitimes en français (« eh bien »,
    # « ah bon »).
    hesitations = [
        r'\b(euh|euuh|heu|heuu)\b',  # French
        r'\b(uh|uhh|um|umm|hmm|huh)\b',  # English
    ]
    for pattern in hesitations:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
    # Add space after punctuation — sauf entre deux chiffres, pour ne pas
    # casser les nombres (« 2,5 », « 3.14 »).
    text = re.sub(r'((?<!\d)[,.!?;:]|[,.!?;:](?!\d))\s*', r'\1 ', text)
    # … mais pas à l'intérieur d'une ponctuation composée : « ?! », « ... ».
    text = re.sub(r'(?<=[.!?]) (?=[.!?])', '', text)

    # Fix French apostrophes (e.g., "qu ' on" -> "qu'on"). `[^\W\d_]` = une
    # lettre, accents compris : `[a-z]` laissait « l ' été » en l'état.
    text = re.sub(r"([^\W\d_])\s*(['’])\s*([^\W\d_])", r"\1\2\3", text)

    # Fix hyphens (e.g., "est - ce" -> "est-ce", "peut - être" -> "peut-être")
    text = re.sub(r"([^\W\d_])\s*-\s*([^\W\d_])", r"\1-\2", text)

    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces

    # Ponctuation orpheline en tête après suppression d'une hésitation
    # (« Euh, oui » → « , oui ») : on la retire avant de capitaliser.
    text = re.sub(r'^[\s,;:]+', '', text)

    # Capitalize first letter
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]

    # Capitalize after sentence-ending punctuation (Unicode-aware, preserve space)
    def capitalize_after_period(match):
        return match.group(1) + match.group(2) + match.group(3).upper()

    text = re.sub(r'([.!?])(\s+)(\w)', capitalize_after_period, text, flags=re.UNICODE)

    # Language-specific improvements
    if language == 'en':
        # Capitalize "I" pronoun
        text = re.sub(r'\bi\b', 'I', text)
        # Common contractions
        text = re.sub(r"\bim\b", "I'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdont\b", "don't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcant\b", "can't", text, flags=re.IGNORECASE)

    if language == 'fr':
        text = french_typography(text)

    # Remove trailing spaces
    text = text.strip()

    return text


# Espace fine insécable (avant ; ! ?) et espace insécable (avant :, dans « »).
NNBSP = "\u202f"
NBSP = "\u00a0"


def french_typography(text: str) -> str:
    """Espacements de la typographie française autour de la ponctuation haute.

    Le nettoyage générique colle la ponctuation au mot précédent, à l'anglaise :
    « Vraiment? » au lieu de « Vraiment ? ». Les espaces posées ici sont
    **insécables**, pour qu'un retour à la ligne ne laisse jamais un « ? » seul
    en tête de ligne — dans l'overlay, où le texte est justement coupé à la
    largeur de l'écran. Pure et idempotente : elle se rejoue sans doubler les
    espaces.

    Deux exceptions : une ponctuation qui en suit une autre (« ?! ») reste
    collée, et `:` entre deux chiffres (« 14:30 ») n'est pas une ponctuation.
    """
    if not text:
        return text
    text = re.sub(r"(?<=[^\s?!;:])[ \u00a0\u202f]?([;!?])", NNBSP + r"\1", text)
    text = re.sub(r"(?<=[^\s?!;:\d])[ \u00a0\u202f]?:", NBSP + ":", text)
    text = re.sub(r"(?<=\d)[ \u00a0\u202f]?:(?!\d)", NBSP + ":", text)
    text = re.sub(r"«[ \u00a0\u202f]*", "«" + NBSP, text)
    text = re.sub(r"[ \u00a0\u202f]*»", NBSP + "»", text)
    return text


def format_for_display(text: str) -> str:
    """
    Format text for display (lighter processing, preserves natural flow).
    """
    if not text or not text.strip():
        return text

    # Just clean up spacing
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def join_words(texts) -> str:
    """Recolle des mots en une phrase affichable.

    Le moteur rend la ponctuation comme un mot à part (`"."`, `","`) : la coller
    au mot précédent évite le « bonjour , monde » qu'un simple `" ".join`
    produisait dans l'overlay. Pure — c'est du rendu, pas du post-traitement de
    la passe finale.
    """
    out = ""
    for text in texts:
        if not text:
            continue
        sep = (
            ""
            if not out or out.endswith(" ") or text.startswith((".", ",", "!", "?", ";", ":", "'", "’"))
            else " "
        )
        out = out + sep + text
    return out.strip()
