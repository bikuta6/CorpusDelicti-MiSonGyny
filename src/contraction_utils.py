import re

# ── Spanish lyric contractions ─────────────────────────────────────────────────
# Pattern: dropped final consonant replaced by apostrophe (e.g. dao → da'o)
# or clitic contractions (pa' → para, etc.)

SPANISH_CONTRACTIONS = {
    # pa' family
    r"\bpa'l": "para el",
    r"\bpa'la": "para la",
    r"\bpa'cá": "para acá",
    r"\bpa'llá": "para allá",
    r"\bpa'lante": "para adelante",
    r"\bpa'trás": "para atrás",
    r"\bpa'trá": "para atrás",
    r"\bpa'que": "para que",
    r"\bpa'ónde": "para dónde",
    r"\bpa'quererte": "para quererte",
    r"\bpa'": "para",
    # vo'a / vamo'a / va' (voy a / vamos a / vas)
    r"\bvo'a": "voy a",
    r"\bvamo'a": "vamos a",
    r"\bva'": "vas",
    # dropped -d- in past participles / adjectives  (*a'o → ado, *a'a → ada)
    r"\bto'a": "toda",
    r"\bto'as": "todas",
    r"\bto's": "todos",
    r"\bla'o": "lado",
    r"\bcuida'o": "cuidado",
    r"\btumba'o": "tumbado",
    r"\bqueda'o": "quedado",
    r"\bpasa'o": "pasado",
    r"\bda'o": "dado",
    r"\bacelera'o": "acelerado",
    r"\bcalla'o": "callado",
    r"\benamora'o": "enamorado",
    r"\bflota'o": "flotado",
    r"\barrebata'o": "arrebatado",
    r"\bpega'o": "pegado",
    r"\bcansa'o": "cansado",
    r"\bguilla'o": "guillado",
    r"\bpara'o": "parado",
    r"\bdobla'o": "doblado",
    r"\bmata'o": "matado",
    r"\bdemasia'o": "demasiado",
    r"\bequivoca'o": "equivocado",
    r"\becha'o": "echado",
    # other
    r"\boí'te": "óyete",
    r"\bllega'n": "llegan",
    r"\bmi'mo": "mismo",
    r"\bna'": "nada",
    r"\bma'": "más",
    r"\bta'": "está",
    r"\bta's": "están",
    r"\bd'": "de",
    r"\be'": "es",
    r"\bto'": "todos",
}

# ── English contractions ───────────────────────────────────────────────────────
ENGLISH_CONTRACTIONS = {
    r"\bdon't": "do not",
    r"\bc'mon": "come on",
    r"\blet's": "let us",
    r"\bi'm": "i am",
    r"\bit's": "it is",
    r"\bthat's": "that is",
    r"\byou're": "you are",
    r"\bi'll": "i will",
    r"\bain't": "is not",
    r"\bshe's": "she is",
    r"\bcan't": "cannot",
    r"\bhe's": "he is",
    r"\bthey're": "they are",
    r"\bwe're": "we are",
    r"\bwon't": "will not",
    r"\bdidn't": "did not",
    r"\bwouldn't": "would not",
    r"\bcouldn't": "could not",
    r"\bshouldn't": "should not",
    r"\bi've": "i have",
    r"\bthey've": "they have",
    r"\bwe've": "we have",
    r"\byou've": "you have",
    r"\bi'd": "i would",
    r"\byou'd": "you would",
    r"\bhe'd": "he would",
    r"\bshe'd": "she would",
    r"\bwe'd": "we would",
    r"\bthey'd": "they would",
    r"\bwhat's": "what is",
    r"\bwhere's": "where is",
    r"\bthere's": "there is",
    r"\bwho's": "who is",
    r"\bhow's": "how is",
    r"\byou'll": "you will",
    r"\bhe'll": "he will",
    r"\bshe'll": "she will",
    r"\bwe'll": "we will",
    r"\bthey'll": "they will",
    r"\bisn't": "is not",
    r"\baren't": "are not",
    r"\bwasn't": "was not",
    r"\bweren't": "were not",
    r"\bhasn't": "has not",
    r"\bhaven't": "have not",
    r"\bhadn't": "had not",
}

CONTRACTIONS = {**SPANISH_CONTRACTIONS, **ENGLISH_CONTRACTIONS}


def normalize_contractions(
    text: str, spanish: bool = True, english: bool = True
) -> str:
    """Replace informal contractions in lyrics with their full forms."""
    rules = {}
    if spanish:
        rules.update(SPANISH_CONTRACTIONS)
    if english:
        rules.update(ENGLISH_CONTRACTIONS)
    for pattern, replacement in rules.items():

        def make_replacer(repl):
            def replacer(m):
                return repl[0].upper() + repl[1:] if m.group(0)[0].isupper() else repl

            return replacer

        text = re.sub(pattern, make_replacer(replacement), text, flags=re.IGNORECASE)
    return text
