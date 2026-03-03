import re

# ── Spanish lyric contractions ─────────────────────────────────────────────────
# Pattern: dropped final consonant replaced by apostrophe (e.g. dao → da'o)
# or clitic contractions (pa' → para, etc.)

SPANISH_CONTRACTIONS = {
    # pa' family
    r"\bpa'l\b":          "para el",
    r"\bpa'la\b":         "para la",
    r"\bpa'cá\b":         "para acá",
    r"\bpa'llá\b":        "para allá",
    r"\bpa'lante\b":      "para adelante",
    r"\bpa'trás\b":       "para atrás",
    r"\bpa'trá\b":        "para atrás",
    r"\bpa'que\b":        "para que",
    r"\bpa'ónde\b":       "para dónde",
    r"\bpa'quererte\b":   "para quererte",
    r"\bpa'\b":           "para",

    # vo'a / vamo'a  (voy a / vamos a)
    r"\bvo'a\b":          "voy a",
    r"\bvamo'a\b":        "vamos a",

    # dropped -d- in past participles / adjectives  (*a'o → ado, *a'a → ada)
    r"\bto'a\b":          "toda",
    r"\bto'as\b":         "todas",
    r"\bto's\b":          "todos",
    r"\bla'o\b":          "lado",
    r"\bcuida'o\b":       "cuidado",
    r"\btumba'o\b":       "tumbado",
    r"\bqueda'o\b":       "quedado",
    r"\bpasa'o\b":        "pasado",
    r"\bda'o\b":          "dado",
    r"\bacelera'o\b":     "acelerado",
    r"\bcalla'o\b":       "callado",
    r"\benamora'o\b":     "enamorado",
    r"\bflota'o\b":       "flotado",
    r"\barrebata'o\b":    "arrebatado",
    r"\bpega'o\b":        "pegado",
    r"\bcansa'o\b":       "cansado",
    r"\bguilla'o\b":      "guillado",
    r"\bpara'o\b":        "parado",
    r"\bdobla'o\b":       "doblado",
    r"\bmata'o\b":        "matado",
    r"\bdemasia'o\b":     "demasiado",
    r"\bequivoca'o\b":    "equivocado",
    r"\becha'o\b":        "echado",

    # other
    r"\boí'te\b":         "óyete",
    r"\bllega'n\b":       "llegan",
    r"\bmi'mo\b":         "mismo",
    r"\bna'\b":           "nada",
    r"\bma'\b":           "más",
    r"\bta'\b":           "está",
    r"\bta's\b":          "están",
    r"\bd'\b":            "de",
}

# ── English contractions ───────────────────────────────────────────────────────
ENGLISH_CONTRACTIONS = {
    r"\bdon't\b":         "do not",
    r"\bc'mon\b":         "come on",
    r"\blet's\b":         "let us",
    r"\bi'm\b":           "i am",
    r"\bit's\b":          "it is",
    r"\bthat's\b":        "that is",
    r"\byou're\b":        "you are",
    r"\bi'll\b":          "i will",
    r"\bain't\b":         "is not",
    r"\bshe's\b":         "she is",
    r"\bcan't\b":         "cannot",
    r"\bhe's\b":          "he is",
    r"\bthey're\b":       "they are",
    r"\bwe're\b":         "we are",
    r"\bwon't\b":         "will not",
    r"\bdidn't\b":        "did not",
    r"\bwouldn't\b":      "would not",
    r"\bcouldn't\b":      "could not",
    r"\bshouldn't\b":     "should not",
    r"\bi've\b":          "i have",
    r"\bthey've\b":       "they have",
    r"\bwe've\b":         "we have",
    r"\byou've\b":        "you have",
    r"\bi'd\b":           "i would",
    r"\byou'd\b":         "you would",
    r"\bhe'd\b":          "he would",
    r"\bshe'd\b":         "she would",
    r"\bwe'd\b":          "we would",
    r"\bthey'd\b":        "they would",
    r"\bwhat's\b":        "what is",
    r"\bwhere's\b":       "where is",
    r"\bthere's\b":       "there is",
    r"\bwho's\b":         "who is",
    r"\bhow's\b":         "how is",
    r"\byou'll\b":        "you will",
    r"\bhe'll\b":         "he will",
    r"\bshe'll\b":        "she will",
    r"\bwe'll\b":         "we will",
    r"\bthey'll\b":       "they will",
    r"\bisn't\b":         "is not",
    r"\baren't\b":        "are not",
    r"\bwasn't\b":        "was not",
    r"\bweren't\b":       "were not",
    r"\bhasn't\b":        "has not",
    r"\bhaven't\b":       "have not",
    r"\bhadn't\b":        "had not",
}

CONTRACTIONS = {**SPANISH_CONTRACTIONS, **ENGLISH_CONTRACTIONS}


def normalize_contractions(text: str, spanish: bool = True, english: bool = True) -> str:
    """Replace informal contractions in lyrics with their full forms."""
    rules = {}
    if spanish:
        rules.update(SPANISH_CONTRACTIONS)
    if english:
        rules.update(ENGLISH_CONTRACTIONS)
    for pattern, replacement in rules.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text