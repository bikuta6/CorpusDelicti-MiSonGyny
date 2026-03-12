import re

import tqdm

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
    r"\bpa'rriba": "para arriba",
    r"\bpa'ti": "para ti",
    r"\bpa'": "para",
    
    # vo'a / vamo'a / va' (voy a / vamos a / vas)
    r"\bvo'a": "voy a",
    r"\bvamo'a": "vamos a",
    r"\bva'": "vas",
    
    # dropped -d- in past participles / adjectives / nouns (*a'o → ado, *i'a → ida)
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
    r"\bactiva'o": "activado",
    r"\btapiza'o": "tapizado",
    r"\blogra'o": "logrado",
    r"\bapreta'o": "apretado",
    r"\bjuquea'o": "juqueado",
    r"\bvira'o": "virado",
    r"\bagranda'o": "agrandado",
    r"\btira'o": "tirado",
    r"\bdesacata'o": "desacatado",
    r"\bllega'o": "llegado",
    r"\bgana'o": "ganado",
    r"\bmoja'o": "mojado",
    r"\bhabla'o": "hablado",
    r"\bmama'o": "mamado",
    r"\bconecta'o": "conectado",
    r"\bburla'o": "burlado",
    r"\benrola'o": "enrolado",
    r"\bamarra'o": "amarrado",
    r"\benreda'os": "enredados",
    r"\bmari'o": "marido",
    r"\bmarí'o": "marido",
    r"\bde'o": "dedo",
    r"\bprendí'a": "prendida",
    r"\bprendi'a": "prendida",
    r"\bescondi'as": "escondidas",
    r"\bcomi'a": "comida",
    r"\bcomprometí'a": "comprometida",
    r"\bjodí'o": "jodido",
    r"\bmeti'o": "metido",
    
    # aspirated 's' / missing letters in pronouns, verbs and nouns
    r"\be'toy": "estoy",
    r"\be'ta": "está",
    r"\be'te": "este",
    r"\be'to": "esto",
    r"\bha'ta": "hasta",
    r"\bmi'mo": "mismo",
    r"\bnue'tro": "nuestro",
    r"\bco'quillita": "cosquillita",
    r"\bdi'que": "dizque", 
    
    # other
    r"\boí'te": "oíste", # Corregido de "óyete" (por contexto musical/urbano suele ser "oíste")
    r"\bo'ite": "oíste",
    r"\bllega'n": "llegan",
    r"\bna'": "nada",
    r"\bma'": "más",
    r"\bta'": "está",
    r"\bta's": "estás",
    r"\bd'": "de",
    r"\be'": "es",
    r"\bto'": "todo", # Ajustado para reflejar "to'" como "todo" (ej: "to' el día")
}
# ── English contractions ───────────────────────────────────────────────────────
ENGLISH_CONTRACTIONS = {
    # --- Verbos con "not" ---
    r"\bdon't": "do not",
    r"\bdoesn't": "does not", # Añadida
    r"\bdidn't": "did not",
    r"\bisn't": "is not",
    r"\baren't": "are not",
    r"\bwasn't": "was not",
    r"\bweren't": "were not",
    r"\bhasn't": "has not",
    r"\bhaven't": "have not",
    r"\bhadn't": "had not",
    r"\bwon't": "will not",
    r"\bwouldn't": "would not",
    r"\bcan't": "cannot",
    r"\bcouldn't": "could not",
    r"\bshouldn't": "should not",
    r"\bmustn't": "must not", # Añadida
    r"\bneedn't": "need not", # Añadida
    r"\bain't": "is not", # (O "are not" / "am not" dependiendo del contexto)

    # --- Pronombres + verbo "to be" (am/is/are) ---
    r"\bi'm": "i am",
    r"\byou're": "you are",
    r"\bhe's": "he is",
    r"\bshe's": "she is",
    r"\bit's": "it is",
    r"\bwe're": "we are",
    r"\bthey're": "they are",
    r"\bthat's": "that is",
    r"\bwhat's": "what is",
    r"\bwhere's": "where is",
    r"\bthere's": "there is",
    r"\bthere're": "there are", # Añadida
    r"\bwho's": "who is",
    r"\bhow's": "how is",

    # --- Pronombres + "will" ---
    r"\bi'll": "i will",
    r"\byou'll": "you will",
    r"\bhe'll": "he will",
    r"\bshe'll": "she will",
    r"\bit'll": "it will", # Añadida
    r"\bwe'll": "we will",
    r"\bthey'll": "they will",

    # --- Pronombres + "have" ---
    r"\bi've": "i have",
    r"\byou've": "you have",
    r"\bwe've": "we have",
    r"\bthey've": "they have",

    # --- Pronombres + "would" / "had" ---
    r"\bi'd": "i would",
    r"\byou'd": "you would",
    r"\bhe'd": "he would",
    r"\bshe'd": "she would",
    r"\bwe'd": "we would",
    r"\bthey'd": "they would",

    # --- Otras / Jerga (De la lista original y comunes) ---
    r"\bc'mon": "come on",
    r"\blet's": "let us",
    r"\by'all": "you all",    # Añadida
    r"\bshorty's": "shorty is", # De la lista (Cuidado: puede ser posesivo)
    r"\b90's": "90s",           # De la lista (Normalización de década)
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
    
    # IMPORTANTE: Ordenar por longitud de la clave (descendente)
    # Esto evita que "pa'" se coma a "pa'lante"
    sorted_patterns = sorted(rules.keys(), key=len, reverse=True)

    for pattern in sorted_patterns:
        replacement = rules[pattern]

        def make_replacer(repl):
            def replacer(m):
                original = m.group(0)
                # Si toda la palabra original está en mayúsculas: PA'L -> PARA EL
                if original.isupper():
                    return repl.upper()
                # Si solo la primera letra es mayúscula: Pa'l -> Para el
                if original[0].isupper():
                    return repl[0].upper() + repl[1:]
                # Por defecto minúsculas
                return repl
            return replacer

        text = re.sub(pattern, make_replacer(replacement), text, flags=re.IGNORECASE)
    
    return text
