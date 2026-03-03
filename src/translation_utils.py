"""
Utilities to detect and translate English fragments in lyrics to Spanish.
Uses langdetect for language detection and deep-translator (GoogleTranslator)
for translation — no API key required.

Install dependencies:
    pip install langdetect deep-translator
"""

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

_translator = GoogleTranslator(source="en", target="es")


def _detect_lang(text: str) -> str:
    """Return ISO language code or '' on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return ""


def translate_english_parts(text: str, min_words: int = 3) -> str:
    """
    Split lyrics into lines, detect each line's language, and translate
    English lines to Spanish. Lines with fewer than `min_words` words are
    left unchanged (too short to detect reliably).

    Parameters
    ----------
    text : str
        Raw / contraction-normalised lyric text.
    min_words : int
        Minimum number of words in a line to attempt language detection.

    Returns
    -------
    str
        Lyric text with English lines replaced by their Spanish translation.
    """
    lines = text.splitlines()
    translated_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped.split()) < min_words:
            translated_lines.append(line)
            continue

        lang = _detect_lang(stripped)
        if lang == "en":
            try:
                translated = _translator.translate(stripped)
                # Preserve leading whitespace/indentation
                indent = line[: len(line) - len(line.lstrip())]
                translated_lines.append(indent + translated)
            except Exception:
                translated_lines.append(line)
        else:
            translated_lines.append(line)

    return "\n".join(translated_lines)