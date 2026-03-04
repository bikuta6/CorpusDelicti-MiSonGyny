import re
import unicodedata
from sentence_transformers import SentenceTransformer, util
from pysentimiento.preprocessing import preprocess_tweet
import torch
import pandas as pd

# Structural section labels to remove from parentheses (Spanish + English)
_STRUCTURAL_LABELS_RE = re.compile(
    r"\(\s*(coro|verso|estrofa|puente|bis|intro|outro|chorus|hook|bridge|verse|refr[aá]n|interludio)\s*\d*\s*\)",
    flags=re.IGNORECASE,
)


def _normalize_lyric_text(text: str) -> str:
    """
    Light normalization before embedding:
    - NFC unicode (preserves ñ, á, é, í, ó, ú)
    - Remove zero-width / control characters
    - Normalize fancy quotes and apostrophes
    - Collapse repeated punctuation: !!! → !, ??? → ?
    - Normalize jajaja variants
    """
    # NFC: ensures accented chars are one codepoint, not two
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width and ASCII control characters (except newlines/tabs)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Normalize fancy quotes/apostrophes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''`]", "'", text)
    # Collapse repeated punctuation
    text = re.sub(r"([!?]){2,}", r"\1", text)
    # Normalize jajaja variants (jajajajaja → jajaja)
    text = re.sub(r"(ja){3,}", "jajaja", text, flags=re.IGNORECASE)
    return text


def remove_redundant_lyrics(
    model: SentenceTransformer,
    text: str,
    threshold: float = 0.82,
    line_threshold: float = 0.95,
) -> str:
    """
    Deduplicate lyrics hierarchically:
    1. Stanza-level (split by double newlines)
    2. Line-level within each stanza
    """
    if not text.strip():
        return ""

    # --- Preprocessing: clean and split stanzas ---
    stanzas = text.split("\n\n")
    clean_stanzas = []
    for stanza in stanzas:
        lines = [re.sub(r"\[.*?\]", "", line).strip() for line in stanza.split("\n")]
        lines = [_STRUCTURAL_LABELS_RE.sub("", line).strip() for line in lines]
        lines = [re.sub(r'[()"]', "", line).strip() for line in lines]
        lines = [_normalize_lyric_text(line) for line in lines if len(line) > 2]
        if lines:
            clean_stanzas.append(lines)

    if not clean_stanzas:
        return ""

    # --- Batch encode all stanzas at once ---
    all_stanza_texts = [
        f"passage: {' '.join([preprocess_tweet(l) for l in s])}" for s in clean_stanzas
    ]
    all_stanza_embs = model.encode(all_stanza_texts, convert_to_tensor=True)

    kept_stanzas = [clean_stanzas[0]]
    kept_stanza_embs = [all_stanza_embs[0]]

    for i in range(1, len(clean_stanzas)):
        current_embedding = all_stanza_embs[i]
        kept_embs_tensor = torch.stack(kept_stanza_embs)
        max_sim = util.cos_sim(current_embedding, kept_embs_tensor).max().item()
        if max_sim < threshold:
            kept_stanzas.append(clean_stanzas[i])
            kept_stanza_embs.append(current_embedding)

    # --- Line-level deduplication within each kept stanza ---
    final_stanzas = []
    for stanza in kept_stanzas:
        if len(stanza) == 1:
            final_stanzas.append(stanza)
            continue

        preprocessed = [f"passage: {preprocess_tweet(line)}" for line in stanza]
        all_embs = model.encode(preprocessed, convert_to_tensor=True)  # batch encode

        kept_lines = [stanza[0]]
        kept_embs = [all_embs[0]]
        for idx, line in enumerate(stanza[1:], start=1):
            current_emb = all_embs[idx]
            similarities = [util.cos_sim(current_emb, l_emb) for l_emb in kept_embs]
            if max([sim.item() for sim in similarities]) < line_threshold:
                kept_lines.append(line)
                kept_embs.append(current_emb)
        final_stanzas.append(kept_lines)

    # --- Return reconstructed lyrics with original text ---
    return "\n\n".join(["\n".join(stanza) for stanza in final_stanzas])


if __name__ == "__main__":
    # Test with Spanglish and similar meanings (nuance check)
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    sample_lyrics = pd.read_csv("../data/task1/train.csv")["lyrics"]
    # Find the longest lyrics to test the function on a complex case
    longest_idx = sample_lyrics.apply(lambda x: len(x.split())).idxmax()
    sample_lyrics = sample_lyrics.iloc[longest_idx : longest_idx + 1].tolist()
    print("--- Original Lyrics ---")
    print(sample_lyrics[0])
    print("Number of words:", len(sample_lyrics[0].split()))

    # 0.80 is usually the "sweet spot" for multilingual deduplication
    cleaned_text = [
        remove_redundant_lyrics(model, lyrics, threshold=0.90)
        for lyrics in sample_lyrics
    ]
    print("--- Cleaned Lyrics ---")
    print(cleaned_text[0])
    print("Number of words after cleaning:", len(cleaned_text[0].split()))
