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
    text = re.sub(r"([¡!¿?]){2,}", r"\1", text)
    # Normalize jajaja variants (jajajajaja → jajaja)
    text = re.sub(r"(ja){3,}", "jajaja", text, flags=re.IGNORECASE)
    return text


def remove_redundant_lyrics(
    model: SentenceTransformer,
    text: str,
    threshold: float = 0.82,
    line_threshold: float = 0.95,
    similarity_scope: str = "stanza_and_verse",
) -> str:
    """
    Deduplicate lyrics with configurable granularity:
    - "stanza": only stanza-level similarity check (split by double newlines)
    - "stanza_and_verse": stanza-level + verse/line-level within each stanza

    Then format output with semantic punctuation (commas and periods).
    """
    if similarity_scope not in {"stanza", "stanza_and_verse"}:
        raise ValueError(
            "similarity_scope must be either 'stanza' or 'stanza_and_verse'"
        )

    if not text.strip():
        return ""

    # --- Preprocessing: clean and split stanzas ---
    stanzas = text.split("\n\n")
    clean_stanzas =[]
    for stanza in stanzas:
        lines = [re.sub(r"\[.*?\]", "", line).strip() for line in stanza.split("\n")]
        lines =[_STRUCTURAL_LABELS_RE.sub("", line).strip() for line in lines]
        lines = [re.sub(r'[()"]', "", line).strip() for line in lines]
        lines =[_normalize_lyric_text(line) for line in lines if len(line) > 2]
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
    kept_stanza_embs =[all_stanza_embs[0]]

    for i in range(1, len(clean_stanzas)):
        current_embedding = all_stanza_embs[i]
        kept_embs_tensor = torch.stack(kept_stanza_embs)
        max_sim = util.cos_sim(current_embedding, kept_embs_tensor).max().item()
        if max_sim < threshold:
            kept_stanzas.append(clean_stanzas[i])
            kept_stanza_embs.append(current_embedding)

    if similarity_scope == "stanza":
        final_stanzas = kept_stanzas
    else:
        # --- Line-level deduplication within each kept stanza ---
        final_stanzas =[]
        for stanza in kept_stanzas:
            if len(stanza) == 1:
                final_stanzas.append(stanza)
                continue

            preprocessed =[f"passage: {preprocess_tweet(line)}" for line in stanza]
            all_embs = model.encode(preprocessed, convert_to_tensor=True)  # batch encode

            kept_lines = [stanza[0]]
            kept_embs =[all_embs[0]]
            for idx, line in enumerate(stanza[1:], start=1):
                current_emb = all_embs[idx]
                similarities =[util.cos_sim(current_emb, l_emb) for l_emb in kept_embs]
                if max([sim.item() for sim in similarities]) < line_threshold:
                    kept_lines.append(line)
                    kept_embs.append(current_emb)
            final_stanzas.append(kept_lines)

    # --- Return reconstructed lyrics with semantic formatting ---
    formatted_stanzas =[]
    for stanza in final_stanzas:
        if not stanza:
            continue
            
        # 1. El primer verso mantiene su capitalización original
        processed_lines = [stanza[0]]
        
        # 2. Siguientes versos: minúscula en la primera letra
        for line in stanza[1:]:
            if line: # check de seguridad
                formatted_line = line[0].lower() + line[1:]
                processed_lines.append(formatted_line)
                
        # 3. Unimos los versos de la misma estrofa con comas
        formatted_stanzas.append(", ".join(processed_lines))

    if not formatted_stanzas:
        return ""
        
    # 4. Unimos las estrofas con puntos y agregamos el punto final
    return ". ".join(formatted_stanzas) + "."


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
