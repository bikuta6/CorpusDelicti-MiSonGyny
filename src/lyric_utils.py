import re
import unicodedata
from sentence_transformers import SentenceTransformer, util
from pysentimiento.preprocessing import preprocess_tweet
import torch
import pandas as pd

# Structural section labels to remove from parentheses (Spanish + English)
_STRUCTURAL_LABELS_RE = re.compile(
    r'\(\s*(coro|verso|estrofa|puente|bis|intro|outro|chorus|hook|bridge|verse|refr[aá]n|interludio)\s*\d*\s*\)',
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
    text = unicodedata.normalize('NFC', text)
    # Remove zero-width and ASCII control characters (except newlines/tabs)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Normalize fancy quotes/apostrophes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''`]", "'", text)
    # Collapse repeated punctuation
    text = re.sub(r'([!?]){2,}', r'\1', text)
    # Normalize jajaja variants (jajajajaja → jajaja)
    text = re.sub(r'(ja){3,}', 'jajaja', text, flags=re.IGNORECASE)
    return text


def remove_redundant_lyrics(model: SentenceTransformer, text: str, threshold: float = 0.82) -> str:
    """
    Optimized for Spanglish lyrics and nuanced semantic redundancy using E5-Large.
    Threshold 0.82 is the sweet spot for E5 to prevent over-deduplication.
    """
    if not text.strip():
        return ""

    # 1. Structural cleaning
    lines = text.split("\n\n")
    # Remove square-bracket markers: [Chorus], [Verse 1], etc.
    lines = [re.sub(r'\[.*?\]', '', line).strip() for line in lines]
    # Remove parenthesised STRUCTURAL labels only: (Coro), (Verso 2), (Chorus)...
    lines = [_STRUCTURAL_LABELS_RE.sub('', line).strip() for line in lines]
    # Strip parenthesis and double-quote characters but keep their content:
    # "(puta madre)" → "puta madre", '"yeah baby"' → "yeah baby"
    lines = [re.sub(r'[()"]', '', line).strip() for line in lines]
    # Apply light unicode/punctuation normalization
    lines = [_normalize_lyric_text(line) for line in lines]
    # Drop empty lines and very short noise (single characters/grunts)
    lines = [line for line in lines if len(line) > 2]

    if not lines:
        return ""

    # 2. Prepare for embedding (E5 models require the "passage: " prefix)
    processed_lines = [f"passage: {preprocess_tweet(line)}" for line in lines]

    # 3. Embedding generation
    embeddings = model.encode(processed_lines, convert_to_tensor=True)

    kept_indices = [0]  # Always keep the first block

    for i in range(1, len(processed_lines)):
        # Compare against all already-kept blocks
        similarities = util.cos_sim(embeddings[i], embeddings[kept_indices])
        # Keep only if it contributes new meaning
        if torch.max(similarities).item() < threshold:
            kept_indices.append(i)

    # 4. Return ORIGINAL lines (not processed_lines) so slang, spelling and
    # offensive terms remain intact for the downstream classifier
    return "\n".join([lines[idx] for idx in kept_indices])



if __name__ == "__main__":
    # Test with Spanglish and similar meanings (nuance check)
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    sample_lyrics = pd.read_csv("../data/task1/train.csv")["lyrics"]
    # Find the longest lyrics to test the function on a complex case
    longest_idx = sample_lyrics.apply(lambda x: len(x.split())).idxmax()
    sample_lyrics = sample_lyrics.iloc[longest_idx:longest_idx+1].tolist()
    print("--- Original Lyrics ---")
    print(sample_lyrics[0])
    print("Number of words:", len(sample_lyrics[0].split()))
    
    # 0.80 is usually the "sweet spot" for multilingual deduplication
    cleaned_text = [remove_redundant_lyrics(model, lyrics, threshold=0.90) for lyrics in sample_lyrics]
    print("--- Cleaned Lyrics ---")
    print(cleaned_text[0])
    print("Number of words after cleaning:", len(cleaned_text[0].split()))
    