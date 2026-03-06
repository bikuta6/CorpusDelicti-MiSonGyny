import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import random
import nltk
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

# Asegurar que los recursos necesarios de NLTK estén presentes
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')

# Method mask constants
SYNONYM      = 1
WORD_SWAP    = 2
CHAR_NOISE   = 3
BACKTRANS_EN = 4
BACKTRANS_RU = 5

HELSINKI_BATCH_SIZE = 32

# ─────────────────────────────────────────────────────────────
# MARIAN MODEL CACHE
# ─────────────────────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HELSINKI_MODELS = {
    "en": {
        "fwd": "Helsinki-NLP/opus-mt-es-en",
        "bwd": "Helsinki-NLP/opus-mt-en-es",
    },
    "ru": {
        "fwd": "Helsinki-NLP/opus-mt-es-ru",
        "bwd": "Helsinki-NLP/opus-mt-ru-es",
    },
}

# cache: model_name -> (tokenizer, model)
_marian_cache: dict = {}


def _get_marian(model_name: str):
    """Lazy-load and cache a MarianMT model+tokenizer."""
    if model_name not in _marian_cache:
        print(f"  [Helsinki] Loading {model_name} ...")
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(_device)
        mdl.eval()
        _marian_cache[model_name] = (tok, mdl)
    return _marian_cache[model_name]


def _translate_single(text: str, tokenizer, model) -> str:
    """
    Translate a single text, chunking by sentence if it exceeds 512 tokens.
    Joins translated chunks back with a space.
    """
    MAX_TOKENS = 490  # leave margin for special tokens

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= MAX_TOKENS:
        # Fast path: fits in one pass
        inputs = tokenizer([text], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(_device)
        with torch.no_grad():
            output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # Split by stanzas (". ") then verses (", ") to respect lyric structure
    import re
    # Split on ", " or ". " keeping the punctuation attached to the preceding segment
    segments = [s.strip() for s in re.split(r'(?<=[.,]) ', text) if s.strip()]

    chunks, current_chunk, current_len = [], [], 0

    for seg in segments:
        seg_len = len(tokenizer.encode(seg, add_special_tokens=False))
        if seg_len > MAX_TOKENS:
            # Single segment too long — fall back to word-level splitting
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0
            words = seg.split()
            for word in words:
                word_len = len(tokenizer.encode(word, add_special_tokens=False))
                if current_len + word_len > MAX_TOKENS and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_len = [], 0
                current_chunk.append(word)
                current_len += word_len
        elif current_len + seg_len > MAX_TOKENS and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [seg], seg_len
        else:
            current_chunk.append(seg)
            current_len += seg_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Translate all chunks in one batched call
    inputs = tokenizer(chunks, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(_device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    translated_chunks = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return " ".join(translated_chunks)


def _translate(texts: list[str], model_name: str) -> list[str]:
    """Translate a list of texts using a MarianMT model, handling texts > 512 tokens."""
    tokenizer, model = _get_marian(model_name)
    results = []
    texts_tqdm = tqdm(texts, desc=f"Translating with {model_name}", leave=False)
    for text in texts_tqdm:
        try:
            results.append(_translate_single(text, tokenizer, model))
        except Exception as e:
            print(f"  [Helsinki] Translation error ({model_name}): {e}")
            results.append(text)  # fallback to original for this item only
    return results


# =========================================================
# Backtranslation
# =========================================================

def backtranslate_batch(texts: list[str], pivot_lang: str) -> list[str]:
    """
    Batch backtranslation ES -> pivot_lang -> ES using local MarianMT models.
    Supported pivot_lang: 'en', 'ru'.
    Returns list with same length as input.
    """
    if pivot_lang not in _HELSINKI_MODELS:
        raise ValueError(f"Unsupported pivot_lang: {pivot_lang}")

    fwd_model = _HELSINKI_MODELS[pivot_lang]["fwd"]
    bwd_model = _HELSINKI_MODELS[pivot_lang]["bwd"]

    pivot_texts = _translate(texts, fwd_model)
    back_texts  = _translate(pivot_texts, bwd_model)

    # Per-item safe merge
    return [
        b if (b and isinstance(b, str)) else src
        for b, src in zip(back_texts, texts)
    ]


class LyricsAugmentor:
    def __init__(self):
        # 1. Sustitución por Sinónimos (Español)
        self.aug_syn = naw.SynonymAug(aug_src='wordnet', lang='spa')
        
        # 2. Random Swap (Intercambia palabras)
        self.aug_swap = naw.RandomWordAug(action="swap", aug_p=0.1)
        
        # 3. Ruido de caracteres (Simula typos)
        self.aug_char = nac.RandomCharAug(action="substitute", aug_char_p=0.1, aug_word_p=0.1)

    def _assign_method_mask(self, n: int) -> np.ndarray:
        """
        Randomly assigns a method to each of the n samples.
        Distribution: 25% each → SYNONYM, WORD_SWAP, CHAR_NOISE, BACKTRANS (50/50 EN/RU)
        """
        choices = np.random.choice(
            [SYNONYM, WORD_SWAP, CHAR_NOISE, BACKTRANS_EN, BACKTRANS_RU],
            size=n,
            p=[0.60, 0.20, 0.20, 0.0, 0.0]  # backtrans split evenly
        )
        return choices

    def _run_backtranslations(self, texts: list[str], mask: np.ndarray, indices: list[int], pivot_lang: str, method_id: int, results: list[str], methods: list[str]):
        """
        Runs batched backtranslation for one pivot language, respecting HELSINKI_BATCH_SIZE.
        """
        lang_indices = [i for i in indices if mask[i] == method_id]
        lang_texts   = [texts[i] for i in lang_indices]

        if not lang_texts:
            return

        print(f"  -> Backtranslation ({pivot_lang}): {len(lang_texts)} texts in batches of {HELSINKI_BATCH_SIZE}...")

        translated_all = []
        ranged_tqdm = tqdm(range(0, len(lang_texts), HELSINKI_BATCH_SIZE), desc=f"Backtrans {pivot_lang}", leave=False)
        for start in ranged_tqdm:
            batch = lang_texts[start : start + HELSINKI_BATCH_SIZE]
            translated_all.extend(backtranslate_batch(batch, pivot_lang))

        for i, aug_text in zip(lang_indices, translated_all):
            results[i] = aug_text
            methods[i] = f"backtranslation_{pivot_lang}"

    def augment_dataframe(self, df, text_col="lyrics", label_col="label", minority_label=1, multiplier=1):
        """
        Aumenta SOLO la clase minoritaria del dataframe entregado.
        Uses pre-assigned method masks and batched translation for speed.
        """
        print(f"--- Iniciando Aumentación (Factor x{multiplier}) ---")

        minority_df = df[df[label_col] == minority_label].copy().reset_index(drop=True)

        # Build flat list of (row_index, text) for all augmentation slots
        all_texts   = []
        all_row_idx = []
        for i, row in minority_df.iterrows():
            text = str(row[text_col])
            if not text or text.lower() == "nan":
                continue
            for _ in range(multiplier):
                all_texts.append(text)
                all_row_idx.append(i)

        n = len(all_texts)
        mask    = self._assign_method_mask(n)
        results = list(all_texts)           # pre-fill with originals as fallback
        methods = ["original_fallback"] * n

        print(f"  Method distribution — "
              f"synonym: {(mask==SYNONYM).sum()}, "
              f"word_swap: {(mask==WORD_SWAP).sum()}, "
              f"char_noise: {(mask==CHAR_NOISE).sum()}, "
              f"backtrans_en: {(mask==BACKTRANS_EN).sum()}, "
              f"backtrans_ru: {(mask==BACKTRANS_RU).sum()}")

        # ── 1. Synonym ────────────────────────────────────────────
        syn_idx = [i for i, m in enumerate(mask) if m == SYNONYM]
        for i in tqdm(syn_idx, desc="Synonym"):
            try:
                results[i] = self.aug_syn.augment(all_texts[i])[0]
                methods[i] = "synonym"
            except Exception:
                pass

        # ── 2. Word Swap ──────────────────────────────────────────
        swap_idx = [i for i, m in enumerate(mask) if m == WORD_SWAP]
        for i in tqdm(swap_idx, desc="Word Swap"):
            try:
                results[i] = self.aug_swap.augment(all_texts[i])[0]
                methods[i] = "word_swap"
            except Exception:
                pass

        # ── 3. Char Noise ─────────────────────────────────────────
        char_idx = [i for i, m in enumerate(mask) if m == CHAR_NOISE]
        for i in tqdm(char_idx, desc="Char Noise"):
            try:
                results[i] = self.aug_char.augment(all_texts[i])[0]
                methods[i] = "char_noise"
            except Exception:
                pass

        # ── 4 & 5. Backtranslation (batched) ─────────────────────
        all_indices = list(range(n))
        self._run_backtranslations(all_texts, mask, all_indices, "en", BACKTRANS_EN, results, methods)
        self._run_backtranslations(all_texts, mask, all_indices, "ru", BACKTRANS_RU, results, methods)

        # ── Build augmented DataFrame ─────────────────────────────
        new_rows = []
        for i, (row_idx, aug_text, method) in enumerate(zip(all_row_idx, results, methods)):
            new_row = minority_df.loc[row_idx].copy()
            new_row[text_col]       = aug_text
            new_row["augmentation"] = method
            new_rows.append(new_row)

        df_augmented = pd.DataFrame(new_rows)

        if "augmentation" not in df.columns:
            df = df.copy()
            df["augmentation"] = "original"

        combined_df = pd.concat([df, df_augmented], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Aumentación completada: {len(df)} -> {len(combined_df)} muestras.")
        return combined_df