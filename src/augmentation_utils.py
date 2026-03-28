import os
import re
import nltk
import numpy as np
import pandas as pd
import torch
import nlpaug.augmenter.word as naw
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# 1. SETUP & RESOURCES
# ─────────────────────────────────────────────────────────────
try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# Method mask constants
SYNONYM = 1
BACKTRANS_EN = 2
BACKTRANS_FR = 3
LLM_PARAPHRASE = 4

HELSINKI_BATCH_SIZE = 32
DEFAULT_METHOD_PROBS = [0.20, 0.30, 0.20, 0.30]

# ─────────────────────────────────────────────────────────────
# 2. LLM PARAPHRASING SETUP (via Local Ollama)
# ─────────────────────────────────────────────────────────────
llm_client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama", 
)

LLM_MODEL = "dolphin-llama3"

def _paraphrase_llm(text: str) -> str:
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Eres un experto compositor. Reescribe y parafrasea la siguiente letra "
                        "de canción en español. Cambia el vocabulario usando sinónimos y "
                        "altera la estructura, pero mantén el sentimiento original. "
                        "REGLA ESTRICTA: NO USES SALTOS DE LÍNEA. Todo el texto debe fluir en un "
                        "solo párrafo. Separa los versos con comas (,) y las estrofas con puntos (.). "
                        "No incluyas el título ni el artista."
                    )
                },
                # --- FEW-SHOT EXAMPLE: We SHOW the model exactly how to behave ---
                {
                    "role": "user",
                    "content": "title: Ejemplo, artist: Fake. Me duele el alma, cuando te vas, y me dejas solo. Vuelve pronto, te lo ruego, no me hagas sufrir."
                },
                {
                    "role": "assistant",
                    # Notice the output: totally different words, strictly one line, comma/period format!
                    "content": "Siento un gran vacío en mi interior, al verte partir, dejándome en total abandono. Regresa rápido a mi lado, te lo imploro, evita que siga padeciendo."
                },
                # --- ACTUAL INPUT ---
                {"role": "user", "content": text}
            ],
            temperature=0.7,  # Bumped to 0.7 for maximum synonym swapping
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return text

# ─────────────────────────────────────────────────────────────
# 3. BACKTRANSLATION (MarianMT)
# ─────────────────────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HELSINKI_MODELS = {
    "en": {"fwd": "Helsinki-NLP/opus-mt-es-en", "bwd": "Helsinki-NLP/opus-mt-en-es"},
    "fr": {"fwd": "Helsinki-NLP/opus-mt-es-fr", "bwd": "Helsinki-NLP/opus-mt-fr-es"}, # Swapped to French
}

_marian_cache: dict = {}

def _get_marian(model_name: str):
    if model_name not in _marian_cache:
        print(f"  [Helsinki] Loading {model_name} on {_device}...")
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(_device)
        mdl.eval()
        _marian_cache[model_name] = (tok, mdl)
    return _marian_cache[model_name]

def _translate_single(text: str, tokenizer, model) -> str:
    MAX_TOKENS = 490 
    
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= MAX_TOKENS:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(_device)
        with torch.no_grad():
            output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    segments = [s.strip() for s in re.split(r"(?<=\.) ", text) if s.strip()]
    chunks, current_chunk, current_len = [], [], 0

    for seg in segments:
        seg_len = len(tokenizer.encode(seg, add_special_tokens=False))
        
        if seg_len > MAX_TOKENS:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0
            
            sub_segments = [sub.strip() for sub in re.split(r"(?<=,) ", seg) if sub.strip()]
            for sub in sub_segments:
                sub_len = len(tokenizer.encode(sub, add_special_tokens=False))
                if sub_len > MAX_TOKENS:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_len = [], 0
                    words = sub.split()
                    for word in words:
                        word_len = len(tokenizer.encode(word, add_special_tokens=False))
                        if current_len + word_len > MAX_TOKENS and current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk, current_len = [], 0
                        current_chunk.append(word)
                        current_len += word_len
                elif current_len + sub_len > MAX_TOKENS and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_len = [sub], sub_len
                else:
                    current_chunk.append(sub)
                    current_len += sub_len

        elif current_len + seg_len > MAX_TOKENS and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [seg], seg_len
        else:
            current_chunk.append(seg)
            current_len += seg_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=512).to(_device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    translated_chunks = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return " ".join(translated_chunks)

def _translate(texts: list[str], model_name: str) -> list[str]:
    tokenizer, model = _get_marian(model_name)
    results = []
    for text in tqdm(texts, desc=f"Translating {model_name}", leave=False):
        try:
            results.append(_translate_single(text, tokenizer, model))
        except Exception as e:
            results.append(text)
    return results

def backtranslate_batch(texts: list[str], pivot_lang: str) -> list[str]:
    fwd_model = _HELSINKI_MODELS[pivot_lang]["fwd"]
    bwd_model = _HELSINKI_MODELS[pivot_lang]["bwd"]
    pivot_texts = _translate(texts, fwd_model)
    back_texts = _translate(pivot_texts, bwd_model)
    return [b if (b and isinstance(b, str)) else src for b, src in zip(back_texts, texts)]

# ─────────────────────────────────────────────────────────────
# 4. MAIN AUGMENTOR CLASS
# ─────────────────────────────────────────────────────────────
class LyricsAugmentor:
    def __init__(self, seed: int = 42, method_probs: list[float] | None = None):
        self.aug_syn = naw.SynonymAug(aug_src="wordnet", lang="spa", aug_p=0.1)
        self.rng = np.random.default_rng(seed)
        self.method_probs = method_probs or DEFAULT_METHOD_PROBS
        
        if len(self.method_probs) != 4:
            raise ValueError("method_probs must have exactly 4 values.")
        if not np.isclose(sum(self.method_probs), 1.0):
            raise ValueError("method_probs must sum to 1.0.")

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).split())

    def _is_valid_augmentation(self, original_text: str, augmented_text: str) -> bool:
        if not isinstance(augmented_text, str): return False
        orig = self._normalize_text(original_text)
        aug = self._normalize_text(augmented_text)
        
        if not aug or aug == orig or len(aug) < 8: return False
        
        ratio = len(aug) / max(1, len(orig))
        if ratio < 0.4 or ratio > 2.0: return False
        return True

    def _assign_method_mask(self, n: int) -> np.ndarray:
        return self.rng.choice(
            [SYNONYM, BACKTRANS_EN, BACKTRANS_FR, LLM_PARAPHRASE],
            size=n,
            p=self.method_probs,
        )

    def _run_backtranslations(self, texts, mask, indices, pivot_lang, method_id, results, methods):
        lang_indices = [i for i in indices if mask[i] == method_id]
        lang_texts = [texts[i] for i in lang_indices]
        if not lang_texts: return

        translated_all = []
        for start in tqdm(range(0, len(lang_texts), HELSINKI_BATCH_SIZE), desc=f"Backtrans {pivot_lang}", leave=False):
            batch = lang_texts[start : start + HELSINKI_BATCH_SIZE]
            translated_all.extend(backtranslate_batch(batch, pivot_lang))

        for i, aug_text in zip(lang_indices, translated_all):
            results[i] = aug_text
            methods[i] = f"backtranslation_{pivot_lang}"

    def augment_dataframe(self, df: pd.DataFrame, text_col="lyrics", label_col="label", target_classes=None, multiplier=1):
        """
        Augments the dataframe.
        - target_classes: None (augments all rows), int/str (augments one class), or list (augments specific classes).
        """
        print(f"--- Initiating Meaning-Preserving Augmentation (Factor x{multiplier}) ---")

        # Determine which rows to augment
        if target_classes is None:
            df_to_augment = df.copy().reset_index(drop=True)
            print("Target: All classes")
        else:
            if not isinstance(target_classes, list):
                target_classes = [target_classes]
            df_to_augment = df[df[label_col].isin(target_classes)].copy().reset_index(drop=True)
            print(f"Target classes: {target_classes}")

        all_texts, all_row_idx = [], []
        for i, row in df_to_augment.iterrows():
            text = str(row[text_col])
            if not text or text.lower() == "nan": continue
            for _ in range(multiplier):
                all_texts.append(text)
                all_row_idx.append(i)

        n = len(all_texts)
        if n == 0:
            print("No texts found to augment.")
            if "augmentation" not in df.columns: 
                df = df.copy()
                df["augmentation"] = "original"
            return df

        mask = self._assign_method_mask(n)
        results = list(all_texts) 
        methods = ["original_fallback"] * n

        # ── 1. Synonym ────────────────────────────────────────────
        syn_idx = [i for i, m in enumerate(mask) if m == SYNONYM]
        for i in tqdm(syn_idx, desc="Synonym Replacement"):
            try:
                results[i] = self.aug_syn.augment(all_texts[i])[0] 
                methods[i] = "synonym"
            except: pass

        # ── 2. LLM Paraphrase ─────────────────────────────────────
        llm_idx = [i for i, m in enumerate(mask) if m == LLM_PARAPHRASE]
        for i in tqdm(llm_idx, desc="LLM Paraphrase"):
            results[i] = _paraphrase_llm(all_texts[i])
            methods[i] = "llm_paraphrase"

        # ── 3 & 4. Backtranslation (Batched) ──────────────────────
        all_indices = list(range(n))
        self._run_backtranslations(all_texts, mask, all_indices, "en", BACKTRANS_EN, results, methods)
        self._run_backtranslations(all_texts, mask, all_indices, "fr", BACKTRANS_FR, results, methods)

        # ── Filtering and Assembly ────────────────────────────────
        filtered_results, filtered_methods, filtered_row_idx = [], [], []
        rejected = 0
        
        for row_idx, original_text, aug_text, method in zip(all_row_idx, all_texts, results, methods):
            if method != "original_fallback" and self._is_valid_augmentation(original_text, aug_text):
                filtered_row_idx.append(row_idx)
                filtered_results.append(self._normalize_text(aug_text))
                filtered_methods.append(method)
            else:
                rejected += 1

        print(f"  Augmentation diagnostics — Accepted: {len(filtered_results)}, Rejected: {rejected}")

        new_rows = []
        for row_idx, aug_text, method in zip(filtered_row_idx, filtered_results, filtered_methods):
            new_row = df_to_augment.loc[row_idx].copy()
            new_row[text_col] = aug_text
            new_row["augmentation"] = method
            new_rows.append(new_row)

        df_augmented = pd.DataFrame(new_rows)
        
        if "augmentation" not in df.columns: 
            df = df.copy()
            df["augmentation"] = "original"
            
        combined_df = pd.concat([df, df_augmented], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Augmentation completed: {len(df)} -> {len(combined_df)} total samples.")
        return combined_df