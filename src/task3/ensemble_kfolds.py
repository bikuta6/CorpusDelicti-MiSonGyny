import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import config and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS, ModelConfig

from bert_pooling import load_bert_like_classifier
from utils import DEFAULT_SEED, set_seed

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "BETO"  # Must match the key in MODEL_CONFIGS
KFOLD_DIR = f"../../models/task3/kfold_{MODEL_NAME}"
TEST_PATH = "../../data/task3/processed_test.csv"
RESULTS_FILE = f"../../results/task3/submission_kfold_{MODEL_NAME}.csv"
BATCH_SIZE = 32
N_FOLDS = 5
SEED = DEFAULT_SEED

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# CHUNKING HELPERS
# ─────────────────────────────────────────────────────────────

CHUNK_STRIDE = 256  # Overlap between chunks


def chunk_tokens(token_ids: list[int], max_len: int, stride: int) -> list[list[int]]:
    """
    Split token_ids into overlapping chunks.
    Each chunk includes CLS at start and SEP at end.
    """
    if len(token_ids) <= max_len:
        return [token_ids]

    cls_token = token_ids[0]
    sep_token = token_ids[-1]
    content = token_ids[1:-1]  # Remove CLS and SEP

    chunks = []
    content_max = max_len - 2  # Reserve space for CLS and SEP

    for start in range(0, len(content), stride):
        chunk_content = content[start : start + content_max]
        chunk = [cls_token] + chunk_content + [sep_token]
        chunks.append(chunk)

        # Stop if we've covered all content
        if start + content_max >= len(content):
            break

    return chunks


def get_chunked_probability(
    text: str,
    tokenizer,
    model,
    max_len: int,
    stride: int,
    device: torch.device,
    use_pysentimiento_preprocess: bool = False,
) -> float:
    """
    Get probability for class Y using sliding window with max pooling.
    Returns single probability (max across all chunks).
    """
    if use_pysentimiento_preprocess:
        text = preprocess_tweet(text, lang="es")

    # Tokenize without truncation
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    token_ids = encoding["input_ids"]

    # Create chunks
    chunks = chunk_tokens(token_ids, max_len, stride)

    # Process all chunks
    chunk_probs = []
    for chunk in chunks:
        # Pad to max_len
        pad_len = max_len - len(chunk)
        input_ids = chunk + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(chunk) + [0] * pad_len

        # Create tensors
        input_ids_tensor = torch.tensor([input_ids], device=device)
        attention_mask_tensor = torch.tensor([attention_mask], device=device)

        # Inference
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            prob_M = probs[0, 1].item()
            chunk_probs.append(prob_M)

    # Max pooling across chunks
    return max(chunk_probs)


# ─────────────────────────────────────────────────────────────
# INFERENCE LOGIC
# ─────────────────────────────────────────────────────────────


def get_fold_probabilities(fold_path, df_test, cfg):
    print(f"  -> Processing Fold at: {fold_path}")
    tokenizer = AutoTokenizer.from_pretrained(fold_path)
    model = load_bert_like_classifier(fold_path, device)
    model.eval()

    texts = df_test["lyrics"].fillna("").astype(str).tolist()
    probs = []

    # Process each text with chunking
    for text in tqdm(texts, leave=False, desc="Chunked inference"):
        prob_M = get_chunked_probability(
            text=text,
            tokenizer=tokenizer,
            model=model,
            max_len=cfg.max_len,
            stride=CHUNK_STRIDE,
            device=device,
            use_pysentimiento_preprocess=cfg.use_pysentimiento_preprocess,
        )
        probs.append(prob_M)

    # Cleanup memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(probs)


def main():
    df_test = pd.read_csv(TEST_PATH)
    cfg = MODEL_CONFIGS[MODEL_NAME]

    all_folds_probs = []

    print(f"🚀 Starting K-Fold Ensemble for {MODEL_NAME}...")

    for f in range(1, N_FOLDS + 1):
        fold_path = os.path.join(KFOLD_DIR, f"fold_{f}")
        if not os.path.exists(fold_path):
            print(f"  ⚠️ Warning: {fold_path} not found. Skipping.")
            continue

        p = get_fold_probabilities(fold_path, df_test, cfg)
        all_folds_probs.append(p)

    if not all_folds_probs:
        print("❌ No models found to ensemble!")
        return

    # Average probabilities across folds
    avg_probs = np.mean(all_folds_probs, axis=0)

    # Global threshold (or you could average the best thresholds found in kfold)
    threshold = 0.5

    predictions = ["Y" if p >= threshold else "N" for p in avg_probs]

    # Save Results
    df_test["label"] = predictions
    df_test[["id", "label"]].to_csv(RESULTS_FILE, index=False)

    print(f"\n✅ K-Fold Ensemble complete!")
    print(f"📂 Saved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
