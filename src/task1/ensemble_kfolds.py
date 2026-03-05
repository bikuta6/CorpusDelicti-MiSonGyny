import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pysentimiento.preprocessing import preprocess_tweet

# Import config and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS, ModelConfig
from utils import set_seed, DEFAULT_SEED

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "BETO"  # Must match the key in MODEL_CONFIGS
KFOLD_DIR = f"../../models/task1/kfold_{MODEL_NAME}"
TEST_PATH = "../../data/task1/processed_test.csv"
RESULTS_FILE = f"../../results/task1/submission_kfold_{MODEL_NAME}.csv"
BATCH_SIZE = 32
N_FOLDS = 5
SEED = DEFAULT_SEED

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# SMART TOKENIZATION (Same as training)
# ─────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer, cfg: ModelConfig):
    def tokenize_fn(batch):
        texts = batch["text"]
        if cfg.use_pysentimiento_preprocess:
            texts = [preprocess_tweet(t, lang="es") for t in texts]

        tokenized = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        max_len = cfg.max_len
        input_ids, attention_mask = [], []

        for ids in tokenized["input_ids"]:
            if len(ids) <= max_len:
                pad_len = max_len - len(ids)
                padded = ids + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            else:
                cls_token, sep_token = ids[0], ids[-1]
                content = ids[1:-1]
                head_len = int((max_len - 2) * 0.75)
                tail_len = (max_len - 2) - head_len
                padded = [cls_token] + content[:head_len] + content[-tail_len:] + [sep_token]
                mask = [1] * max_len
            
            input_ids.append(padded)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return tokenize_fn

# ─────────────────────────────────────────────────────────────
# INFERENCE LOGIC
# ─────────────────────────────────────────────────────────────

def get_fold_probabilities(fold_path, df_test, cfg):
    print(f"  -> Processing Fold at: {fold_path}")
    tokenizer = AutoTokenizer.from_pretrained(fold_path)
    model = AutoModelForSequenceClassification.from_pretrained(fold_path).to(device)
    model.eval()

    ds = Dataset.from_pandas(df_test.rename(columns={"lyrics": "text"}), preserve_index=False)
    tok_fn = make_tokenize_fn(tokenizer, cfg)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds_tok.set_format("torch")
    
    loader = DataLoader(ds_tok, batch_size=BATCH_SIZE)
    probs = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Inferencing"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            p = torch.softmax(outputs.logits, dim=-1)[:, 1] # Probability of class 'M'
            probs.extend(p.cpu().numpy())
    
    # Cleanup memory
    del model, tokenizer, ds_tok, loader
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
    
    predictions = ["M" if p >= threshold else "NM" for p in avg_probs]
    
    # Save Results
    df_test["label"] = predictions
    df_test[["id", "label"]].to_csv(RESULTS_FILE, index=False)
    
    print(f"\n✅ K-Fold Ensemble complete!")
    print(f"📂 Saved results to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()