import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
)
from bert_model_configs import MODEL_CONFIGS, ModelConfig

# Asegurar que cargamos las funciones del script original
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_seed, DEFAULT_SEED
from train_single_model import compute_metrics, load_model_with_config, make_training_args
from trainer import WeightedTrainer

# Configuración
MODEL_NAME = "BETO"  # Cambiar por el modelo deseado
N_SPLITS = 5
DATA_PATH = "../../data/task1/processed_train.csv"
SAVE_DIR = f"../../models/task1/kfold_{MODEL_NAME}"
SEED = DEFAULT_SEED
set_seed(SEED)

def make_tokenize_fn(tokenizer, cfg: ModelConfig):
    """Tokenization with smart head+tail truncation (75% Head, 25% Tail)."""
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

def run_kfold():
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["label"].map({"NM": 0, "M": 1})
    cfg = MODEL_CONFIGS[MODEL_NAME]
    
    # --- Lógica Anti-Leakage ---
    originals_df = df[df["augmentation"] == "original"].copy()
    id_to_label = originals_df.drop_duplicates("id").set_index("id")["label"]
    unique_ids = id_to_label.index.to_numpy()
    stratify_labels = id_to_label.values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(unique_ids, stratify_labels)):
        print(f"\n>>> INICIANDO FOLD {fold+1}/{N_SPLITS}")
        
        train_ids = set(unique_ids[train_idx])
        val_ids = set(unique_ids[val_idx])

        train_df = df[df["id"].isin(train_ids)].sample(frac=1, random_state=SEED)
        val_df = originals_df[originals_df["id"].isin(val_ids)].copy()

        # Calcular Pesos
        train_orig_labels = train_df[train_df["augmentation"] == "original"]["label"]
        w0 = len(train_orig_labels) / (2 * (train_orig_labels == 0).sum())
        w1 = len(train_orig_labels) / (2 * (train_orig_labels == 1).sum())
        weights = torch.tensor([w0, w1]).float().to(device)

        # Dataset
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        tokenize_fn = make_tokenize_fn(tokenizer, cfg)
        
        train_ds = Dataset.from_pandas(train_df.rename(columns={"lyrics": "text"}), preserve_index=False)
        val_ds = Dataset.from_pandas(val_df.rename(columns={"lyrics": "text"}), preserve_index=False)

        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"]).rename_column("label", "labels")
        val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"]).rename_column("label", "labels")
        train_tok.set_format("torch")
        val_tok.set_format("torch")

        # Modelo y Entrenamiento
        model = load_model_with_config(cfg.model_id, cfg, device)
        fold_path = os.path.join(SAVE_DIR, f"fold_{fold+1}")
        args = make_training_args(cfg, os.path.join(fold_path, "checkpoints"))

        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
            compute_metrics=compute_metrics, class_weights=weights,
            loss_type=cfg.loss_type, focal_gamma=cfg.focal_gamma
        )

        trainer.train()
        trainer.save_model(fold_path)
        tokenizer.save_pretrained(fold_path)

        del trainer, model, tokenizer, train_tok, val_tok
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_kfold()