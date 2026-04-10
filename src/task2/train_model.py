"""
Entrenamiento y evaluación para un único modelo en Task 2.
"""

import argparse
import gc
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from augmentation_utils import LyricsAugmentor
from bert_pooling import build_bert_like_classifier, predict_with_chunks
from trainer import WeightedTrainer
from utils import DEFAULT_SEED, set_seed
from bert_model_configs import MODEL_CONFIGS, ModelConfig, apply_baseline_settings

SEED = DEFAULT_SEED
set_seed(SEED)

def create_label_column(df: pd.DataFrame, label_cols: list[str]) -> pd.Series:
    return df[label_cols].astype(int).values.tolist()

def main(model_name, augment=False, baseline=False):
    if baseline:
        apply_baseline_settings()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"El modelo '{model_name}' no existe en las configuraciones. Opciones: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]

    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task2/{pre}train_df.csv"
    VAL_PATH = f"../../data/task2/{pre}val_df.csv"
    DEV_PATH = f"../../data/task2/{pre}dev_df.csv"
    
    suffix = "_baseline" if baseline else ""
    suffix += "_aug" if augment else ""
    SAVE_DIR = f"../../models/task2/single/{model_name}{suffix}"

    label_cols = ["sexualization", "violence", "hate"]

    print(f"Cargando datos de entrenamiento desde {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["label"] = create_label_column(train_df, label_cols)
    print(f"Cargando datos de validación desde {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    val_df["label"] = create_label_column(val_df, label_cols)
    print(f"Cargando datos de test/dev desde {DEV_PATH}...")
    dev_df = pd.read_csv(DEV_PATH)
    dev_df["label"] = create_label_column(dev_df, label_cols)

    train_originals_labels = (
        train_df[train_df["augmentation"] == "original"]["label"]
        if "augmentation" in train_df.columns
        else train_df["label"]
    )
    n_samples = len(train_originals_labels)
    n_pos = np.array(
        [
            train_originals_labels.apply(lambda x: int(x[i])).sum()
            for i in range(len(label_cols))
        ]
    )
    n_neg = n_samples - n_pos
    weights_tensor = (n_neg / np.maximum(1, n_pos)).astype(float)

    augmentor = LyricsAugmentor()
    if augment:
        train_df = augmentor.augment_dataframe(
            train_df,
            minority_label=1, # Para multilabel puede variar, pero se mantiene la API actual
            multiplier=2,
        )

    train_ds = Dataset.from_pandas(train_df.rename(columns={"lyrics": "text"}), preserve_index=False)
    val_ds = Dataset.from_pandas(val_df.rename(columns={"lyrics": "text"}), preserve_index=False)
    dev_ds = Dataset.from_pandas(dev_df.rename(columns={"lyrics": "text"}), preserve_index=False)

    def compute_metrics(
        pred, label_names=["sexualization", "violence", "hate"], thresholds=None
    ):
        logits = pred.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        if thresholds is None:
            thresholds = [0.5] * probs.shape[1]
        thresholds = np.array(thresholds)
        preds = (probs >= thresholds).astype(int)
        true = pred.label_ids

        results = {}
        for i, name in enumerate(label_names):
            p = precision_score(true[:, i], preds[:, i], zero_division=0)
            r = recall_score(true[:, i], preds[:, i], zero_division=0)
            f1 = f1_score(true[:, i], preds[:, i], zero_division=0)
            results[f"{name}_precision"] = round(p, 4)
            results[f"{name}_recall"] = round(r, 4)
            results[f"{name}_f1"] = round(f1, 4)

        results["eval_f1_micro"] = round(f1_score(true, preds, average="micro", zero_division=0), 4)
        results["eval_f1_macro"] = round(f1_score(true, preds, average="macro", zero_division=0), 4)
        return results

    def make_tokenize_fn(tokenizer, cfg: ModelConfig):
        def tokenize_fn(batch):
            texts = batch["text"]
            if cfg.use_pysentimiento_preprocess:
                texts = [preprocess_tweet(t, lang="es") for t in texts]
            return tokenizer(texts, padding="max_length", truncation=True, max_length=cfg.max_len)
        return tokenize_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- INICIANDO ENTRENAMIENTO PARA {model_name} EN {device.type.upper()} ---")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    tokenize_fn = make_tokenize_fn(tokenizer, cfg)
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
    dev_tok = dev_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
    
    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("labels" if "labels" in val_tok.column_names else "label", "labels")
    dev_tok = dev_tok.rename_column("labels" if "labels" in dev_tok.column_names else "label", "labels")
    train_tok.set_format("torch")
    val_tok.set_format("torch")
    dev_tok.set_format("torch")

    model = build_bert_like_classifier(cfg, device, num_labels=3)
    checkpoints_path = os.path.join(SAVE_DIR, "checkpoints")

    args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=cfg.learning_rate,
        optim=cfg.optim,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=None,
        is_multilabel=True,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    print("Entrenando...")
    trainer.train()

    print("Evaluando en Dev Set...")
    pred_output = predict_with_chunks(
        dataset=dev_ds,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=cfg.max_len,
        batch_size=cfg.per_device_eval_batch_size if hasattr(cfg, "per_device_eval_batch_size") else 8,
        aggregation="max",
    )

    logits = pred_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    true_labels = pred_output.label_ids

    best_thr = 0.5
    opt_preds = (probs >= best_thr).astype(int)

    precision, recall, f1_opt, _ = precision_recall_fscore_support(
        true_labels, opt_preds, average="macro", zero_division=0.0
    )
    acc_opt = accuracy_score(true_labels, opt_preds)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    
    results = {
        "Modelo": model_name,
        "F1-Macro": f1_opt,
        "Accuracy": acc_opt,
        "Precision": precision,
        "Recall": recall,
        "Best-Threshold": best_thr,
    }

    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nModelo guardado en: {SAVE_DIR}")
    print(f"Resultados: {results}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Entrenamiento de un modelo basado en BERT para Task 2")
    arg_parser.add_argument("--model", type=str, default="BETO", help="Nombre del modelo en config")
    arg_parser.add_argument("--augment", action="store_true", help="Activar augmentación de datos")
    arg_parser.add_argument("--baseline", action="store_true", help="Usar configuraciones baseline y datos crudos")
    args = arg_parser.parse_args()
    main(model_name=args.model, augment=args.augment, baseline=args.baseline)
