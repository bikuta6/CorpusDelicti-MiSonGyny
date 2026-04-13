"""
Entrenamiento y evaluación para un único modelo en Task 1: Clasificación Binaria de Misoginia.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
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

def main(model_name, augment=False, baseline=False):
    if baseline:
        apply_baseline_settings()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"El modelo '{model_name}' no existe en las configuraciones. Opciones: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]

    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task1/{pre}train_df.csv"
    VAL_PATH = f"../../data/task1/{pre}val_df.csv"
    DEV_PATH = f"../../data/task1/{pre}dev_df.csv"
    
    suffix = "_baseline" if baseline else ""
    suffix += "_aug" if augment else ""
    SAVE_DIR = f"../../models/task1/single/{model_name}{suffix}"

    print(f"Cargando datos de entrenamiento desde {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["label"] = train_df["label"].map({"NM": 0, "M": 1})
    print(f"Cargando datos de validación desde {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    val_df["label"] = val_df["label"].map({"NM": 0, "M": 1})
    print(f"Cargando datos de test/dev desde {DEV_PATH}...")
    dev_df = pd.read_csv(DEV_PATH)
    dev_df["label"] = dev_df["label"].map({"NM": 0, "M": 1})

    train_originals_labels = (
        train_df[train_df["augmentation"] == "original"]["label"]
        if "augmentation" in train_df.columns
        else train_df["label"]
    )
    n_pos = (train_originals_labels == 1).sum()
    n_neg = (train_originals_labels == 0).sum()
    total = n_neg + n_pos
    w0 = total / (2 * n_neg)
    w1 = total / (2 * n_pos)
    weights_tensor = torch.sqrt(torch.tensor([w0, w1]).float())

    augmentor = LyricsAugmentor()
    if augment:
        train_df = augmentor.augment_dataframe(
            train_df,
            multiplier=2,
        )

    train_ds = Dataset.from_pandas(train_df.rename(columns={"lyrics": "text"}), preserve_index=False)
    val_ds = Dataset.from_pandas(val_df.rename(columns={"lyrics": "text"}), preserve_index=False)
    dev_ds = Dataset.from_pandas(dev_df.rename(columns={"lyrics": "text"}), preserve_index=False)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0.0
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": round(acc, 4),
            "eval_f1_macro": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    def make_tokenize_fn(tokenizer, cfg: ModelConfig):
        def tokenize_fn(batch):
            texts = batch["text"]
            if cfg.use_pysentimiento_preprocess:
                texts = [preprocess_tweet(t, lang="es") for t in texts]
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_len,
            )
        return tokenize_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- INICIANDO ENTRENAMIENTO PARA {model_name} EN {device.type.upper()} ---")
    print(f"Config: lr={cfg.learning_rate}, dropout_cls={cfg.classifier_dropout}, max_len={cfg.max_len}")

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

    model = build_bert_like_classifier(cfg, device, num_labels=2)
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
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
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
    arg_parser = argparse.ArgumentParser(description="Entrenamiento de un modelo basado en BERT para Task 1")
    arg_parser.add_argument("--model", type=str, default="BETO", help="Nombre del modelo en config (ej. BETO)")
    arg_parser.add_argument("--augment", action="store_true", help="Activar augmentación de datos")
    arg_parser.add_argument("--baseline", action="store_true", help="Usar configuraciones baseline y datos crudos")
    args = arg_parser.parse_args()
    main(model_name=args.model, augment=args.augment, baseline=args.baseline)
