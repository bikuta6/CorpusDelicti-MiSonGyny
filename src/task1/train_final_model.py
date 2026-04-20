"""
Entrenamiento y evaluación final para un único modelo en Task 1: Clasificación Binaria de Misoginia.
Entrena con un split 80/20 de los datos.
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pandas.core.arrays import base
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sympy.geometry.plane import t
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS, ModelConfig, apply_baseline_settings
from augment_loading import augment_df

from bert_pooling import build_bert_like_classifier, predict_with_chunks
from random_crop_collator import RandomCropDataCollator
from trainer import WeightedTrainer
from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED
set_seed(SEED)


def find_best_threshold(true_labels, probs, step=0.01):
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_thr = 0.5
    best_f1 = 0.0

    true_labels = np.array(true_labels)
    probs = np.array(probs)

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(true_labels, preds, average="macro", zero_division=0.0)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return round(best_thr, 3), round(best_f1, 4)


def main(model_name, baseline=False, processed=False, epochs=None, augment=False):
    if baseline:
        apply_baseline_settings()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"El modelo '{model_name}' no existe en las configuraciones. Opciones: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_name]

    pre = "processed_" if processed else ""
    # Cargar los datos completos
    DATA_PATH = f"../../data/task1/{pre}train.csv"
    if not os.path.exists(DATA_PATH):
        # Fallback a train_df si train.csv no existe
        DATA_PATH = f"../../data/task1/{pre}train_df.csv"

    suffix = ""
    if baseline:
        suffix += "_baseline"
    if augment:
        suffix += "_augmented"
    SAVE_DIR = f"../../models/task1/final/{model_name}{suffix if suffix else ''}"

    print(f"Cargando todos los datos desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    if augment and not baseline:
        print("Aplicando data augmentation...")
        df = augment_df(df, aug_path="../../data/processed_train_augmented.csv")

    # Asumimos que los labels pueden venir como 'NM'/'M' o numéricos
    if True:
        df["label"] = df["label"].apply(
            lambda x: 1 if str(x).strip().upper() in ["M", "1", "1.0"] else 0
        )

    # print("Dividiendo en train y eval (80/20)...")
    # , eval_df = train_test_split(
    #   df, test_size=0.2, random_state=SEED, stratify=df["label"]
    # )
    train_df = df.copy()
    eval_df = df.head(10).copy()  # Dummy eval set just for Trainer compatibility

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

    train_ds = Dataset.from_pandas(
        train_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    eval_ds = Dataset.from_pandas(
        eval_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

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

    def make_tokenize_fn(tokenizer, cfg: ModelConfig, training: bool = False):
        def tokenize_fn(batch):
            texts = batch["text"]
            if cfg.use_pysentimiento_preprocess:
                texts = [preprocess_tweet(t, lang="es") for t in texts]
            if training:
                return tokenizer(
                    texts,
                    padding=False,
                    truncation=False,
                )
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_len,
            )

        return tokenize_fn

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(
        f"--- INICIANDO ENTRENAMIENTO FINAL PARA {model_name} EN {device.type.upper()} ---"
    )
    print(
        f"Config: lr={cfg.learning_rate}, dropout_cls={cfg.classifier_dropout}, max_len={cfg.max_len}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    train_tokenize_fn = make_tokenize_fn(tokenizer, cfg, training=True)
    eval_tokenize_fn = make_tokenize_fn(tokenizer, cfg, training=False)

    train_tok = train_ds.map(
        train_tokenize_fn,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )

    train_tok = train_tok.rename_column("label", "labels")
    train_tok.set_format("torch")

    eval_tok = eval_ds.map(
        eval_tokenize_fn,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )
    eval_tok = eval_tok.rename_column("label", "labels")
    eval_tok.set_format("torch")

    model = build_bert_like_classifier(cfg, device, num_labels=2)
    checkpoints_path = os.path.join(SAVE_DIR, "checkpoints")

    args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=cfg.learning_rate,
        optim=cfg.optim,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=epochs if epochs is not None else cfg.num_train_epochs,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=False,
    )

    train_collator = RandomCropDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_len,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=None,
    )

    print("Entrenando...")
    trainer.train()
    pred_output = predict_with_chunks(
        dataset=eval_ds,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=cfg.max_len,
        batch_size=cfg.per_device_eval_batch_size
        if hasattr(cfg, "per_device_eval_batch_size")
        else 8,
        aggregation="max",
    )

    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    true_labels = pred_output.label_ids

    base_f1 = f1_score(
        true_labels, (probs >= 0.5).astype(int), average="macro", zero_division=0.0
    )
    print(f"F1 Macro con umbral 0.5: {base_f1:.4f}")
    thr, f1 = find_best_threshold(true_labels, probs)
    print(f"Mejor umbral encontrado: {thr} con F1 Macro: {f1}")
    print("Evaluando en el conjunto de validación...")
    eval_results = trainer.evaluate()
    print(eval_results)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    results = {
        "Modelo": model_name,
        "Epochs": epochs if epochs is not None else cfg.num_train_epochs,
        "Trained_on": "80/20 Split",
        "Best_Threshold": thr,
        "eval_f1_macro": eval_results["eval_eval_f1_macro"]
        if "eval_eval_f1_macro" in eval_results
        else eval_results.get("eval_f1_macro", 0.0),
        "eval_accuracy": eval_results["eval_accuracy"]
        if "eval_accuracy" in eval_results
        else 0.0,
    }

    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nModelo guardado en: {SAVE_DIR}")
    print(f"Resultados: {results}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Entrenamiento final de un modelo basado en BERT para Task 1"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="BETO",
        help="Nombre del modelo en config (ej. BETO)",
    )
    arg_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usar configuraciones baseline",
    )
    arg_parser.add_argument(
        "--processed",
        action="store_true",
        help="Usar dataset procesado",
    )
    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de epochs a entrenar (sobreescribe la config)",
    )
    args = arg_parser.parse_args()
    main(
        model_name=args.model,
        baseline=args.baseline,
        processed=args.processed,
        epochs=args.epochs,
    )
