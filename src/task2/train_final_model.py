"""
Entrenamiento y evaluación final para un único modelo en Task 2: Clasificación Multi-etiqueta.
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
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
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


def create_label_column(df: pd.DataFrame, label_cols: list[str]) -> pd.Series:
    return df[label_cols].astype(int).values.tolist()


def find_best_thresholds(true_labels, probs, step=0.01):
    """Encuentra el mejor threshold para cada clase de forma independiente."""
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_thrs = [0.5] * probs.shape[1]
    best_f1s = [0.0] * probs.shape[1]

    true_labels = np.array(true_labels)
    probs = np.array(probs)

    for i in range(probs.shape[1]):
        best_t = 0.5
        best_f = 0.0
        for thr in thresholds:
            preds = (probs[:, i] >= thr).astype(int)
            f = f1_score(true_labels[:, i], preds, zero_division=0.0)
            if f > best_f:
                best_f = f
                best_t = thr
        best_thrs[i] = round(best_t, 3)
        best_f1s[i] = round(best_f, 4)

    return best_thrs, best_f1s


def main(model_name, baseline=False, processed=False, epochs=None, augment=False):
    if baseline:
        apply_baseline_settings()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"El modelo '{model_name}' no existe en las configuraciones. Opciones: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_name]

    pre = "processed_" if processed else ""
    DATA_PATH = f"../../data/task2/{pre}train.csv"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = f"../../data/task2/{pre}train_df.csv"

    suffix = ""
    if baseline:
        suffix += "_baseline"
    if augment:
        suffix += "_augmented"
    SAVE_DIR = f"../../models/task2/final/{model_name}{suffix if suffix else ''}"

    label_cols = ["sexualization", "violence", "hate"]

    print(f"Cargando todos los datos desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    if augment and not baseline:
        print("Aplicando data augmentation...")
        df = augment_df(df, aug_path="../../data/processed_train_augmented.csv")
    df["label"] = create_label_column(df, label_cols)

    print("Realizando split 80/20 de los datos...")
    stratify_col = (
        df[["sexualization", "violence", "hate"]].astype(str).agg("_".join, axis=1)
    )
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=stratify_col
    )

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
    weights_tensor = torch.sqrt(torch.tensor(n_neg / np.maximum(1, n_pos)).float())

    train_ds = Dataset.from_pandas(
        train_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    eval_ds = Dataset.from_pandas(
        val_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

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

        results["eval_f1_micro"] = round(
            f1_score(true, preds, average="micro", zero_division=0), 4
        )
        results["eval_f1_macro"] = round(
            f1_score(true, preds, average="macro", zero_division=0), 4
        )
        return results

    def make_tokenize_fn(tokenizer, cfg: ModelConfig, training: bool = False):
        def tokenize_fn(batch):
            texts = batch["text"]
            if cfg.use_pysentimiento_preprocess:
                texts = [preprocess_tweet(t, lang="es") for t in texts]
            if training:
                return tokenizer(texts, padding=False, truncation=False)
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

    model = build_bert_like_classifier(cfg, device, num_labels=3)
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
        load_best_model_at_end=True,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=None,
        is_multilabel=True,
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
        aggregation="max",  # or "mean"
    )

    logits = pred_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    true_labels = pred_output.label_ids

    base_thresholds = [0.5] * probs.shape[1]

    preds_base = (probs >= base_thresholds).astype(int)
    f1_base = f1_score(true_labels, preds_base, average="macro", zero_division=0)
    print(f"F1 Macro con threshold 0.5: {f1_base:.4f}")

    best_thrs, best_f1 = find_best_thresholds(true_labels, probs)
    print(best_thrs, best_f1, sum(best_f1) / len(best_f1))

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
        "eval_f1_macro": eval_results["eval_eval_f1_macro"]
        if "eval_eval_f1_macro" in eval_results
        else eval_results.get("eval_f1_macro", 0.0),
    }

    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nModelo guardado en: {SAVE_DIR}")
    print(f"Resultados: {results}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Entrenamiento final de un modelo basado en BERT para Task 2"
    )
    arg_parser.add_argument(
        "--model", type=str, default="BETO", help="Nombre del modelo en config"
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
        "--augment",
        action="store_true",
        help="Usar datos aumentados (solo si no es baseline)",
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
        augment=args.augment,
    )
