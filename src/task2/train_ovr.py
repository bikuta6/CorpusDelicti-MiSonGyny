"""
Entrenamiento y evaluación para modelos One-Vs-Rest (OVR) en Task 2.
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


def create_label_column(df: pd.DataFrame, label: str) -> pd.Series:
    return df[label].astype(float).values.reshape(-1, 1).tolist()


def train_single_label(model_name: str, label: str, baseline: bool = False, augment: bool = False):
    if baseline:
        apply_baseline_settings()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"El modelo '{model_name}' no existe en las configuraciones. Opciones: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_name]

    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task2/{pre}train_df.csv"
    VAL_PATH = f"../../data/task2/{pre}val_df.csv"
    DEV_PATH = f"../../data/task2/{pre}dev_df.csv"

    suffix = ""
    if baseline:
        suffix += "_baseline"
    if augment:
        suffix += "_augmented"
    SAVE_DIR = f"../../models/task2/OVR/{model_name}{suffix}_{label.capitalize()}"

    print(f"\n--- INICIANDO ENTRENAMIENTO PARA {model_name} - {label.upper()} ---")

    print(f"Cargando datos de entrenamiento desde {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    if augment and not baseline:
        print("Aplicando data augmentation...")
        train_df = augment_df(train_df, aug_path="../../data/processed_train_augmented.csv")
    train_df["label"] = create_label_column(train_df, label)
    print(f"Cargando datos de validación desde {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    val_df["label"] = create_label_column(val_df, label)
    print(f"Cargando datos de test/dev desde {DEV_PATH}...")
    dev_df = pd.read_csv(DEV_PATH)
    dev_df["label"] = create_label_column(dev_df, label)

    train_originals_labels = (
        train_df[train_df["augmentation"] == "original"]["label"]
        if "augmentation" in train_df.columns
        else train_df["label"]
    )
    n_samples = len(train_originals_labels)
    n_pos = sum([x[0] for x in train_originals_labels])
    n_neg = n_samples - n_pos
    weights_tensor = torch.tensor([float(np.sqrt(n_neg / max(1, n_pos)))]).float()

    train_ds = Dataset.from_pandas(
        train_df[["lyrics", "label"]].rename(columns={"lyrics": "text"}),
        preserve_index=False,
    )
    val_ds = Dataset.from_pandas(
        val_df[["lyrics", "label"]].rename(columns={"lyrics": "text"}),
        preserve_index=False,
    )
    dev_ds = Dataset.from_pandas(
        dev_df[["lyrics", "label"]].rename(columns={"lyrics": "text"}),
        preserve_index=False,
    )

    def compute_metrics(pred):
        logits = pred.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        true = pred.label_ids

        p = precision_score(true, preds, zero_division=0)
        r = recall_score(true, preds, zero_division=0)
        f1 = f1_score(true, preds, zero_division=0)
        acc = accuracy_score(true, preds)

        return {
            "eval_precision": round(p, 4),
            "eval_recall": round(r, 4),
            "eval_f1_macro": round(
                f1, 4
            ),  # Using f1 as macro to reuse the same early stopping metric
            "eval_accuracy": round(acc, 4),
        }

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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    train_tokenize_fn = make_tokenize_fn(tokenizer, cfg, training=True)
    eval_tokenize_fn = make_tokenize_fn(tokenizer, cfg, training=False)

    train_tok = train_ds.map(train_tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(eval_tokenize_fn, batched=True, remove_columns=["text"])
    dev_tok = dev_ds.map(eval_tokenize_fn, batched=True, remove_columns=["text"])

    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")
    dev_tok = dev_tok.rename_column("label", "labels")

    train_tok.set_format("torch")
    val_tok.set_format("torch")
    dev_tok.set_format("torch")

    model = build_bert_like_classifier(cfg, device, num_labels=1)
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

    train_collator = RandomCropDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.max_len,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=None,
        is_multilabel=True,  # Actually binary classification with BCEWithLogitsLoss requires is_multilabel=True behavior in WeightedTrainer usually or custom handling
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        ],
    )

    print("Entrenando...")
    trainer.train()

    eval_logs = [l for l in trainer.state.log_history if "eval_f1_macro" in l]
    if eval_logs:
        best_eval = max(eval_logs, key=lambda x: x["eval_f1_macro"])
        best_epoch = best_eval["epoch"]
    else:
        best_epoch = None
    print(f"    ✓ Best epoch: {best_epoch}")

    print("Evaluando en Dev Set...")
    pred_output = predict_with_chunks(
        dataset=dev_ds,
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
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    true_labels = pred_output.label_ids

    best_thr = 0.5
    opt_preds = (probs >= best_thr).astype(int)

    precision, recall, f1_opt, _ = precision_recall_fscore_support(
        true_labels, opt_preds, average="binary", zero_division=0.0
    )
    acc_opt = accuracy_score(true_labels, opt_preds)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    results = {
        "Modelo": model_name,
        "Label": label,
        "F1": f1_opt,
        "Accuracy": acc_opt,
        "Precision": precision,
        "Recall": recall,
        "Best-Threshold": best_thr,
    }

    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nModelo guardado en: {SAVE_DIR}")
    print(f"Resultados: {results}")

    # Clean up memory
    del model, trainer, train_tok, val_tok, dev_tok
    torch.cuda.empty_cache()
    gc.collect()

    return f1_opt


def main():
    arg_parser = argparse.ArgumentParser(
        description="Entrenamiento OVR (One-Vs-Rest) basado en BERT para Task 2"
    )
    arg_parser.add_argument(
        "--model", type=str, default="BETO", help="Nombre del modelo en config"
    )
    arg_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usar configuraciones baseline y datos crudos",
    )
    arg_parser.add_argument(
        "--augment",
        action="store_true",
        help="Usar datos aumentados (solo si no es baseline)",
    )
    arg_parser.add_argument(
        "--label",
        type=str,
        default="all",
        choices=["sexualization", "violence", "hate", "all"],
        help="Etiqueta específica a entrenar. Usa 'all' para entrenar las tres.",
    )
    args = arg_parser.parse_args()

    labels_to_train = (
        ["sexualization", "violence", "hate"] if args.label == "all" else [args.label]
    )

    f1_scores = {}
    for label in labels_to_train:
        f1 = train_single_label(
            model_name=args.model, label=label, baseline=args.baseline, augment=args.augment
        )
        f1_scores[label] = f1

    if args.label == "all":
        macro_f1 = sum(f1_scores.values()) / len(f1_scores)
        print(f"\n==================================================")
        print(f"RESUMEN GLOBAL OVR - MODELO: {args.model}")
        print(f"==================================================")
        for label, score in f1_scores.items():
            print(f"  - {label.capitalize()}: {score:.4f}")
        print(f"  >> Macro F1 Global: {macro_f1:.4f}")
        print(f"==================================================")

        # Save global summary
        suffix = ""
        if args.baseline:
            suffix += "_baseline"
        if args.augment:
            suffix += "_augmented"
        global_results_dir = f"../../models/task2/OVR/{args.model}{suffix}_Global"
        os.makedirs(global_results_dir, exist_ok=True)

        summary = {
            "Modelo": args.model,
            "Baseline": args.baseline,
            "Augment": args.augment,
            "Individual_F1": f1_scores,
            "Macro_F1_Global": macro_f1,
        }

        with open(os.path.join(global_results_dir, "global_results.json"), "w") as f:
            json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
