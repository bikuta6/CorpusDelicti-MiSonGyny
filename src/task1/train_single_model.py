"""
Script de entrenamiento individual para un modelo específico.
Permite elegir el modelo y activar/desactivar augmentación mediante argumentos.

Uso:
    python train_single_model.py --model BETO --augment
    python train_single_model.py --model MarIA --no-augment
    python train_single_model.py --model LongFormer --augment --multiplier 3
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS, ModelConfig

from augmentation_utils import LyricsAugmentor
from bert_pooling import build_bert_like_classifier
from trainer import WeightedTrainer
from utils import DEFAULT_SEED, set_seed

# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────


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


def make_tokenize_fn(tokenizer, cfg: ModelConfig):
    """Tokenization with smart head+tail truncation (75% head, 25% tail)."""

    def tokenize_fn(batch):
        texts = batch["text"]

        if cfg.use_pysentimiento_preprocess:
            texts = [preprocess_tweet(t, lang="es") for t in texts]

        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

        max_len = cfg.max_len
        input_ids = []
        attention_mask = []

        for ids in tokenized["input_ids"]:
            if len(ids) <= max_len:
                pad_len = max_len - len(ids)
                padded = ids + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            else:
                cls_token = ids[0]
                sep_token = ids[-1]
                content = ids[1:-1]

                head_len = int((max_len - 2) * 0.75)
                tail_len = (max_len - 2) - head_len

                truncated = (
                    [cls_token] + content[:head_len] + content[-tail_len:] + [sep_token]
                )

                padded = truncated
                mask = [1] * max_len

            input_ids.append(padded)
            attention_mask.append(mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return tokenize_fn


def load_model_with_config(model_id: str, cfg: ModelConfig, device: torch.device):
    model = build_bert_like_classifier(cfg, device)
    print(f"    Pooling: {getattr(model.config, 'pooling_strategy', 'cls')}")
    return model


def make_training_args(cfg: ModelConfig, checkpoints_path: str) -> TrainingArguments:
    """Construye TrainingArguments a partir de la configuración del modelo."""
    return TrainingArguments(
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


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo individual con opción de augmentación"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Nombre del modelo a entrenar (debe estar en MODEL_CONFIGS)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Activar augmentación de datos para la clase minoritaria",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Desactivar augmentación de datos (default)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="Factor de augmentación (cuántas versiones nuevas por muestra minoritaria)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../data/task1/processed_train.csv",
        help="Ruta al CSV de entrenamiento",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="../../models/task1/single",
        help="Directorio base para guardar el modelo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed para reproducibilidad (default: {DEFAULT_SEED})",
    )

    parser.set_defaults(augment=False)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model
    cfg = MODEL_CONFIGS[model_name]

    print(f"\n{'=' * 60}")
    print(f"ENTRENAMIENTO: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Modelo ID: {cfg.model_id}")
    print(f"  Max length: {cfg.max_len}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Loss type: {cfg.loss_type}")
    print(f"  Augmentación: {'ACTIVADA' if args.augment else 'DESACTIVADA'}")
    if args.augment:
        print(f"  Multiplier: {args.multiplier}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    # ─────────────────────────────────────────────────────────────
    # CARGA DE DATOS
    # ─────────────────────────────────────────────────────────────

    print(f"Cargando datos desde {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df["label"] = df["label"].map({"NM": 0, "M": 1})
    print(f"Dataset cargado: {len(df)} canciones")
    print(f"Distribución original:\n{df['label'].value_counts()}")

    # Anti-leakage split
    if "augmentation" in df.columns:
        originals_df = df[df["augmentation"] == "original"].copy()
        augmented_df = df[df["augmentation"] != "original"].copy()
        print(
            f"[Anti-leakage] Originals: {len(originals_df)} | Augmented: {len(augmented_df)}"
        )

        id_to_label = originals_df.drop_duplicates("id").set_index("id")["label"]
        unique_ids = id_to_label.index.to_numpy()
        stratify_labels = id_to_label.loc[unique_ids].to_numpy()

        train_ids, val_ids = train_test_split(
            unique_ids, test_size=0.2, random_state=args.seed, stratify=stratify_labels
        )
        train_ids_set = set(train_ids)
        val_ids_set = set(val_ids)

        val_df = originals_df[originals_df["id"].isin(val_ids_set)].copy()

        train_originals = originals_df[originals_df["id"].isin(train_ids_set)]
        train_augmented = augmented_df[augmented_df["id"].isin(train_ids_set)]
        train_df = pd.concat(
            [train_originals, train_augmented], ignore_index=True
        ).sample(frac=1, random_state=args.seed)

        print(
            f"  Train size (originals + augmented): {len(train_df)} | Val size (originals only): {len(val_df)}"
        )
        print(
            f"  Train augmentation distribution:\n{train_df['augmentation'].value_counts()}"
        )

        leaked = train_df[
            (train_df["augmentation"] != "original")
            & (train_df["id"].isin(val_ids_set))
        ]
        print(f"  Augmented samples with val ID in train: {len(leaked)} (should be 0)")
    else:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=args.seed, stratify=df["label"]
        )

    # Compute class weights from original training samples only
    if "augmentation" in df.columns:
        train_originals_labels = train_df[train_df["augmentation"] == "original"][
            "label"
        ]
    else:
        train_originals_labels = train_df["label"]

    n_pos = (train_originals_labels == 1).sum()
    n_neg = (train_originals_labels == 0).sum()
    total = n_neg + n_pos
    w0 = total / (2 * n_neg)
    w1 = total / (2 * n_pos)
    weights_tensor = torch.tensor([w0, w1]).float().to(device)
    print(f"Class weights: Neg={n_neg}, Pos={n_pos} -> w0={w0:.2f}, w1={w1:.2f}")

    # Token length stats
    tokenizer_temp = AutoTokenizer.from_pretrained(cfg.model_id)
    if "augmentation" in df.columns:
        train_originals_texts = train_df[train_df["augmentation"] == "original"][
            "lyrics"
        ].tolist()
    else:
        train_originals_texts = train_df["lyrics"].tolist()

    tokenized_lengths = tokenizer_temp(
        train_originals_texts,
        padding=False,
        truncation=False,
    )
    tokenized_lengths = [len(t) for t in tokenized_lengths["input_ids"]]
    print(
        f"Token length stats (original train samples): "
        f"mean={np.mean(tokenized_lengths):.1f}, "
        f"median={int(np.median(tokenized_lengths))}, "
        f"p95={int(np.percentile(tokenized_lengths, 95))}, "
        f"max={max(tokenized_lengths)}"
    )

    # ─────────────────────────────────────────────────────────────
    # AUGMENTACIÓN (si está activada)
    # ─────────────────────────────────────────────────────────────

    if args.augment:
        print(f"\n>>> Aplicando augmentación con multiplier={args.multiplier}...")
        augmentor = LyricsAugmentor()
        train_df = augmentor.augment_dataframe(
            train_df, minority_label=1, multiplier=args.multiplier
        )
        print(f"Train size después de augmentación: {len(train_df)}")
        print(f"Distribución final:\n{train_df['label'].value_counts()}")
    else:
        print("\n>>> Augmentación DESACTIVADA. Usando datos originales.")

    # ─────────────────────────────────────────────────────────────
    # PREPARAR DATASETS
    # ─────────────────────────────────────────────────────────────

    train_ds = Dataset.from_pandas(
        train_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    val_ds = Dataset.from_pandas(
        val_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenize_fn = make_tokenize_fn(tokenizer, cfg)

    train_tok = train_ds.map(
        tokenize_fn, batched=True, remove_columns=["text"]
    ).rename_column("label", "labels")
    val_tok = val_ds.map(
        tokenize_fn, batched=True, remove_columns=["text"]
    ).rename_column("label", "labels")

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # ─────────────────────────────────────────────────────────────
    # CARGAR MODELO
    # ─────────────────────────────────────────────────────────────

    print(f"\nCargando modelo: {cfg.model_id}")
    model = load_model_with_config(cfg.model_id, cfg, device)

    # ─────────────────────────────────────────────────────────────
    # TRAINING ARGS
    # ─────────────────────────────────────────────────────────────

    aug_suffix = f"_aug{args.multiplier}x" if args.augment else "_no_aug"
    model_save_dir = os.path.join(args.save_dir, f"{model_name}{aug_suffix}")
    checkpoints_dir = os.path.join(model_save_dir, "checkpoints")

    training_args = make_training_args(cfg, checkpoints_dir)

    # ─────────────────────────────────────────────────────────────
    # TRAINER
    # ─────────────────────────────────────────────────────────────

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        ],
        class_weights=weights_tensor,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
    )

    # ─────────────────────────────────────────────────────────────
    # ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────

    print("\n>>> Iniciando entrenamiento...")
    trainer.train()

    # ─────────────────────────────────────────────────────────────
    # EVALUACIÓN FINAL Y THRESHOLD OPTIMIZATION
    # ─────────────────────────────────────────────────────────────

    print("\n>>> Evaluación en validación...")
    predictions = trainer.predict(val_tok)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    true_labels = predictions.label_ids

    best_thr, best_f1 = find_best_threshold(true_labels, probs)
    print(f"  Best threshold: {best_thr} → F1-Macro: {best_f1}")

    # ─────────────────────────────────────────────────────────────
    # GUARDAR MODELO Y RESULTADOS
    # ─────────────────────────────────────────────────────────────

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    # Guardar threshold
    threshold_file = os.path.join(model_save_dir, "best_threshold.txt")
    with open(threshold_file, "w") as f:
        f.write(f"{best_thr}\n")
        f.write(f"# F1-Macro: {best_f1}\n")

    # Guardar métricas
    metrics_file = os.path.join(model_save_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Augmentation: {'Yes' if args.augment else 'No'}\n")
        if args.augment:
            f.write(f"Multiplier: {args.multiplier}\n")
        f.write(f"Best Threshold: {best_thr}\n")
        f.write(f"Best F1-Macro: {best_f1}\n")
        f.write(f"\nFinal eval metrics:\n")
        for k, v in predictions.metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n{'=' * 60}")
    print(f"✓ Modelo guardado en: {model_save_dir}")
    print(f"✓ Threshold guardado en: {threshold_file}")
    print(f"✓ Métricas guardadas en: {metrics_file}")
    print(f"{'=' * 60}\n")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
