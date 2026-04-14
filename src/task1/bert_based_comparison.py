"""
Comparativa de modelos para Task 1: Clasificación Binaria de Misoginia en Canciones.
Genera una tabla para el paper con métricas de cada modelo.
"""

import argparse
import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from bert_model_configs import MODEL_CONFIGS, ModelConfig, apply_baseline_settings
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
from augmentation_utils import LyricsAugmentor
from bert_pooling import build_bert_like_classifier, predict_with_chunks
from random_crop_collator import RandomCropDataCollator
from trainer import WeightedTrainer
from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED


SAVE_DIR = "../../models/task1/comparison"

# ─────────────────────────────────────────────────────────────
# CONFIGURACIONES POR MODELO
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
def main(augment=False, baseline=False):
    if baseline:
        apply_baseline_settings()

    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task1/{pre}train_df.csv"
    VAL_PATH = f"../../data/task1/{pre}val_df.csv"
    DEV_PATH = f"../../data/task1/{pre}dev_df.csv"

    suffix = ""
    if baseline:
        suffix += "_baseline"
    if augment:
        suffix += "_aug"

    RESULTS_FILE = (
        f"../../results/task1/tabla_paper{suffix if suffix else '_processed'}.csv"
    )

    print(f"Cargando datos de entrenamiento desde {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["label"] = train_df["label"].map({"NM": 0, "M": 1})
    print(f"Cargando datos de validación desde {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    val_df["label"] = val_df["label"].map({"NM": 0, "M": 1})
    print(f"Cargando datos de test/dev desde {DEV_PATH}...")
    dev_df = pd.read_csv(DEV_PATH)
    dev_df["label"] = dev_df["label"].map({"NM": 0, "M": 1})

    # Compute class weights from original training samples only
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
    print(
        f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 0: {w0:.2f}, clase 1: {w1:.2f}"
    )
    # Using beto tokenizer WITHOUT truncation to get real token length stats on original training samples
    tokenizer_beto = AutoTokenizer.from_pretrained(MODEL_CONFIGS["BETO"].model_id)
    train_originals_texts = (
        train_df[train_df["augmentation"] == "original"]["lyrics"].tolist()
        if "augmentation" in train_df.columns
        else train_df["lyrics"].tolist()
    )
    tokenized_lengths = tokenizer_beto(
        train_originals_texts,
        padding=False,
        truncation=False,
    )
    tokenized_lengths = [len(t) for t in tokenized_lengths["input_ids"]]
    print(
        f"Tokenized length stats (BETO, no truncation, original train samples): "
        f"mean={np.mean(tokenized_lengths):.1f}, std={np.std(tokenized_lengths):.1f}, "
        f"median={int(np.median(tokenized_lengths))}, "
        f"p95={int(np.percentile(tokenized_lengths, 95))}, "
        f"max={max(tokenized_lengths)}"
    )

    augmentor = LyricsAugmentor()
    if augment:
        train_df = augmentor.augment_dataframe(
            train_df,
            multiplier=2,  # Genera 2 versiones nuevas por cada canción de odio original
        )

    train_ds = Dataset.from_pandas(
        train_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    val_ds = Dataset.from_pandas(
        val_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    dev_ds = Dataset.from_pandas(
        dev_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

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

    def make_tokenize_fn(tokenizer, cfg: ModelConfig, training: bool = False):
        """Tokenization for training/eval. Training keeps full sequences for dynamic random crop."""

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

    def load_model_with_config(model_id: str, cfg: ModelConfig, device: torch.device):
        model = build_bert_like_classifier(cfg, device, num_labels=2)
        print(
            "    Pooling="
            f"{getattr(model.config, 'pooling_strategy', 'cls')} | "
            f"dropout_cls={cfg.classifier_dropout}"
        )
        return model

    def make_training_args(
        cfg: ModelConfig, checkpoints_path: str
    ) -> TrainingArguments:
        """Construye TrainingArguments a partir de la configuración del modelo."""
        return TrainingArguments(
            output_dir=checkpoints_path,
            learning_rate=cfg.learning_rate,
            optim=cfg.optim,
            data_seed=SEED,
            seed=SEED,
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
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────

    results_list = []
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"--- INICIANDO COMPARATIVA EN {device.type.upper()} ---")

    for name, cfg in MODEL_CONFIGS.items():
        set_seed(SEED)
        print(f"\n{'=' * 50}")
        print(f">>> Evaluando: {name} ({cfg.model_id})")
        print(
            f"    lr={cfg.learning_rate}, dropout_cls={cfg.classifier_dropout}, max_len={cfg.max_len}"
        )
        print(f"{'=' * 50}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

            train_tokenize_fn = make_tokenize_fn(tokenizer, cfg, training=True)
            eval_tokenize_fn = make_tokenize_fn(tokenizer, cfg, training=False)
            train_tok = train_ds.map(
                train_tokenize_fn,
                batched=True,
                remove_columns=["text"],
                load_from_cache_file=False,
            )
            val_tok = val_ds.map(
                eval_tokenize_fn,
                batched=True,
                remove_columns=["text"],
                load_from_cache_file=False,
            )
            dev_tok = dev_ds.map(
                eval_tokenize_fn,
                batched=True,
                remove_columns=["text"],
                load_from_cache_file=False,
            )
            train_tok = train_tok.rename_column("label", "labels")
            val_tok = val_tok.rename_column(
                "labels" if "labels" in val_tok.column_names else "label", "labels"
            )
            dev_tok = dev_tok.rename_column(
                "labels" if "labels" in dev_tok.column_names else "label", "labels"
            )
            train_tok.set_format("torch")
            val_tok.set_format("torch")
            dev_tok.set_format("torch")

            model = load_model_with_config(cfg.model_id, cfg, device)

            model_save_path = os.path.join(SAVE_DIR, name + suffix)
            checkpoints_path = os.path.join(model_save_path, "checkpoints")

            args = make_training_args(cfg, checkpoints_path)

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
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg.early_stopping_patience
                    )
                ],
            )

            trainer.train()
            # Standard evaluation (argmax / threshold=0.5)
            metrics = trainer.evaluate()

            # ─────────────────────────────────────────
            # Collect validation probabilities (on Dev Set)
            # ─────────────────────────────────────────
            pred_output = predict_with_chunks(
                dataset=dev_ds,
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
            probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            true_labels = pred_output.label_ids

            # ─────────────────────────────────────────
            # Threshold sweep on validation
            # ─────────────────────────────────────────
            # best_thr, best_f1 = find_best_threshold(true_labels, probs)

            # print(f"    🔎 Mejor threshold validación: {best_thr}")
            # print(f"    🔎 Macro-F1 con threshold óptimo: {best_f1}")

            # Recompute metrics using optimal threshold
            best_thr = 0.5
            opt_preds = (probs >= best_thr).astype(int)

            precision, recall, f1_opt, _ = precision_recall_fscore_support(
                true_labels, opt_preds, average="macro", zero_division=0.0
            )
            acc_opt = accuracy_score(true_labels, opt_preds)

            os.makedirs(model_save_path, exist_ok=True)
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"Modelo guardado en: {model_save_path}")

            results_list.append(
                {
                    "Modelo": name,
                    "F1-Macro": f1_opt,
                    "Accuracy": acc_opt,
                    "Precision": precision,
                    "Recall": recall,
                    "Best-Threshold": best_thr,
                }
            )
            print(f"✓ {name}: F1={f1_opt:.4f}")

        except Exception as e:
            print(f"✗ Error con {name}: {e}")
            import traceback

            traceback.print_exc()
            results_list.append(
                {
                    "Modelo": name,
                    "F1-Macro": None,
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                }
            )

        finally:
            for var in [
                "trainer",
                "model",
                "tokenizer",
                "train_tok",
                "val_tok",
                "dev_tok",
            ]:
                if var in locals():
                    del locals()[var]
            gc.collect()
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────
    # GUARDAR RESULTADOS
    # ─────────────────────────────────────────────────────────────

    df_res = pd.DataFrame(results_list).sort_values("F1-Macro", ascending=False)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    df_res.to_csv(RESULTS_FILE, index=False)

    print(f"\n{'=' * 60}")
    print("RESULTADOS COMPARATIVA")
    print(f"{'=' * 60}")
    print(df_res.to_markdown(index=False))
    print(f"\nGuardado en: {RESULTS_FILE}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Comparativa de  basados en BERT para Task 1"
    )
    arg_parser.add_argument(
        "--augment", action="store_true", help="Activar augmentación de datos"
    )
    arg_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usar configuraciones baseline y datos crudos",
    )
    args = arg_parser.parse_args()
    main(augment=args.augment, baseline=args.baseline)
