"""
Comparativa de modelos para Task 1: Clasificación Binaria de Misoginia en Canciones.
Genera una tabla para el paper con métricas de cada modelo.
"""

import os
import sys
import gc
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    AutoConfig,
)
from bert_model_configs import ModelConfig, MODEL_CONFIGS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_seed, DEFAULT_SEED
from trainer import WeightedTrainer

SEED = DEFAULT_SEED
set_seed(SEED)

DATA_PATH = "../../data/task1/augmented_processed_train.csv"
RESULTS_FILE = "../../results/task1/tabla_paper.csv"
SAVE_DIR = "../../models/task1/comparison"

# ─────────────────────────────────────────────────────────────
# CONFIGURACIONES POR MODELO
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].map({"NM": 0, "M": 1})
print(f"Dataset cargado: {len(df)} canciones")
print(f"Distribución original:\n{df['label'].value_counts()}")

if "augmentation" in df.columns:
    originals_df = df[df["augmentation"] == "original"].copy()
    augmented_df = df[df["augmentation"] != "original"].copy()
    print(f"[Anti-leakage] Originals: {len(originals_df)} | Augmented: {len(augmented_df)}")

    # Build a Series mapping unique_id -> label for stratification
    # Augmented samples share the same ID as their originals
    id_to_label = originals_df.drop_duplicates("id").set_index("id")["label"]
    unique_ids = id_to_label.index.to_numpy()
    stratify_labels = id_to_label.loc[unique_ids].to_numpy()

    train_ids, val_ids = train_test_split(
        unique_ids, test_size=0.2, random_state=SEED,
        stratify=stratify_labels
    )
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    # Validation: only original samples whose ID is in val_ids
    val_df = originals_df[originals_df["id"].isin(val_ids_set)].copy()

    # Train: original samples with train IDs + augmented whose ID is in train_ids only
    train_originals = originals_df[originals_df["id"].isin(train_ids_set)]
    train_augmented = augmented_df[augmented_df["id"].isin(train_ids_set)]
    train_df = pd.concat([train_originals, train_augmented], ignore_index=True).sample(
        frac=1, random_state=SEED
    )

    print(f"  Train size (originals + augmented): {len(train_df)} | Val size (originals only): {len(val_df)}")
    print(f"  Train augmentation distribution:\n{train_df['augmentation'].value_counts()}")
    leaked = train_df[(train_df["augmentation"] != "original") & (train_df["id"].isin(val_ids_set))]
    print(f"  Augmented samples with val ID in train: {len(leaked)} (should be 0)")
else:
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )

# Compute class weights from original training samples only
train_originals_labels = train_df[train_df["augmentation"] == "original"]["label"] if "augmentation" in df.columns else train_df["label"]
n_pos = (train_originals_labels == 1).sum()
n_neg = (train_originals_labels == 0).sum()
total = n_neg + n_pos
w0 = total / (2 * n_neg)
w1 = total / (2 * n_pos)
weights_tensor = torch.tensor([w0, w1]).float()
print(f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 0: {w0:.2f}, clase 1: {w1:.2f}")

train_ds = Dataset.from_pandas(
    train_df.rename(columns={"lyrics": "text"}), preserve_index=False
)
val_ds = Dataset.from_pandas(
    val_df.rename(columns={"lyrics": "text"}), preserve_index=False
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


def make_tokenize_fn(tokenizer, cfg: ModelConfig):
    """Factoría de funciones de tokenización, evita bug de closure."""

    def tokenize_fn(batch):
        texts = batch["text"]
        if cfg.use_pysentimiento_preprocess:
            texts = [preprocess_tweet(t, lang="es") for t in texts]
        return tokenizer(
            texts, padding="max_length", truncation=True, max_length=cfg.max_len
        )

    return tokenize_fn


def load_model_with_config(model_id: str, cfg: ModelConfig, device: torch.device):
    """
    Carga el modelo aplicando los dropouts de la configuración.
    Mapea los kwargs según la arquitectura:
      - BERT/RoBERTa: classifier_dropout, hidden_dropout_prob, attention_probs_dropout_prob
      - DistilBERT:   seq_classif_dropout, dropout, attention_dropout
    """

    # Detectar arquitectura antes de cargar el modelo completo
    arch_config = AutoConfig.from_pretrained(model_id)
    arch = type(
        arch_config
    ).__name__  # e.g. "BertConfig", "RobertaConfig", "DistilBertConfig"

    if "DistilBert" in arch:
        # DistilBERT usa nombres distintos
        dropout_kwargs = {
            "seq_classif_dropout": cfg.classifier_dropout,  # Capa de clasificación final
            "dropout": cfg.hidden_dropout_prob,  # Dropout general
            "attention_dropout": cfg.attention_probs_dropout_prob,  # Dropout en atención
        }
    else:
        # BERT, RoBERTa, XLM-R, MarIA, Robertuito
        dropout_kwargs = {
            "classifier_dropout": cfg.classifier_dropout,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
        }

    print(f"    Arquitectura detectada: {arch} → kwargs: {list(dropout_kwargs.keys())}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=2,
        ignore_mismatched_sizes=cfg.ignore_mismatched_sizes,
        **dropout_kwargs,
    )

    # ── Verificación de dropouts aplicados ──────────────────────────
    cfg_loaded = model.config
    print(f"    [Dropout verificado]")
    if "DistilBert" in arch:
        print(
            f"      seq_classif_dropout : {getattr(cfg_loaded, 'seq_classif_dropout', 'N/A')}"
        )
        print(f"      dropout             : {getattr(cfg_loaded, 'dropout', 'N/A')}")
        print(
            f"      attention_dropout   : {getattr(cfg_loaded, 'attention_dropout', 'N/A')}"
        )
    else:
        print(
            f"      classifier_dropout          : {getattr(cfg_loaded, 'classifier_dropout', 'N/A')}"
        )
        print(
            f"      hidden_dropout_prob         : {getattr(cfg_loaded, 'hidden_dropout_prob', 'N/A')}"
        )
        print(
            f"      attention_probs_dropout_prob: {getattr(cfg_loaded, 'attention_probs_dropout_prob', 'N/A')}"
        )
    # ────────────────────────────────────────────────────────────────

    return model.to(device)


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
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────

results_list = []
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"--- INICIANDO COMPARATIVA EN {device.type.upper()} ---")

for name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*50}")
    print(f">>> Evaluando: {name} ({cfg.model_id})")
    print(
        f"    lr={cfg.learning_rate}, dropout_cls={cfg.classifier_dropout}, max_len={cfg.max_len}"
    )
    print(f"{'='*50}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

        tokenize_fn = make_tokenize_fn(tokenizer, cfg)
        train_tok = train_ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=False,
        )
        val_tok = val_ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=False,
        )
        train_tok = train_tok.rename_column("label", "labels")
        val_tok = val_tok.rename_column(
            "labels" if "labels" in val_tok.column_names else "label", "labels"
        )
        train_tok.set_format("torch")
        val_tok.set_format("torch")

        model = load_model_with_config(cfg.model_id, cfg, device)

        model_save_path = os.path.join(SAVE_DIR, name)
        checkpoints_path = os.path.join(model_save_path, "checkpoints")

        args = make_training_args(cfg, checkpoints_path)

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
        # Collect validation probabilities
        # ─────────────────────────────────────────
        pred_output = trainer.predict(val_tok)

        logits = pred_output.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        true_labels = pred_output.label_ids

        # ─────────────────────────────────────────
        # Threshold sweep on validation
        # ─────────────────────────────────────────
        best_thr, best_f1 = find_best_threshold(true_labels, probs)

        print(f"    🔎 Mejor threshold validación: {best_thr}")
        print(f"    🔎 Macro-F1 con threshold óptimo: {best_f1}")

        # Recompute metrics using optimal threshold
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
        for var in ["trainer", "model", "tokenizer", "train_tok", "val_tok"]:
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

print(f"\n{'='*60}")
print("RESULTADOS COMPARATIVA")
print(f"{'='*60}")
print(df_res.to_markdown(index=False))
print(f"\nGuardado en: {RESULTS_FILE}")
