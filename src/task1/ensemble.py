"""
Ensemble K-Fold: Entrena los 3 mejores modelos con Cross-Validation
para después hacer stacking con meta-modelo.
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoConfig,
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED
from trainer import WeightedTrainer

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
TRAIN_FILE = "../../data/task1/processed_train.csv"
SAVE_DIR = "../../models/task1/ensemble"
N_FOLDS = 5

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR MODELO (top 3 de la comparativa)
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_id: str
    classifier_dropout: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    max_len: int = 512
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    early_stopping_patience: int = 5
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None


# Top 3 modelos según tabla_paper_test.csv
MODELS_TO_TRAIN: dict[str, ModelConfig] = {
    "DistilBETO": ModelConfig(
        model_id="dccuchile/distilbert-base-spanish-uncased",
        classifier_dropout=0.3,
        attention_probs_dropout_prob=0.15,
        hidden_dropout_prob=0.15,
        learning_rate=4e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        weight_decay=0.05,
    ),
    "XLM-R": ModelConfig(
        model_id="xlm-roberta-base",
        classifier_dropout=0.15,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.05,
    ),
    "MarIA": ModelConfig(
        model_id="IsGarrido/roberta-base-bne",
        classifier_dropout=0.15,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        learning_rate=2e-5,
        warmup_ratio=0.06,
    ),
}

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

print(f"Cargando datos desde {TRAIN_FILE}...")
df = pd.read_csv(TRAIN_FILE)
df["label"] = df["label"].map({"NM": 0, "M": 1}) if df["label"].dtype == object else df["label"]
texts = df["lyrics"].tolist()
labels = df["label"].tolist()

n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
total = n_neg + n_pos
w0 = total / (2 * n_neg)
w1 = total / (2 * n_pos)
weights_tensor = torch.tensor([w0, w1]).float()

print(f"Dataset: {len(df)} canciones | Neg={n_neg}, Pos={n_pos}")
print(f"Pesos: clase 0={w0:.2f}, clase 1={w1:.2f}")

# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────

def load_model_with_config(model_id: str, cfg: ModelConfig, device: torch.device):
    """Carga modelo con dropouts según arquitectura."""
    arch_config = AutoConfig.from_pretrained(model_id)
    arch = type(arch_config).__name__

    if "DistilBert" in arch:
        dropout_kwargs = {
            "seq_classif_dropout": cfg.classifier_dropout,
            "dropout": cfg.hidden_dropout_prob,
            "attention_dropout": cfg.attention_probs_dropout_prob,
        }
    else:
        dropout_kwargs = {
            "classifier_dropout": cfg.classifier_dropout,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        problem_type="single_label_classification",
        **dropout_kwargs,
    )
    return model.to(device)


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


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL: K-FOLD POR MODELO
# ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- DISPOSITIVO: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ---")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for model_name, cfg in MODELS_TO_TRAIN.items():
    print(f"\n{'='*60}")
    print(f">>> {model_name} ({cfg.model_id})")
    print(f"{'='*60}")

    model_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n   --- Fold {fold+1}/{N_FOLDS} ---")
        set_seed(SEED)  # Reset seed cada fold para reproducibilidad

        # Crear datasets
        train_df_fold = pd.DataFrame({
            "text": [texts[i] for i in train_idx],
            "label": [labels[i] for i in train_idx],
        })
        val_df_fold = pd.DataFrame({
            "text": [texts[i] for i in val_idx],
            "label": [labels[i] for i in val_idx],
        })

        train_ds = Dataset.from_pandas(train_df_fold, preserve_index=False)
        val_ds = Dataset.from_pandas(val_df_fold, preserve_index=False)

        # Tokenizar
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=cfg.max_len,
            )

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False)
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False)

        # Cargar modelo
        model = load_model_with_config(cfg.model_id, cfg, device)

        # Training args
        checkpoints_path = f"{SAVE_DIR}/temp_checkpoints/{model_name}_fold{fold}"
        args = TrainingArguments(
            output_dir=checkpoints_path,
            learning_rate=cfg.learning_rate,
            optim="adamw_torch",
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
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

        # Entrenar
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            class_weights=weights_tensor.to(device),
            loss_type=cfg.loss_type,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        )

        trainer.train()

        # Evaluar
        eval_result = trainer.evaluate()
        fold_f1 = eval_result["eval_eval_f1_macro"]
        model_f1_scores.append(fold_f1)
        print(f"   Fold {fold+1} F1-Macro: {fold_f1:.4f}")

        # Guardar modelo y tokenizer
        save_path = f"{SAVE_DIR}/{model_name}/fold_{fold}"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

        # Limpiar memoria
        del model, trainer, tokenizer
        torch.cuda.empty_cache()

    # Resumen por modelo
    mean_f1 = np.mean(model_f1_scores)
    std_f1 = np.std(model_f1_scores)
    print(f"\n   >>> {model_name}: F1-Macro = {mean_f1:.4f} ± {std_f1:.4f}")
    fold_results.append({
        "Modelo": model_name,
        "F1-Macro (mean)": round(mean_f1, 4),
        "F1-Macro (std)": round(std_f1, 4),
        "Folds": model_f1_scores,
    })

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESUMEN K-FOLD ENSEMBLE")
print(f"{'='*60}")
for r in fold_results:
    print(f"  {r['Modelo']}: {r['F1-Macro (mean)']:.4f} ± {r['F1-Macro (std)']:.4f}")

print(f"\nModelos guardados en: {SAVE_DIR}/")
print("¡Entrenamiento completado! Ejecutar ensemble_stacking.py para el meta-modelo.")