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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED
from trainer import WeightedTrainer

SEED = DEFAULT_SEED
set_seed(SEED)

DATA_PATH = "../../data/task1/processed_train.csv"
RESULTS_FILE = "../../results/task1/tabla_paper.csv"
SAVE_DIR = "../../models/task1/comparison"

# ─────────────────────────────────────────────────────────────
# CONFIGURACIONES POR MODELO
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuración de arquitectura y entrenamiento para un modelo."""
    model_id: str

    # --- Arquitectura del clasificador ---
    classifier_dropout: float = 0.1         # Dropout en la capa de clasificación final
    attention_probs_dropout_prob: float = 0.1  # Dropout en atención (BERT/RoBERTa)
    hidden_dropout_prob: float = 0.1        # Dropout en capas ocultas (BERT/RoBERTa)

    # --- Tokenización ---
    max_len: int = 512
    use_pysentimiento_preprocess: bool = False  # Solo para Robertuito

    # --- Entrenamiento ---
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    early_stopping_patience: int = 5

    # --- Focal Loss ---
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "DistilBETO": ModelConfig(
        model_id="dccuchile/distilbert-base-spanish-uncased",
        classifier_dropout=0.3,          # ↑ más regularización para modelo pequeño
        attention_probs_dropout_prob=0.15,
        hidden_dropout_prob=0.15,
        max_len=512,
        learning_rate=4e-5,              # ↑ tolera lr más alto
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        weight_decay=0.05,               # ↑ más regularización
    ),
    "BETO": ModelConfig(
        model_id="dccuchile/bert-base-spanish-wwm-cased",
        classifier_dropout=0.2,          # ↑ ligeramente más dropout
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=2e-5,
        warmup_ratio=0.06,
    ),
    "MarIA": ModelConfig(
        model_id="IsGarrido/roberta-base-bne",
        classifier_dropout=0.15,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=2e-5,
        warmup_ratio=0.06,
    ),
    "XLM-R": ModelConfig(
        model_id="xlm-roberta-base",
        classifier_dropout=0.15,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=1e-5,         # XLM-R es más sensible a lr altos
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.05,               # ↑ ayuda con estabilidad
    ),
    "Robertuito": ModelConfig(
        model_id="pysentimiento/robertuito-base-uncased",
        classifier_dropout=0.2,          # ↑ más regularización
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=128,                # Tweets → contexto corto
        use_pysentimiento_preprocess=True,
        learning_rate=3e-5,              # ↑ ligeramente más alto
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        warmup_ratio=0.06,              # ← añadir warmup
    ),
}

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].map({"NM": 0, "M": 1})
print(f"Dataset cargado: {len(df)} canciones")
print(f"Distribución original:\n{df['label'].value_counts()}")

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df.rename(columns={"lyrics": "text"}), preserve_index=False)
val_ds = Dataset.from_pandas(val_df.rename(columns={"lyrics": "text"}), preserve_index=False)

n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
total = n_neg + n_pos
w0 = total / (2 * n_neg)
w1 = total / (2 * n_pos)
weights_tensor = torch.tensor([w0, w1]).float()
print(f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 0: {w0:.2f}, clase 1: {w1:.2f}")

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


def make_tokenize_fn(tokenizer, cfg: ModelConfig):
    """Factoría de funciones de tokenización, evita bug de closure."""
    def tokenize_fn(batch):
        texts = batch["text"]
        if cfg.use_pysentimiento_preprocess:
            texts = [preprocess_tweet(t, lang="es") for t in texts]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=cfg.max_len)
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
    arch = type(arch_config).__name__  # e.g. "BertConfig", "RobertaConfig", "DistilBertConfig"

    if "DistilBert" in arch:
        # DistilBERT usa nombres distintos
        dropout_kwargs = {
            "seq_classif_dropout": cfg.classifier_dropout,   # Capa de clasificación final
            "dropout": cfg.hidden_dropout_prob,              # Dropout general
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
        model_id,
        num_labels=2,
        problem_type="single_label_classification",
        **dropout_kwargs,
    )

    # ── Verificación de dropouts aplicados ──────────────────────────
    cfg_loaded = model.config
    print(f"    [Dropout verificado]")
    if "DistilBert" in arch:
        print(f"      seq_classif_dropout : {getattr(cfg_loaded, 'seq_classif_dropout', 'N/A')}")
        print(f"      dropout             : {getattr(cfg_loaded, 'dropout', 'N/A')}")
        print(f"      attention_dropout   : {getattr(cfg_loaded, 'attention_dropout', 'N/A')}")
    else:
        print(f"      classifier_dropout          : {getattr(cfg_loaded, 'classifier_dropout', 'N/A')}")
        print(f"      hidden_dropout_prob         : {getattr(cfg_loaded, 'hidden_dropout_prob', 'N/A')}")
        print(f"      attention_probs_dropout_prob: {getattr(cfg_loaded, 'attention_probs_dropout_prob', 'N/A')}")
    # ────────────────────────────────────────────────────────────────

    return model.to(device)


def make_training_args(cfg: ModelConfig, checkpoints_path: str) -> TrainingArguments:
    """Construye TrainingArguments a partir de la configuración del modelo."""
    return TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=cfg.learning_rate,
        optim="adamw_torch",
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INICIANDO COMPARATIVA EN {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ---")

for name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*50}")
    print(f">>> Evaluando: {name} ({cfg.model_id})")
    print(f"    lr={cfg.learning_rate}, dropout_cls={cfg.classifier_dropout}, max_len={cfg.max_len}")
    print(f"{'='*50}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

        tokenize_fn = make_tokenize_fn(tokenizer, cfg)
        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
        val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
        train_tok = train_tok.rename_column("label", "labels")
        val_tok = val_tok.rename_column("labels" if "labels" in val_tok.column_names else "label", "labels")
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
            focal_alpha=cfg.focal_alpha,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        )

        trainer.train()
        metrics = trainer.evaluate()

        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Modelo guardado en: {model_save_path}")

        results_list.append({
            "Modelo": name,
            "F1-Macro": metrics["eval_f1_macro"],
            "Accuracy": metrics["eval_accuracy"],
            "Precision": metrics["eval_precision"],
            "Recall": metrics["eval_recall"],
        })
        print(f"✓ {name}: F1={metrics['eval_f1_macro']:.4f}")

    except Exception as e:
        print(f"✗ Error con {name}: {e}")
        import traceback; traceback.print_exc()
        results_list.append({
            "Modelo": name, "F1-Macro": None,
            "Accuracy": None, "Precision": None, "Recall": None,
        })

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
