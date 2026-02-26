"""
Comparativa de modelos para Task 1: Clasificación Binaria de Misoginia en Canciones.
Genera una tabla para el paper con métricas de cada modelo.
"""

import os
import sys
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback
)

# Añadimos la carpeta padre al path para poder importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED
from trainer import WeightedTrainer  # Importamos el trainer personalizado con pesos de clase

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
DATA_PATH = "../../data/task1/train.csv"
OUTPUT_DIR = "../../models/task1/comparativa"
RESULTS_FILE = "../../results/task1/tabla_paper.csv"
SAVE_DIR = "../../models/task1/comparison"

# Estadísticas del dataset:
# - Mediana: 392 tokens, Percentil 90: 970 tokens
# - Clase positiva más larga: mediana 683 tokens
MAX_LEN = 512  # Captura ~60% del contenido vs 30% con 256

# Modelos a comparar para el Paper
MODELS = {
    "DistilBETO": "dccuchile/distilbert-base-spanish-uncased",
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    "MarIA": "IsGarrido/roberta-base-bne",
    "XLM-R": "xlm-roberta-base",
    "mDeBERTa": "microsoft/mdeberta-v3-base",
    "XLM-Longformer": "markussagen/xlm-roberta-longformer-base-4096",
}

# --- CARGA DE DATOS ---
print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].map({"NM": 0, "M": 1})
print(f"Dataset cargado: {len(df)} canciones")
print(f"Distribución original:\n{df['label'].value_counts()}")
# Split simple 80/20 solo para esta tabla comparativa
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

# column lyrics -> text, label -> label
train_ds = Dataset.from_pandas(train_df.rename(columns={"lyrics": "text", "label": "label"}), preserve_index=False)
val_ds = Dataset.from_pandas(val_df.rename(columns={"lyrics": "text", "label": "label"}), preserve_index=False)


# Calcular pesos para clase desbalanceada
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
total = n_neg + n_pos
w0 = total / (2 * n_neg)
w1 = total / (2 * n_pos)

weights_tensor = torch.tensor([w0, w1]).float()
print(f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 1: {w1:.2f}")



def compute_metrics(pred):
    """Calcula métricas para evaluación"""
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


results_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- INICIANDO COMPARATIVA EN {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ---")

for name, model_id in MODELS.items():
    print(f"\n{'='*50}")
    print(f">>> Evaluando: {name}")
    print(f"{'='*50}")
    
    try:
        # 1. Tokenizador
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        is_robertuito = name == "Robertuito"
        
        # 2. Función de tokenización con truncación estándar
        def tokenize_fn(batch):
            texts = batch["text"]
            if is_robertuito:
                texts = [preprocess_tweet(t, lang="es") for t in texts]
                return tokenizer(texts, padding="max_length", truncation=True, max_length=128)
            # Truncación estándar (más efectiva para textos largos con mean pooling)
            return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)
        
        # 3. Tokenizar datasets
        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
        val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
        train_tok = train_tok.rename_column("label", "labels")
        val_tok = val_tok.rename_column("labels" if "labels" in val_tok.column_names else "label", "labels")
        train_tok.set_format("torch")
        val_tok.set_format("torch")
        
        # 4. Crear modelo
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            problem_type="single_label_classification"
        ).to(device)
        is_deberta = name == "mDeBERTa"
        current_lr = 2e-6 if is_deberta else 2e-5  # DeBERTa necesita un LR mucho más bajo
        print(f'{torch.cuda.is_bf16_supported()} -> Usando bf16: {torch.cuda.is_bf16_supported()} (ajustando configuración de entrenamiento)')
        # 5. Configurar entrenamiento (optimizado para MAX_LEN=512)
        args = TrainingArguments(
            #output_dir=f"{SAVE_DIR}/{name}/temp_checkpoints", # Carpeta temporal
            learning_rate=current_lr,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            num_train_epochs=10,              # Aumentamos épocas porque Early Stopping parará antes
            #bf16=torch.cuda.is_bf16_supported(),
            bf16=False,  # Desactivamos bf16 para evitar problemas con mDeBERTa, aunque sacrifiquemos algo de velocidad
            fp16=False,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",      # Evaluar cada época
            save_strategy="epoch",            # Guardar cada época (necesario para Early Stopping)
            load_best_model_at_end=True,      # Cargar el mejor modelo al terminar
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,               # Mantiene SOLO el mejor checkpoint, borra el resto
            report_to="none",
            gradient_checkpointing=False,
        )

        
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            compute_metrics=compute_metrics,
            class_weights=weights_tensor,  # Pasamos los pesos al trainer personalizado
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping
        )
        
        # 6. Entrenar y evaluar
        trainer.train()
        metrics = trainer.evaluate()
        
        results_list.append({
            "Modelo": name,
            "F1-Macro": metrics["eval_f1_macro"],
            "Accuracy": metrics["eval_accuracy"],
            "Precision": metrics["eval_precision"],
            "Recall": metrics["eval_recall"],
        })
        
        print(f" {name}: F1={metrics['eval_f1_macro']:.4f}")
        
        # Limpiar memoria
        del model, trainer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f" Error con {name}: {e}")
        results_list.append({
            "Modelo": name,
            "F1-Macro": None,
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
        })

# --- GUARDAR RESULTADOS ---
df_res = pd.DataFrame(results_list)
df_res = df_res.sort_values("F1-Macro", ascending=False)

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df_res.to_csv(RESULTS_FILE, index=False)

print(f"\n{'='*60}")
print("RESULTADOS COMPARATIVA")
print(f"{'='*60}")
print(df_res.to_markdown(index=False))
print(f"\n Guardado en: {RESULTS_FILE}")
