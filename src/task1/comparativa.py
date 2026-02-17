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
)

# Añadimos la carpeta padre al path para poder importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import MisogynyClassifier
from utils import set_seed, DEFAULT_SEED

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
DATA_PATH = "../../data/task1/train.csv"
OUTPUT_DIR = "../../models/task1/comparativa"
RESULTS_FILE = "../../results/task1/tabla_paper.csv"

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
    "Robertuito": "pysentimiento/robertuito-base-uncased",
}

# --- CARGA DE DATOS ---
print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Split simple 80/20 solo para esta tabla comparativa
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds = Dataset.from_pandas(val_df, preserve_index=False)

# Calcular pesos para clase desbalanceada
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
ratio = n_neg / (n_pos + 1e-5)
weights_tensor = torch.tensor([1.0, ratio]).float()
print(f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 1: {ratio:.2f}")


# --- SMART TRUNCATE PARA CANCIONES ---
def smart_truncate(text, tokenizer, max_len=256):
    """
    Truncado inteligente: mantiene inicio y final de la canción.
    Preserva contexto de intro y conclusión.
    """
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    
    if len(tokens) <= max_len - 2:  # -2 para [CLS] y [SEP]
        return text
    
    keep_tokens = max_len - 2
    head_len = int(keep_tokens * 0.45)
    tail_len = int(keep_tokens * 0.45)
    
    head_tokens = tokens[:head_len]
    tail_tokens = tokens[-tail_len:]
    
    head_text = tokenizer.decode(head_tokens, skip_special_tokens=True)
    tail_text = tokenizer.decode(tail_tokens, skip_special_tokens=True)
    
    return head_text + " [...] " + tail_text


def compute_metrics(pred):
    """Calcula métricas para evaluación"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
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
            # Truncación estándar (más efectiva para textos largos con mean pooling)
            return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)
        
        # 3. Tokenizar datasets
        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        train_tok = train_tok.rename_column("label", "labels")
        val_tok = val_tok.rename_column("labels" if "labels" in val_tok.column_names else "label", "labels")
        train_tok.set_format("torch")
        val_tok.set_format("torch")
        
        # 4. Crear modelo
        model = MisogynyClassifier(
            model_name_or_path=model_id,
            num_labels=2,
            dropout_rate=0.3,
            class_weights=weights_tensor.to(device),
            use_focal_loss=True,
            focal_gamma=2.0,
            pooling_strategy="mean",  # Mean pooling para canciones largas
        ).to(device)
        
        # 5. Configurar entrenamiento (optimizado para MAX_LEN=512)
        args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/{name}",
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Reducido para MAX_LEN=512
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,  # Batch efectivo = 32
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            fp16=True,
            report_to="none",
            save_total_limit=1,
            gradient_checkpointing=True,  # Ahorra VRAM
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            compute_metrics=compute_metrics,
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
