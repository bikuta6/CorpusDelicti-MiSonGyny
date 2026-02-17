import os
import time

import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# --- CONFIGURACIÓN ---
DATA_PATH = "../../data/task1/train.csv"
OUTPUT_DIR = "../../models/task1/comparativa"
RESULTS_FILE = "../../results/task1/tabla_paper.csv"

# Modelos a comparar para el Paper
MODELS = {
    "DistilBETO": "dccuchile/distilbert-base-spanish-uncased",  # Español eficiente
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",  # Español clásico
    "MarIA": "PlanTL-GOB-ES/roberta-base-bne",  # SOTA en español
    "XLM-R": "xlm-roberta-base",  # Clásico multilingual
    "mDeBERTa": "microsoft/mdeberta-v3-base",  # SOTA multilingual
    "Robertuito": "pysentimiento/robertuito-base-uncased",  # Especializado en slang en español
}

# --- CARGA DE DATOS ---
print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
# Split simple 80/20 solo para esta tabla comparativa
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1_macro": f1}


results_list = []

print(f"--- INICIANDO COMPARATIVA EN {torch.cuda.get_device_name(0)} ---")

for name, model_id in MODELS.items():
    print(f"\n>>> Evaluando: {name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(batch):
        texts = batch["text"]
        # Pre-procesamiento especial SOLO para Robertuito
        if name == "Robertuito":
            texts = [preprocess_tweet(t, lang="es") for t in texts]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_val = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2
    ).to("cuda")

    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,  # Pocas épocas para la comparativa
        fp16=True,  # RTX 4070 Ti optimization
        evaluation_strategy="epoch",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # Medir Tiempos y Memoria
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    metrics = trainer.evaluate()

    results_list.append(
        {
            "Modelo": name,
            "F1-Macro": round(metrics["eval_f1_macro"], 4),
            "Tiempo (s)": round(end_time - start_time, 2),
            "VRAM Max (GB)": round(peak_mem, 2),
            "Params (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        }
    )

    del model, trainer, tokenizer
    torch.cuda.empty_cache()

# Guardar Resultados
df_res = pd.DataFrame(results_list)
os.makedirs("../../task1/results", exist_ok=True)
df_res.to_csv(RESULTS_FILE, index=False)
print(f"\nTabla guardada en {RESULTS_FILE}")
print(df_res)
