import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet

# --- CONFIGURACIÓN ---
TRAIN_FILE = "../../data/task1/train.csv"
SAVE_DIR = "../../models/task1/ensemble"
N_FOLDS = 5
MAX_LEN = 256  # Aumentado para canciones largas (tienes 12GB VRAM)

MODELS_TO_TRAIN = {
    "mDeBERTa": "microsoft/mdeberta-v3-base",
    "Robertuito": "pysentimiento/robertuito-base-uncased",
}

# --- CARGAR DATOS ---
df = pd.read_csv(TRAIN_FILE)
texts = df["text"].tolist()
labels = df["label"].tolist()

# --- CÁLCULO DE PESOS (CLASE DESBALANCEADA) ---
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
pos_weight = torch.tensor(n_neg / (n_pos + 1e-5)).to("cuda")
print(f"Peso para clase Misógina: {pos_weight.item():.2f}")


# --- TRAINER PERSONALIZADO ---
class WeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight]).to(model.device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- BUCLE K-FOLD ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for model_name, model_id in MODELS_TO_TRAIN.items():
    print(f"\n>>> Entrenando Arquitectura: {model_name}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"   > Fold {fold+1}/{N_FOLDS}")

        # Datasets
        train_ds = Dataset.from_dict(
            {
                "text": [texts[i] for i in train_idx],
                "label": [labels[i] for i in train_idx],
            }
        )
        val_ds = Dataset.from_dict(
            {"text": [texts[i] for i in val_idx], "label": [labels[i] for i in val_idx]}
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def tokenize(batch):
            batch_texts = batch["text"]
            if model_name == "Robertuito":
                batch_texts = [preprocess_tweet(t, lang="es") for t in batch_texts]
            return tokenizer(
                batch_texts, padding="max_length", truncation=True, max_length=MAX_LEN
            )

        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2
        ).to("cuda")

        # Argumentos optimizados para RTX 4070 Ti (12GB)
        args = TrainingArguments(
            output_dir=f"{SAVE_DIR}/{model_name}_fold_{fold}",
            learning_rate=2e-5,
            per_device_train_batch_size=4,  # Bajamos batch por MAX_LEN=256
            gradient_accumulation_steps=8,  # Batch efectivo = 32
            num_train_epochs=4,
            fp16=True,  # Vital para velocidad/memoria
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            report_to="none",
        )

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return {"f1": f1_score(p.label_ids, preds, average="macro")}

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        del model, trainer, tokenizer
        torch.cuda.empty_cache()

print("¡Entrenamiento completado!")
