import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet

# --- IMPORTAR MODELO PERSONALIZADO ---
# Añadimos el directorio padre al path para importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from augmentation import create_augmented_dataset
from utils import set_seed, DEFAULT_SEED
from trainer import WeightedTrainer

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
TRAIN_FILE = "../../data/task1/train.csv"
SAVE_DIR = "../../models/task1/ensemble"
N_FOLDS = 5

# Estadísticas del dataset:
# - Mediana: 392 tokens (50% canciones)
# - Percentil 90: 970 tokens
# - Clase positiva más larga: mediana 683 tokens
MAX_LEN = 512  # Captura ~60% del contenido (vs 256 que solo capturaba ~30%)

# Augmentation para canciones
USE_AUGMENTATION = True   # Activar data augmentation
AUGMENT_POSITIVE_ONLY = True  # Solo aumentar clase minoritaria
N_AUGMENTS = 2            # Variaciones por muestra positiva

# Ensemble de 3 modelos optimizados para español
MODELS_TO_TRAIN = {
    "mDeBERTa": "microsoft/mdeberta-v3-base",     # SOTA multilingual
    "Robertuito": "pysentimiento/robertuito-base-uncased",  # Slang español
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",  # BERT español clásico
}


# --- CARGAR DATOS ---
print(f"Cargando datos desde {TRAIN_FILE}...")
df = pd.read_csv(TRAIN_FILE)
texts = df["text"].tolist()
labels = df["label"].tolist()

# --- CÁLCULO DE PESOS (CLASE DESBALANCEADA) ---
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)

total = n_neg + n_pos
w0 = total / (2 * n_neg)
w1 = total / (2 * n_pos)

weights_tensor = torch.tensor([w0, w1]).float()


print(f"Desbalance detectado -> Neg: {n_neg}, Pos: {n_pos}")
print(f"Pesos aplicados al Loss: {weights_tensor}")

# --- BUCLE K-FOLD ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for model_name, model_id in MODELS_TO_TRAIN.items():
    print(f"\n>>> Entrenando Arquitectura: {model_name}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"   > Fold {fold+1}/{N_FOLDS}")

        # Crear DataFrame temporal para train
        train_df_fold = pd.DataFrame({
            "text": [texts[i] for i in train_idx],
            "label": [labels[i] for i in train_idx],
        })
        
        # Aplicar Data Augmentation (solo en train, nunca en val)
        if USE_AUGMENTATION:
            train_df_fold = create_augmented_dataset(
                train_df_fold, 
                augment_positive_only=AUGMENT_POSITIVE_ONLY,
                n_augments=N_AUGMENTS
            )
            print(f"      Augmentation: {len(train_idx)} -> {len(train_df_fold)} muestras")
        
        # Crear Datasets
        train_ds = Dataset.from_pandas(train_df_fold, preserve_index=False)
        val_ds = Dataset.from_dict({
            "text": [texts[i] for i in val_idx],
            "label": [labels[i] for i in val_idx]
        })

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def tokenize(batch):
            batch_texts = batch["text"]
            # Preprocesamiento para Robertuito
            if model_name == "Robertuito":
                batch_texts = [preprocess_tweet(t, lang="es") for t in batch_texts]
            # Truncación estándar (más efectiva que smart_truncate para estas estadísticas)
            return tokenizer(
                batch_texts, padding="max_length", truncation=True, max_length=MAX_LEN
            )

        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)

        # --- INSTANCIAR MODELO PERSONALIZADO ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            problem_type="single_label_classification"
        )

        # Argumentos de Entrenamiento (optimizados para MAX_LEN=512)
        args = TrainingArguments(
            output_dir=f"{SAVE_DIR}/temp_checkpoints", # Carpeta temporal
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            num_train_epochs=10,              # Aumentamos épocas porque Early Stopping parará antes
            bf16=torch.cuda.is_bf16_supported(),
            fp16=False,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",      # Evaluar cada época
            save_strategy="epoch",            # Guardar cada época (necesario para Early Stopping)
            load_best_model_at_end=True,      # Cargar el mejor modelo al terminar
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,               # Mantiene SOLO el mejor checkpoint, borra el resto
            report_to="none",
            gradient_checkpointing=True,
        )

        def compute_metrics(p):
            # Como es binario, usamos argmax sobre los logits
            preds = np.argmax(p.predictions, axis=1)
            return {"f1": f1_score(p.label_ids, preds, average="macro")}

        # --- TRAINER ESTÁNDAR ---
        # Usamos el Trainer normal de Hugging Face.
        # Al llamar a model(labels=...), tu clase MisogynyClassifier ya calcula y devuelve 'loss'.
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            class_weights=weights_tensor,  # Pasamos los pesos al trainer personalizado
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Early stopping
        )

        trainer.train()

        # Guardar compatible con HuggingFace
        trainer.save_model(f"{SAVE_DIR}/{model_name}/fold_{fold}")
        tokenizer.save_pretrained(f"{SAVE_DIR}/{model_name}/fold_{fold}")

        del model, trainer, tokenizer
        torch.cuda.empty_cache()

print("\n¡Entrenamiento completado exitosamente!")