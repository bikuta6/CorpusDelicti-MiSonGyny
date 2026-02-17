import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet

# --- IMPORTAR MODELO PERSONALIZADO ---
# Añadimos el directorio padre al path para importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import MisogynyClassifier
from augmentation import create_augmented_dataset

# --- CONFIGURACIÓN ---
TRAIN_FILE = "../../data/task1/train.csv"
SAVE_DIR = "../../models/task1/ensemble"
N_FOLDS = 5
MAX_LEN = 256  # Optimizado para GPU de 12GB (4070 Ti)

# Augmentation para canciones (ajustar según necesidad)
USE_AUGMENTATION = True   # Activar data augmentation
AUGMENT_POSITIVE_ONLY = True  # Solo aumentar clase minoritaria
N_AUGMENTS = 2            # Variaciones por muestra positiva

MODELS_TO_TRAIN = {
    "mDeBERTa": "microsoft/mdeberta-v3-base",
    "Robertuito": "pysentimiento/robertuito-base-uncased",
}

# --- FUNCIONES AUXILIARES PARA CANCIONES ---
def smart_truncate(text, tokenizer, max_len=256):
    """
    Truncado inteligente para canciones largas:
    - Mantiene inicio (intro/contexto)
    - Mantiene final (conclusión)
    - Corta del medio si es necesario
    
    Esto preserva mejor el significado que truncar solo al final.
    """
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    
    if len(tokens) <= max_len - 2:  # -2 para [CLS] y [SEP]
        return text
    
    # Mantener 45% inicio, 45% final, descartar 10% del medio
    keep_tokens = max_len - 2
    head_len = int(keep_tokens * 0.45)
    tail_len = int(keep_tokens * 0.45)
    
    head_tokens = tokens[:head_len]
    tail_tokens = tokens[-tail_len:]
    
    # Reconstruir texto
    head_text = tokenizer.decode(head_tokens, skip_special_tokens=True)
    tail_text = tokenizer.decode(tail_tokens, skip_special_tokens=True)
    
    return head_text + " [...] " + tail_text

# --- CARGAR DATOS ---
print(f"Cargando datos desde {TRAIN_FILE}...")
df = pd.read_csv(TRAIN_FILE)
texts = df["text"].tolist()
labels = df["label"].tolist()

# --- CÁLCULO DE PESOS (CLASE DESBALANCEADA) ---
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)

# Calculamos el ratio para equilibrar la balanza
# Si hay 1000 negativos y 100 positivos, el ratio es 10.
ratio = n_neg / (n_pos + 1e-5)

# Creamos el tensor de pesos: [Peso_Clase_0, Peso_Clase_1]
# Clase 0 (No Misógino) = 1.0
# Clase 1 (Misógino) = ratio
weights_tensor = torch.tensor([1.0, ratio]).float()

print(f"Desbalance detectado -> Neg: {n_neg}, Pos: {n_pos}")
print(f"Pesos aplicados al Loss: {weights_tensor}")

# --- BUCLE K-FOLD ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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
            # Smart truncation para canciones largas
            batch_texts = [smart_truncate(t, tokenizer, MAX_LEN) for t in batch_texts]
            return tokenizer(
                batch_texts, padding="max_length", truncation=True, max_length=MAX_LEN
            )

        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)

        # --- INSTANCIAR MODELO PERSONALIZADO ---
        # Aquí ocurre la magia: Pasamos los pesos y el dropout al constructor
        model = MisogynyClassifier.from_pretrained_base(
            model_id,
            num_labels=2,
            dropout_rate=0.2,              # Ajustable si ves overfitting
            class_weights=weights_tensor.to(device),
            use_focal_loss=True,  # ✅ Focal Loss integrado
            focal_gamma=2.0,
        ).to("cuda")

        # Argumentos de Entrenamiento
        args = TrainingArguments(
            output_dir=f"{SAVE_DIR}/{model_name}_fold_{fold}",
            learning_rate=2e-5,
            per_device_train_batch_size=4,   # Batch bajo por MAX_LEN alto
            gradient_accumulation_steps=8,   # Batch efectivo 32
            num_train_epochs=4,
            fp16=True,                       # Vital para RTX 4070 Ti
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            report_to="none",
        )

        def compute_metrics(p):
            # Como es binario, usamos argmax sobre los logits
            preds = np.argmax(p.predictions, axis=1)
            return {"f1": f1_score(p.label_ids, preds, average="macro")}

        # --- TRAINER ESTÁNDAR ---
        # Usamos el Trainer normal de Hugging Face.
        # Al llamar a model(labels=...), tu clase MisogynyClassifier ya calcula y devuelve 'loss'.
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Guardar compatible con HuggingFace
        trainer.save_model(f"{SAVE_DIR}/{model_name}/fold_{fold}")

        del model, trainer, tokenizer
        torch.cuda.empty_cache()

print("\n¡Entrenamiento completado exitosamente!")