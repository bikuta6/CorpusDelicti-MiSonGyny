"""
Ensemble Stacking: Meta-modelo de Regresión Logística que aprende pesos óptimos
para combinar predicciones de múltiples modelos base.

Uso:
    1. Entrenar modelos base con entrenamiento.py
    2. Ejecutar: python ensemble_stacking.py
    3. Usar el meta-modelo en inferencia.py automáticamente
"""

import os
import sys
import torch
import gc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pysentimiento.preprocessing import preprocess_tweet
from tqdm import tqdm
import joblib
from sklearn.model_selection import StratifiedKFold

# Añadimos path para importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
MODELS_DIR = "../../models/task1/ensemble"
TRAIN_FILE = "../../data/task1/train.csv"
VAL_FILE = "../../data/task1/validation.csv"
STACKER_OUTPUT = "../../models/task1/ensemble/stacker.pkl"
MAX_LEN = 512

# Modelos base a usar en el ensemble
MODEL_MAP = {
    "mDeBERTa": "microsoft/mdeberta-v3-base",
    "Robertuito": "pysentimiento/robertuito-base-uncased",
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_name, fold=0):
    # La ruta debe coincidir exactamente con como guardaste en entrenamiento.py
    ckpt_path = os.path.join(MODELS_DIR, model_name, f"fold_{fold}")
    
    if not os.path.exists(ckpt_path):
        # Backup por si la estructura es diferente
        ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_fold_{fold}")
    
    if not os.path.exists(ckpt_path):
         raise FileNotFoundError(f"No se encontró el fold en: {ckpt_path}")

    # Cargar configuración para asegurar que problem_type sea correcto
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    
    # Si el tokenizer no está en la carpeta del fold, intenta cargarlo del ID original
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    except:
        print(f"  Aviso: Tokenizer no encontrado en local, cargando de {MODEL_MAP[model_name]}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    
    model.to(device)
    model.eval()
    return model, tokenizer


def get_model_predictions(df, model_name, fold=0):
    """
    Obtiene probabilidades [prob_class_0, prob_class_1] de un modelo.
    
    Args:
        df: DataFrame con columna 'text'
        model_name: Nombre del modelo
        fold: Fold a usar
    
    Returns:
        np.array: (N_samples, 2) con probabilidades
    """
    print(f"\n>>> Obteniendo predicciones de {model_name} (fold {fold})")
    
    try:
        model, tokenizer = load_trained_model(model_name, fold)
    except FileNotFoundError as e:
        print(f"  {e}")
        print(f"  Saltando {model_name}")
        return None
    
    is_robertuito = model_name == "Robertuito"
    predictions = []
    
    with torch.no_grad():
        for text in tqdm(df["text"], desc=f"  Inferencia {model_name}"):
            # Preprocesamiento específico
            if is_robertuito:
                text = preprocess_tweet(text, lang="es")
            
            # Tokenizar
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN
            ).to(device)
            
            # Predecir
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            predictions.append(probs)
    
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect() # Forzar recolección de basura de Python
    
    return np.array(predictions)


def create_meta_features(df, fold=0):
    """
    Crea matriz de meta-features: probabilidades de todos los modelos base.
    
    Args:
        df: DataFrame con textos
        fold: Fold a usar de los modelos
    
    Returns:
        np.array: (N_samples, N_models * 2) con todas las probabilidades
    """
    meta_features_list = []
    model_names_used = []
    
    for model_name in MODEL_MAP.keys():
        probs = get_model_predictions(df, model_name, fold)
        
        if probs is not None:
            meta_features_list.append(probs)
            model_names_used.append(model_name)
    
    if not meta_features_list:
        raise ValueError("No se pudo cargar ningún modelo base")
    
    # Concatenar: (N_samples, N_models, 2) -> (N_samples, N_models*2)
    meta_features = np.hstack(meta_features_list)
    
    print(f"\n Meta-features creadas: {meta_features.shape}")
    print(f"   Modelos usados: {', '.join(model_names_used)}")
    
    return meta_features, model_names_used


def train_stacker():
    """
    Entrena el meta-modelo de stacking usando predicciones Out-of-Fold (OOF)
    para evitar el sobreajuste y el leakage.
    """
    print("="*60)
    print("ENTRENAMIENTO DE META-MODELO (STACKING OOF)")
    print("="*60)
    
    # 1. Cargar el dataset de entrenamiento original
    # Usaremos el mismo orden que en el K-Fold de entrenamiento.py
    train_df = pd.read_csv(TRAIN_FILE)
    y_train = train_df["label"].values
    
    # 2. Generar Meta-Features OOF (Entrenamiento)
    # Necesitamos reconstruir las predicciones de CADA fold para CADA modelo
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Matriz para guardar las predicciones OOF: (N_samples, N_modelos * 2)
    X_train_oof = np.zeros((len(train_df), len(MODEL_MAP) * 2))
    
    print(f"\nGenerando predicciones Out-of-Fold para {len(MODEL_MAP)} modelos...")
    
    for m_idx, (model_name, _) in enumerate(MODEL_MAP.items()):
        print(f" > Procesando modelo: {model_name}")
        
        # Recorremos los mismos folds que en entrenamiento.py
        for fold, (_, val_idx) in enumerate(skf.split(train_df, y_train)):
            # Cargar el modelo específico de ese fold
            try:
                model, tokenizer = load_trained_model(model_name, fold)
                
                # Predecir SOLO el fragmento que fue validación en ese fold
                fold_val_df = train_df.iloc[val_idx]
                
                # Reutilizamos (o adaptamos) get_model_predictions para este subset
                # Nota: Es más eficiente pasarle el subset directamente aquí
                fold_probs = []
                is_robertuito = "Robertuito" in model_name
                
                with torch.no_grad():
                    for text in fold_val_df["text"]:
                        if is_robertuito: text = preprocess_tweet(text, lang="es")
                        inputs = tokenizer(text, return_tensors="pt", padding="max_length", 
                                         truncation=True, max_length=MAX_LEN).to(device)
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                        fold_probs.append(probs)
                
                # Guardar en la posición correcta de la matriz global
                # m_idx*2 es prob_neg, m_idx*2 + 1 es prob_pos
                X_train_oof[val_idx, m_idx*2 : m_idx*2 + 2] = np.array(fold_probs)
                
                del model, tokenizer
                torch.cuda.empty_cache()
                
            except FileNotFoundError:
                print(f"   [!] Error: No se encontró el checkpoint del Fold {fold}")
                return

    # 3. Generar Meta-Features para Validación (Promedio de Folds)
    # Para el set de validación externo o test, promediamos las predicciones de los 5 folds
    print("\nGenerando predicciones para el set de Validación externo (Promedio de Folds)...")
    val_df = pd.read_csv(VAL_FILE)
    y_val = val_df["label"].values
    X_val_meta = np.zeros((len(val_df), len(MODEL_MAP) * 2))

    for m_idx, (model_name, _) in enumerate(MODEL_MAP.items()):
        model_val_probs = []
        for fold in range(5):
            # En una implementación real, aquí promediarías los 5 modelos. 
            # Para simplificar, usaremos el fold 0 o el promedio si lo prefieres:
            fold_probs = get_model_predictions(val_df, model_name, fold)
            model_val_probs.append(fold_probs)
        
        # Promedio de los 5 folds para este modelo
        X_val_meta[:, m_idx*2 : m_idx*2 + 2] = np.mean(model_val_probs, axis=0)

    # 4. Entrenar Meta-Modelo
    stacker = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    stacker.fit(X_train_oof, y_train)
    
    # 5. Evaluación y Guardado
    val_preds = stacker.predict(X_val_meta)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL STACKING OOF")
    print(f"F1-Macro Validación: {val_f1:.4f}")
    print(f"{'='*60}")

    # Guardar igual que antes
    stacker_data = {
        "model": stacker,
        "models_used": list(MODEL_MAP.keys()),
        "val_f1": val_f1
    }
    joblib.dump(stacker_data, STACKER_OUTPUT)
    return stacker, val_f1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar meta-modelo de stacking")
    parser.add_argument("--subset", action="store_true", 
                       help="Usar subset de train para pruebas rápidas")
    
    args = parser.parse_args()
    
    try:
        stacker, val_f1 = train_stacker(use_train_subset=args.subset)
        print(f"\n{'='*60}")
        print(f" ENTRENAMIENTO COMPLETADO")
        print(f"   F1-Macro en Validación: {val_f1:.4f}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
