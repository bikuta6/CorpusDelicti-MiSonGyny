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
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet
from tqdm import tqdm
import joblib

# Añadimos path para importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import MisogynyClassifier
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
    """
    Carga un modelo entrenado desde el directorio de ensemble.
    
    Args:
        model_name: Nombre del modelo (mDeBERTa, BETO, etc.)
        fold: Número de fold a cargar (default: 0 = mejor)
    
    Returns:
        model, tokenizer
    """
    model_folder = os.path.join(MODELS_DIR, f"{model_name}_fold_{fold}")
    
    if not os.path.exists(model_folder):
        # Intentar con fold/checkpoint
        model_folder = os.path.join(MODELS_DIR, model_name, f"fold_{fold}")
    
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"No se encontró modelo en: {model_folder}")
    
    # Buscar checkpoint dentro del folder
    ckpt_dirs = [d for d in os.listdir(model_folder) if "checkpoint" in d or d == "fold_0"]
    
    if ckpt_dirs:
        ckpt_path = os.path.join(model_folder, ckpt_dirs[0])
    else:
        ckpt_path = model_folder
    
    print(f"  Cargando desde: {ckpt_path}")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # Cargar modelo
    base_model_id = MODEL_MAP[model_name]
    model = MisogynyClassifier(
        model_name_or_path=base_model_id,
        num_labels=2,
        dropout_rate=0.3,
        is_multilabel=False,
        pooling_strategy="mean"
    )
    
    # Cargar pesos entrenados
    weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        # Intentar cargar del safetensors o completo
        from transformers import AutoModel
        model = MisogynyClassifier.from_pretrained(ckpt_path)
    
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
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
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


def train_stacker(use_train_subset=False):
    """
    Entrena el meta-modelo de stacking.
    
    Args:
        use_train_subset: Si True, usa solo subset de train para velocidad
    """
    print("="*60)
    print("ENTRENAMIENTO DE META-MODELO (STACKING)")
    print("="*60)
    
    # Cargar datos
    print(f"\nCargando datos...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    
    # Usar subset si se especifica (para pruebas rápidas)
    if use_train_subset and len(train_df) > 500:
        print(f" Usando subset de train ({500} muestras) para velocidad")
        train_df = train_df.sample(n=500, random_state=42)
    
    print(f"  Train: {len(train_df)} muestras")
    print(f"  Val:   {len(val_df)} muestras")
    
    # Crear meta-features (probabilidades de modelos base)
    print("\n" + "="*60)
    print("PASO 1: EXTRAER PREDICCIONES DE MODELOS BASE EN TRAIN")
    print("="*60)
    X_train, models_used = create_meta_features(train_df, fold=0)
    y_train = train_df["label"].values
    
    print("\n" + "="*60)
    print("PASO 2: EXTRAER PREDICCIONES DE MODELOS BASE EN VALIDATION")
    print("="*60)
    X_val, _ = create_meta_features(val_df, fold=0)
    y_val = val_df["label"].values
    
    # Entrenar meta-modelo
    print("\n" + "="*60)
    print("PASO 3: ENTRENAR META-MODELO (LOGISTIC REGRESSION)")
    print("="*60)
    
    stacker = LogisticRegression(
        penalty="l2",
        C=1.0,                    # Regularización moderada
        class_weight="balanced",  # Compensar desbalance
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED
    )
    
    print("  Ajustando Logistic Regression...")
    stacker.fit(X_train, y_train)
    
    # Evaluar
    print("\n" + "="*60)
    print("PASO 4: EVALUACIÓN DEL META-MODELO")
    print("="*60)
    
    # Predicciones
    train_preds = stacker.predict(X_train)
    val_preds = stacker.predict(X_val)
    
    # Métricas
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average="macro")
    
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    
    print(f"\n Resultados en TRAIN:")
    print(f"   Accuracy: {train_acc:.4f}")
    print(f"   F1-Macro: {train_f1:.4f}")
    
    print(f"\n Resultados en VALIDATION:")
    print(f"   Accuracy: {val_acc:.4f}")
    print(f"   F1-Macro: {val_f1:.4f}")
    
    # Mostrar pesos aprendidos
    print("\n" + "="*60)
    print("PESOS APRENDIDOS POR EL META-MODELO")
    print("="*60)
    
    feature_names = []
    for model_name in models_used:
        feature_names.extend([f"{model_name}_prob_neg", f"{model_name}_prob_pos"])
    
    weights = stacker.coef_[0]
    intercept = stacker.intercept_[0]
    
    print(f"\n{'Feature':<35} {'Peso':>10}")
    print("-" * 45)
    for name, weight in zip(feature_names, weights):
        sign = "" if abs(weight) > 0.1 else "  "
        print(f"{sign} {name:<33} {weight:+10.4f}")
    print("-" * 45)
    print(f"   {'Intercept':<33} {intercept:+10.4f}")
    
    # Calcular importancia por modelo
    print("\n IMPORTANCIA POR MODELO (suma abs de pesos):")
    for model_name in models_used:
        idx_neg = feature_names.index(f"{model_name}_prob_neg")
        idx_pos = feature_names.index(f"{model_name}_prob_pos")
        importance = abs(weights[idx_neg]) + abs(weights[idx_pos])
        print(f"   {model_name:<15}: {importance:.4f}")
    
    # Guardar meta-modelo
    print(f"\n Guardando meta-modelo en: {STACKER_OUTPUT}")
    os.makedirs(os.path.dirname(STACKER_OUTPUT), exist_ok=True)
    
    # Guardar modelo y metadatos
    stacker_data = {
        "model": stacker,
        "models_used": models_used,
        "feature_names": feature_names,
        "val_f1": val_f1,
        "val_accuracy": val_acc
    }
    
    joblib.dump(stacker_data, STACKER_OUTPUT)
    print(" Meta-modelo guardado exitosamente")
    
    # Reporte detallado de validación
    print("\n CLASSIFICATION REPORT (Validation):")
    print(classification_report(y_val, val_preds, target_names=["No Misógino", "Misógino"]))
    
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
