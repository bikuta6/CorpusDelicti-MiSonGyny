import os
import sys
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from pysentimiento.preprocessing import preprocess_tweet
from tqdm import tqdm
import joblib

# Añadimos path para importar models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import MisogynyClassifier

# --- CONFIGURACIÓN ---
TEST_FILE = "../../data/task1/test.csv"
VAL_FILE = "../../data/task1/validation.csv"
MODELS_DIR = "../../models/task1/ensemble"
OUTPUT_FILE = "../../results/task1/submission.csv"
STACKER_PATH = "../../models/task1/ensemble/stacker.pkl"  # Meta-modelo

# Parámetros de Ventana Deslizante (optimizado para canciones largas)
# Estadísticas: Mediana 392, P90: 970, Max: 4513 tokens
WINDOW_LEN = 512  # Aumentado de 256 para capturar más contenido
STRIDE = 256  # 50% overlap entre ventanas
MIN_CHUNK_TOKENS = 30  # Mínimo de tokens por ventana

# Estrategia de agregación para canciones
# Opciones: "max", "mean", "weighted", "top_k"
AGGREGATION_STRATEGY = "weighted"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Diccionario para saber qué arquitectura base cargar según el nombre de la carpeta
MODEL_MAP = {
    "DistilBETO": "dccuchile/distilbert-base-spanish-uncased",
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    "MarIA": "PlanTL-GOB-ES/roberta-base-bne",
    "XLM-R": "xlm-roberta-base",
    "mDeBERTa": "microsoft/mdeberta-v3-base",
    "Robertuito": "pysentimiento/robertuito-base-uncased",
}


def detect_chorus_lines(text):
    """
    Detecta líneas de estribillo (repetidas 2+ veces).
    En canciones, el estribillo suele contener el mensaje central.
    """
    lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
    line_counts = Counter(lines)
    return {line for line, count in line_counts.items() if count >= 2}


def calculate_chunk_weight(chunk_text, chorus_lines):
    """
    Calcula el peso de una ventana basándose en si contiene
    líneas del estribillo (que suele ser más importante).
    """
    chunk_lines = [l.strip().lower() for l in chunk_text.split("\n") if l.strip()]
    chorus_overlap = sum(1 for l in chunk_lines if l in chorus_lines)

    # Base weight 1.0, +0.3 por cada línea de estribillo encontrada
    return 1.0 + (chorus_overlap * 0.3)


# --- FUNCIÓN SLIDING WINDOW OPTIMIZADA PARA CANCIONES ---
def predict_sliding_window(text, model, tokenizer, is_robertuito, strategy=None):
    """
    Sliding window con ponderación de estribillos para letras de canciones.

    Args:
        text: Letra de la canción
        model: Modelo de clasificación
        tokenizer: Tokenizador
        is_robertuito: Si es Robertuito, aplicar preprocesamiento
        strategy: Estrategia de agregación (None usa AGGREGATION_STRATEGY global)
    """
    if strategy is None:
        strategy = AGGREGATION_STRATEGY

    # Detectar estribillo antes de preprocesar
    chorus_lines = detect_chorus_lines(text)

    if is_robertuito:
        text = preprocess_tweet(text, lang="es")

    # Tokenizamos sin añadir tokens especiales todavía para poder cortar
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    total_tokens = len(tokens)

    # Caso 1: El texto cabe entero en la ventana
    if total_tokens <= WINDOW_LEN:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=WINDOW_LEN,
            padding="max_length",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
        return probs[0][1].item()

    # Caso 2: Canción larga -> Ventana Deslizante con pesos
    window_probs = []
    window_weights = []

    for i in range(0, total_tokens, STRIDE):
        chunk_ids = tokens[i : i + WINDOW_LEN]
        if len(chunk_ids) < MIN_CHUNK_TOKENS:
            break

        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Calcular peso según presencia de estribillo
        weight = calculate_chunk_weight(chunk_text, chorus_lines)

        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=WINDOW_LEN,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            window_probs.append(probs[0][1].item())
            window_weights.append(weight)

        if i + WINDOW_LEN >= total_tokens:
            break

    if not window_probs:
        return 0.5  # Fallback para casos extremos

    # Aplicar estrategia de agregación
    if strategy == "max":
        # Si alguna ventana es muy misógina, toda la canción lo es
        return max(window_probs)
    elif strategy == "mean":
        return np.mean(window_probs)
    elif strategy == "weighted":
        # Ponderar por importancia (estribillo pesa más)
        return np.average(window_probs, weights=window_weights)
    elif strategy == "top_k":
        # Promedio de las k ventanas más misóginas
        k = min(3, len(window_probs))
        return np.mean(sorted(window_probs)[-k:])
    else:
        # Estrategia original: top-2 mean
        return np.mean(sorted(window_probs)[-2:])


# --- CARGAR DATOS ---
print("Cargando datasets...")
df_test = pd.read_csv(TEST_FILE)
df_val = pd.read_csv(VAL_FILE)

# --- CARGAR META-MODELO (STACKING) SI EXISTE ---
USE_STACKING = os.path.exists(STACKER_PATH)
stacker_data = None

if USE_STACKING:
    print(f"\n Meta-modelo de stacking encontrado: {STACKER_PATH}")
    stacker_data = joblib.load(STACKER_PATH)
    stacker = stacker_data["model"]
    models_used = stacker_data["models_used"]
    val_f1_stacker = stacker_data["val_f1"]
    print(f"   Modelos usados: {', '.join(models_used)}")
    print(f"   F1 en validación (entrenamiento): {val_f1_stacker:.4f}")
    print(f"   → Usando STACKING para ensemble\n")
else:
    print(f"\n No se encontró meta-modelo de stacking")
    print(f"   → Usando PROMEDIO SIMPLE para ensemble")
    print(f"   → Ejecuta 'python ensemble_stacking.py' para entrenar el meta-modelo\n")

# Matrices para guardar resultados: [N_Modelos, N_Samples]
val_preds_matrix = []
test_preds_matrix = []

# --- BUCLE DE INFERENCIA ---
# Buscamos carpetas en MODELS_DIR
model_folders = [f.path for f in os.scandir(MODELS_DIR) if f.is_dir()]
print(f"--- INICIANDO INFERENCIA CON {len(model_folders)} MODELOS ---")

for folder in model_folders:
    # Buscar el checkpoint dentro de la carpeta del fold
    ckpt_dirs = [d for d in os.listdir(folder) if "checkpoint" in d]
    if not ckpt_dirs:
        continue

    # Ruta completa al archivo de pesos (pytorch_model.bin)
    ckpt_path = os.path.join(folder, ckpt_dirs[0])
    weights_path = os.path.join(ckpt_path, "pytorch_model.bin")

    folder_name = os.path.basename(folder)
    print(f"\nProcesando modelo: {folder_name}")

    # 1. Identificar qué arquitectura base es (mDeBERTa, Robertuito, etc.)
    base_model_id = None
    for key, val in MODEL_MAP.items():
        if key in folder_name:
            base_model_id = val
            break

    if base_model_id is None:
        print(f" No se reconoció la arquitectura en {folder_name}, saltando...")
        continue

    # 2. carga el modelo con los pesos del checkpoint
    print(f"  > Cargando modelo desde: {weights_path}")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path
    )  # El tokenizer sí se carga del checkpoint
    model.eval()
    is_robertuito = "Robertuito" in folder_name

    # 4. Predecir Validación
    print("  > Inferenciando Validación...")
    fold_val_probs = []
    for text in tqdm(df_val["text"], leave=False):
        p = predict_sliding_window(text, model, tokenizer, is_robertuito)
        fold_val_probs.append(p)
    val_preds_matrix.append(fold_val_probs)

    # 5. Predecir Test
    print("  > Inferenciando Test...")
    fold_test_probs = []
    for text in tqdm(df_test["text"], leave=False):
        p = predict_sliding_window(text, model, tokenizer, is_robertuito)
        fold_test_probs.append(p)
    test_preds_matrix.append(fold_test_probs)

    del model, tokenizer
    torch.cuda.empty_cache()

# --- ENSEMBLE Y UMBRAL ---
if not val_preds_matrix:
    print("\n Error: No se procesó ningún modelo. Revisa las rutas.")
    sys.exit()

print("\n--- CALCULANDO ENSEMBLE ---")

if USE_STACKING:
    # STACKING: Usar meta-modelo entrenado
    print("Método: STACKING (Logistic Regression)")

    # Convertir lista de arrays a formato correcto
    # val_preds_matrix: [N_models, N_samples, 2] -> [N_samples, N_models*2]
    val_preds_array = np.array(val_preds_matrix)  # (N_models, N_samples, 2)
    test_preds_array = np.array(test_preds_matrix)

    # Reorganizar a (N_samples, N_models, 2)
    val_preds_array = np.transpose(val_preds_array, (1, 0, 2))
    test_preds_array = np.transpose(test_preds_array, (1, 0, 2))

    # Aplanar a (N_samples, N_models*2)
    X_val_meta = val_preds_array.reshape(val_preds_array.shape[0], -1)
    X_test_meta = test_preds_array.reshape(test_preds_array.shape[0], -1)

    # Predecir con meta-modelo
    avg_val_probs = stacker.predict_proba(X_val_meta)[:, 1]  # Solo prob clase positiva
    avg_test_probs = stacker.predict_proba(X_test_meta)[:, 1]

    print(f" Ensemble calculado con meta-modelo")

else:
    # PROMEDIO SIMPLE: Fallback si no hay stacker
    print("Método: PROMEDIO SIMPLE (Soft Voting)")

    # Convertir a array y promediar
    val_preds_array = np.array(val_preds_matrix)  # (N_models, N_samples, 2)
    test_preds_array = np.array(test_preds_matrix)

    # Promedio de probabilidades: solo clase positiva (columna 1)
    avg_val_probs = np.mean(val_preds_array[:, :, 1], axis=0)
    avg_test_probs = np.mean(test_preds_array[:, :, 1], axis=0)

    print(f" Ensemble calculado con promedio simple")

# Buscar mejor umbral
print("Optimizando umbral en Validación...")
best_th = 0.5
best_f1 = 0.0
true_labels = df_val["label"].values

# Barrido de umbral
for th in np.arange(0.15, 0.85, 0.01):
    preds = (avg_val_probs >= th).astype(int)
    score = f1_score(true_labels, preds, average="macro")
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f" MEJOR UMBRAL: {best_th:.2f}")
print(f" F1-Macro Validación (Ensemble): {best_f1:.4f}")
print(f"   Método usado: {'STACKING' if USE_STACKING else 'PROMEDIO SIMPLE'}")

# --- GENERAR CSV FINAL ---
final_preds = (avg_test_probs >= best_th).astype(int)

submission = pd.DataFrame(
    {
        "id": df_test["id"] if "id" in df_test.columns else df_test.index,
        "label": final_preds,
    }
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
submission.to_csv(OUTPUT_FILE, index=False)
print(f"\n Archivo generado exitosamente: {OUTPUT_FILE}")
