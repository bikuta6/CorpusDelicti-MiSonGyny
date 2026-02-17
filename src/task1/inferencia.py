import os
import torch
import pandas as pd
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pysentimiento.preprocessing import preprocess_tweet
from tqdm import tqdm

# --- CONFIGURACIÓN ---
TEST_FILE = "../../data/task1/test.csv"
VAL_FILE = "../../data/task1/validation.csv"
MODELS_DIR = "../../models/task1/ensemble"
OUTPUT_FILE = "../../results/task1/submission.csv"

# Parámetros de Ventana Deslizante
WINDOW_LEN = 256  # Tamaño de ventana
STRIDE = 128  # Solapamiento

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- FUNCIÓN SLIDING WINDOW (Para canciones largas) ---
def predict_sliding_window(text, model, tokenizer, is_robertuito):
    if is_robertuito:
        text = preprocess_tweet(text, lang="es")

    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    total_tokens = len(tokens)

    # Caso corto: Cabe en una pasada
    if total_tokens <= WINDOW_LEN:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=WINDOW_LEN,
            padding="max_length",
        ).to(device)
        with torch.no_grad():
            probs = softmax(model(**inputs).logits, dim=1)
        return probs[0][1].item()

    # Caso largo: Ventana Deslizante
    window_probs = []
    for i in range(0, total_tokens, STRIDE):
        chunk_ids = tokens[i : i + WINDOW_LEN]
        if len(chunk_ids) < 10:
            break  # Ignorar trozos muy pequeños al final

        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=WINDOW_LEN,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            probs = softmax(model(**inputs).logits, dim=1)
            window_probs.append(probs[0][1].item())

        if i + WINDOW_LEN >= total_tokens:
            break

    # Max Pooling: Si hay odio en una parte, hay odio en la canción
    return max(window_probs) if window_probs else 0.0


# --- CARGAR DATOS ---
df_test = pd.read_csv(TEST_FILE)
df_val = pd.read_csv(VAL_FILE)

# Matrices para guardar resultados: [N_Modelos, N_Samples]
val_preds_matrix = []
test_preds_matrix = []

# --- BUCLE DE INFERENCIA ---
model_folders = [f.path for f in os.scandir(MODELS_DIR) if f.is_dir()]
print(f"--- INICIANDO INFERENCIA CON {len(model_folders)} MODELOS ---")

for folder in model_folders:
    ckpt_dirs = [d for d in os.listdir(folder) if "checkpoint" in d]
    if not ckpt_dirs:
        continue
    model_path = os.path.join(folder, ckpt_dirs[0])

    print(f"Procesando: {os.path.basename(folder)}")
    is_robertuito = "Robertuito" in folder

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # 1. Predecir Validación (para umbral)
    print("  > Prediciendo Validación...")
    fold_val_probs = []
    for text in tqdm(df_val["text"]):
        p = predict_sliding_window(text, model, tokenizer, is_robertuito)
        fold_val_probs.append(p)
    val_preds_matrix.append(fold_val_probs)

    # 2. Predecir Test (para entrega)
    print("  > Prediciendo Test...")
    fold_test_probs = []
    for text in tqdm(df_test["text"]):
        p = predict_sliding_window(text, model, tokenizer, is_robertuito)
        fold_test_probs.append(p)
    test_preds_matrix.append(fold_test_probs)

    del model, tokenizer
    torch.cuda.empty_cache()

# --- ENSEMBLE Y UMBRAL ---
print("\nCalculando Ensemble...")
# Promedio de todos los modelos
avg_val_probs = np.mean(val_preds_matrix, axis=0)  # Probabilidad final Validación
avg_test_probs = np.mean(test_preds_matrix, axis=0)  # Probabilidad final Test

# Buscar mejor umbral
print("Optimizando umbral...")
best_th = 0.5
best_f1 = 0.0
true_labels = df_val["label"].values

for th in np.arange(0.15, 0.85, 0.01):
    preds = (avg_val_probs >= th).astype(int)
    score = f1_score(true_labels, preds, average="macro")
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f"¡MEJOR UMBRAL: {best_th:.2f} (F1 Val: {best_f1:.4f})!")

# --- GENERAR CSV FINAL ---
final_preds = (avg_test_probs >= best_th).astype(int)

submission = pd.DataFrame(
    {
        "id": df_test["id"] if "id" in df_test.columns else df_test.index,
        "label": final_preds,
    }
)
os.makedirs("../results", exist_ok=True)
submission.to_csv(OUTPUT_FILE, index=False)
print(f"Archivo generado: {OUTPUT_FILE}")
