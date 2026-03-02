"""
Inferencia del Ensemble: Promedia los 5 folds de cada modelo
y usa el meta-modelo stacker para la predicción final.
"""
import os
import sys
import torch
import gc
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED

SEED = DEFAULT_SEED
set_seed(SEED)

MODELS_DIR = "../../models/task1/ensemble"
STACKER_PATH = "../../models/task1/ensemble/stacker.pkl"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_predictions_batched(texts, model, tokenizer, max_len=512, batch_size=BATCH_SIZE):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_len,
        ).to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def predict_ensemble(texts: list[str], stacker_path: str = STACKER_PATH) -> np.ndarray:
    """
    Genera predicciones del ensemble completo.
    
    Returns:
        np.ndarray: Labels predichas (0 o 1)
    """
    # Cargar meta-modelo
    stacker_data = joblib.load(stacker_path)
    stacker = stacker_data["model"]
    model_names = stacker_data["models_used"]
    model_ids = stacker_data["model_ids"]
    n_folds = stacker_data["n_folds"]
    max_len = stacker_data["max_len"]

    n_models = len(model_names)
    X_meta = np.zeros((len(texts), n_models * 2))

    for m_idx, model_name in enumerate(model_names):
        print(f">>> Inferencia: {model_name}")
        fold_probs_list = []

        for fold in range(n_folds):
            ckpt_path = os.path.join(MODELS_DIR, model_name, f"fold_{fold}")
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

            probs = get_predictions_batched(texts, model, tokenizer, max_len)
            fold_probs_list.append(probs)

            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        # Promedio de los N folds
        avg_probs = np.mean(fold_probs_list, axis=0)
        X_meta[:, m_idx * 2 : m_idx * 2 + 2] = avg_probs

    # Meta-modelo predice
    predictions = stacker.predict(X_meta)
    probabilities = stacker.predict_proba(X_meta)

    return predictions, probabilities


if __name__ == "__main__":
    # Ejemplo: evaluar en test
    TEST_FILE = "../../data/task1/processed_test.csv"

    df = pd.read_csv(TEST_FILE)
    texts = df["lyrics"].tolist()

    preds, probs = predict_ensemble(texts)

    if "label" in df.columns:
        y_true = df["label"].map({"NM": 0, "M": 1}) if df["label"].dtype == object else df["label"]
        f1 = f1_score(y_true, preds, average="macro")
        print(f"\n{'='*60}")
        print(f"ENSEMBLE F1-Macro: {f1:.4f}")
        print(f"{'='*60}")
        print(classification_report(y_true, preds, target_names=["NM", "M"]))

    # Guardar predicciones
    df["prediction"] = preds
    df["prob_M"] = probs[:, 1]
    output_path = "../../results/task1/ensemble_predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")