"""
Ensemble Stacking: Meta-modelo con predicciones Out-of-Fold.
Combina DistilBETO + XLM-R + MarIA usando Regresión Logística.
"""
import os
import sys
import torch
import gc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import joblib
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED

SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
MODELS_DIR = "../../models/task1/ensemble"
TRAIN_FILE = "../../data/task1/processed_train.csv"
STACKER_OUTPUT = "../../models/task1/ensemble/stacker.pkl"
MAX_LEN = 512
N_FOLDS = 3
BATCH_SIZE = 32  # Para inferencia batched

MODEL_MAP = {
    "DistilBETO": "dccuchile/distilbert-base-spanish-uncased",
    "Robertuito": "pysentimiento/robertuito-base-uncased",
    "MarIA": "IsGarrido/roberta-base-bne",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_name: str, fold: int):
    """Carga modelo entrenado de un fold específico."""
    ckpt_path = os.path.join(MODELS_DIR, model_name, f"fold_{fold}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def get_predictions_batched(
    texts: list[str],
    model,
    tokenizer,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Inferencia batched eficiente. Devuelve (N, 2) con probabilidades.
    """
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # if robertuito, aplicar preprocesamiento específico
        if "robertuito" in model.config._name_or_path.lower():
            batch_texts = [preprocess_tweet(t) for t in batch_texts]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)


def train_stacker():
    """
    Entrena meta-modelo con predicciones Out-of-Fold (OOF).
    Evalúa con CV interno sobre las propias predicciones OOF.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DE META-MODELO (STACKING OOF)")
    print("=" * 60)

    # 1. Cargar datos
    df = pd.read_csv(TRAIN_FILE)
    df["label"] = df["label"].map({"NM": 0, "M": 1})
    texts = df["lyrics"].tolist()
    y = df["label"].values

    # 2. Generar predicciones OOF
    n_models = len(MODEL_MAP)
    X_oof = np.zeros((len(df), n_models * 2))  # 2 probs por modelo

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for m_idx, (model_name, _) in enumerate(MODEL_MAP.items()):
        print(f"\n>>> Generando OOF para: {model_name}")

        for fold, (_, val_idx) in enumerate(skf.split(texts, y)):
            print(f"   Fold {fold + 1}/{N_FOLDS}...", end=" ")

            try:
                model, tokenizer = load_trained_model(model_name, fold)
            except FileNotFoundError as e:
                print(f"SKIP ({e})")
                continue

            # Textos de validación de este fold
            val_texts = [texts[i] for i in val_idx]
            max_len = MAX_LEN if model_name != "Robertuito" else 128  # Robertuito tiene max_len=128
            # Predicciones batched
            probs = get_predictions_batched(val_texts, model, tokenizer, max_len=max_len, batch_size=BATCH_SIZE)
            X_oof[val_idx, m_idx * 2 : m_idx * 2 + 2] = probs

            f1 = f1_score(y[val_idx], probs.argmax(axis=1), average="macro")
            print(f"F1={f1:.4f}")

            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    # 3. Verificar que no hay filas sin predicciones
    zero_rows = np.all(X_oof == 0, axis=1).sum()
    if zero_rows > 0:
        print(f"\n⚠️  {zero_rows} filas sin predicciones OOF")

    # 4. Entrenar meta-modelo
    print(f"\n{'─'*40}")
    print("Entrenando Logistic Regression...")
    stacker = LogisticRegression(
        class_weight="balanced",
        random_state=SEED,
        max_iter=1000,
        C=1.0,
    )
    stacker.fit(X_oof, y)

    # 5. Evaluación OOF del stacker
    oof_preds = stacker.predict(X_oof)
    oof_f1 = f1_score(y, oof_preds, average="macro")

    print(f"\n{'='*60}")
    print(f"RESULTADO STACKING OOF")
    print(f"F1-Macro OOF: {oof_f1:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report (OOF):")
    print(classification_report(y, oof_preds, target_names=["NM", "M"]))

    # 6. Mostrar pesos aprendidos
    print("Pesos del meta-modelo:")
    for i, model_name in enumerate(MODEL_MAP.keys()):
        w_neg, w_pos = stacker.coef_[0][i * 2], stacker.coef_[0][i * 2 + 1]
        print(f"  {model_name}: w_NM={w_neg:.3f}, w_M={w_pos:.3f}")

    # 7. Guardar
    os.makedirs(os.path.dirname(STACKER_OUTPUT), exist_ok=True)
    stacker_data = {
        "model": stacker,
        "models_used": list(MODEL_MAP.keys()),
        "model_ids": MODEL_MAP,
        "oof_f1": oof_f1,
        "n_folds": N_FOLDS,
        "max_len": MAX_LEN,
    }
    joblib.dump(stacker_data, STACKER_OUTPUT)
    print(f"\nMeta-modelo guardado en: {STACKER_OUTPUT}")

    return stacker, oof_f1


if __name__ == "__main__":
    try:
        stacker, oof_f1 = train_stacker()
        print(f"\n✅ Stacking completado — F1-Macro OOF: {oof_f1:.4f}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
