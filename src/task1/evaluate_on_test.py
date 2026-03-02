"""
Inferencia y evaluación en test con los modelos entrenados.
Genera una tabla comparativa con métricas por modelo.
"""

import os
import sys
import gc
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED

SEED = DEFAULT_SEED
set_seed(SEED)

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

TEST_LYRICS_PATH  = "../../data/task1/processed_test.csv"
TEST_LABELS_PATH  = "../../data/task1/test_labels.csv"
MODELS_DIR        = "../../models/task1/comparison"
RESULTS_FILE      = "../../results/task1/tabla_paper_test.csv"
BATCH_SIZE        = 32
ID2LABEL          = {0: "NM", 1: "M"}

# max_len y pysentimiento_preprocess por modelo
# Deben coincidir con lo usado en entrenamiento
MODEL_INFERENCE_CFG: dict[str, dict] = {
    "DistilBETO": {"max_len": 512, "use_pysentimiento_preprocess": False},
    "BETO":       {"max_len": 512, "use_pysentimiento_preprocess": False},
    "MarIA":      {"max_len": 512, "use_pysentimiento_preprocess": False},
    "XLM-R":      {"max_len": 512, "use_pysentimiento_preprocess": False},
    "Robertuito": {"max_len": 128, "use_pysentimiento_preprocess": True},
}

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

print(f"Cargando test desde {TEST_LYRICS_PATH} y {TEST_LABELS_PATH}...")
df_lyrics = pd.read_csv(TEST_LYRICS_PATH)
df_labels = pd.read_csv(TEST_LABELS_PATH)

df = df_lyrics.merge(df_labels, on="id")
df["label"] = df["label"].map({"NM": 0, "M": 1})
df = df.rename(columns={"lyrics": "text"})
df["text"] = df["text"].fillna("").astype(str)

print(f"Test cargado: {len(df)} canciones")
print(f"Distribución test:\n{df['label'].value_counts()}")

true_labels = df["label"].tolist()

# ─────────────────────────────────────────────────────────────
# FUNCIÓN DE INFERENCIA
# ─────────────────────────────────────────────────────────────

def run_inference(
    model_path: str,
    texts: list[str],
    true_labels: list[int],
    max_len: int,
    use_pysentimiento_preprocess: bool,
    device: torch.device,
) -> dict:
    """
    Carga un modelo guardado, ejecuta inferencia en batch y devuelve métricas.
    """
    print(f"  Cargando tokenizador y modelo desde: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)

    # Verificar arquitectura para log
    arch = type(AutoConfig.from_pretrained(model_path)).__name__
    print(f"  Arquitectura: {arch} | max_len={max_len}")

    # Preprocesar textos si es necesario
    if use_pysentimiento_preprocess:
        texts = [preprocess_tweet(t, lang="es") for t in texts]

    # Tokenizar todo de golpe
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Crear DataLoader para inferencia en batches
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        # token_type_ids solo existe en BERT, no en DistilBERT/RoBERTa
        *(
            [encodings["token_type_ids"]]
            if "token_type_ids" in encodings
            else []
        ),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inferencia", leave=False):
            input_ids      = batch[0].to(device)
            attention_mask = batch[1].to(device)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if len(batch) == 3:
                kwargs["token_type_ids"] = batch[2].to(device)

            outputs = model(**kwargs)
            probs   = torch.softmax(outputs.logits, dim=-1)
            preds   = probs.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # prob clase M

    # Métricas
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, all_preds, average="macro", zero_division=0.0
    )
    acc = accuracy_score(true_labels, all_preds)
    cm  = confusion_matrix(true_labels, all_preds)

    print(f"\n  {classification_report(true_labels, all_preds, target_names=['NM','M'])}")
    print(f"  Confusion matrix:\n{cm}\n")

    return {
        "F1-Macro":  round(f1, 4),
        "Accuracy":  round(acc, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "preds":     all_preds,
        "probs_M":   all_probs,
    }

# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────

results_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- INICIANDO INFERENCIA EN TEST ({device}) ---\n")

for name, inf_cfg in MODEL_INFERENCE_CFG.items():
    model_path = os.path.join(MODELS_DIR, name)

    print(f"\n{'='*50}")
    print(f">>> Modelo: {name}")
    print(f"{'='*50}")

    if not Path(model_path).exists():
        print(f"  ⚠ Modelo no encontrado en {model_path}, saltando.")
        results_list.append({
            "Modelo": name, "F1-Macro": None,
            "Accuracy": None, "Precision": None, "Recall": None,
        })
        continue

    try:
        metrics = run_inference(
            model_path=model_path,
            texts=df["text"].tolist(),
            true_labels=true_labels,
            max_len=inf_cfg["max_len"],
            use_pysentimiento_preprocess=inf_cfg["use_pysentimiento_preprocess"],
            device=device,
        )

        # Guardar predicciones individuales por modelo
        pred_path = os.path.join(MODELS_DIR, name, "test_predictions.csv")
        df_preds = df[["id", "label"]].copy()
        df_preds["pred"]   = metrics.pop("preds")
        df_preds["prob_M"] = metrics.pop("probs_M")
        df_preds["pred_label"]  = df_preds["pred"].map(ID2LABEL)
        df_preds["true_label"]  = df_preds["label"].map(ID2LABEL)
        df_preds["correct"] = df_preds["pred"] == df_preds["label"]
        df_preds.to_csv(pred_path, index=False)
        print(f"  Predicciones guardadas en: {pred_path}")

        results_list.append({"Modelo": name, **metrics})
        print(f"  ✓ {name}: F1={metrics['F1-Macro']:.4f} | Acc={metrics['Accuracy']:.4f}")

    except Exception as e:
        print(f"  ✗ Error con {name}: {e}")
        import traceback; traceback.print_exc()
        results_list.append({
            "Modelo": name, "F1-Macro": None,
            "Accuracy": None, "Precision": None, "Recall": None,
        })

    finally:
        gc.collect()
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────────────────────

df_res = pd.DataFrame(results_list).sort_values("F1-Macro", ascending=False)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df_res.to_csv(RESULTS_FILE, index=False)

print(f"\n{'='*60}")
print("RESULTADOS EN TEST")
print(f"{'='*60}")
print(df_res.to_markdown(index=False))
print(f"\nGuardado en: {RESULTS_FILE}")