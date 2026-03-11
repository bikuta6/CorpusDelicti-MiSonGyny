"""
Inferencia y evaluación en test con los modelos entrenados.
Genera una tabla comparativa con métricas por modelo.
"""

import gc
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS, ModelConfig

from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED
set_seed(SEED)

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

TEST_LYRICS_PATH = "../../data/task1/processed_test.csv"
TEST_LABELS_PATH = "../../data/task1/test_labels.csv"
MODELS_DIR = "../../models/task1/comparison"
RESULTS_FILE = "../../results/task1/tabla_paper_test.csv"
BEST_THRESHOLDS_PATH = "../../results/task1/tabla_paper.csv"
BATCH_SIZE = 32
ID2LABEL = {0: "NM", 1: "M"}
CHUNK_STRIDE = 256  # Overlap between chunks

# max_len y pysentimiento_preprocess por modelo
# Deben coincidir con lo usado en entrenamiento
MODEL_INFERENCE_CFG: dict[str, ModelConfig] = MODEL_CONFIGS
if Path(BEST_THRESHOLDS_PATH).exists():
    df_thr = pd.read_csv(BEST_THRESHOLDS_PATH)
    best_thresholds = dict(zip(df_thr["Modelo"], df_thr["Best-Threshold"]))
    print(f"Cargados thresholds óptimos por modelo:\n{best_thresholds}")
else:
    print(f"⚠ No se encontró {BEST_THRESHOLDS_PATH}, se usarán 0.5 por defecto")
    best_thresholds = {}

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
# CHUNKING HELPERS
# ─────────────────────────────────────────────────────────────


def chunk_tokens(token_ids: list[int], max_len: int, stride: int) -> list[list[int]]:
    """
    Split token_ids into overlapping chunks.
    Each chunk includes CLS at start and SEP at end.
    """
    if len(token_ids) <= max_len:
        return [token_ids]

    cls_token = token_ids[0]
    sep_token = token_ids[-1]
    content = token_ids[1:-1]  # Remove CLS and SEP

    chunks = []
    content_max = max_len - 2  # Reserve space for CLS and SEP

    for start in range(0, len(content), stride):
        chunk_content = content[start : start + content_max]
        chunk = [cls_token] + chunk_content + [sep_token]
        chunks.append(chunk)

        # Stop if we've covered all content
        if start + content_max >= len(content):
            break

    return chunks


def get_chunked_probability(
    text: str,
    tokenizer,
    model,
    max_len: int,
    stride: int,
    device: torch.device,
    use_pysentimiento_preprocess: bool = False,
) -> float:
    """
    Get probability for class M using sliding window with max pooling.
    Returns single probability (max across all chunks).
    """
    if use_pysentimiento_preprocess:
        text = preprocess_tweet(text, lang="es")

    # Tokenize without truncation
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    token_ids = encoding["input_ids"]

    # Create chunks
    chunks = chunk_tokens(token_ids, max_len, stride)

    # Process all chunks
    chunk_probs = []
    for chunk in chunks:
        # Pad to max_len
        pad_len = max_len - len(chunk)
        input_ids = chunk + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(chunk) + [0] * pad_len

        # Create tensors
        input_ids_tensor = torch.tensor([input_ids], device=device)
        attention_mask_tensor = torch.tensor([attention_mask], device=device)

        # Inference
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            prob_M = probs[0, 1].item()
            chunk_probs.append(prob_M)

    # Max pooling across chunks
    return max(chunk_probs)


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
    threshold: float = 0.5,
) -> dict:
    """
    Carga un modelo guardado, ejecuta inferencia con sliding window chunking y devuelve métricas.
    """
    print(f"  Cargando tokenizador y modelo desde: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)

    # Verificar arquitectura para log
    arch = type(AutoConfig.from_pretrained(model_path)).__name__
    print(
        f"  Arquitectura: {arch} | max_len={max_len} | chunking enabled with stride={CHUNK_STRIDE}"
    )

    all_preds = []
    all_probs = []

    # Process each text individually with chunking
    for text in tqdm(texts, desc="  Inferencia con chunking", leave=False):
        prob_M = get_chunked_probability(
            text=text,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            stride=CHUNK_STRIDE,
            device=device,
            use_pysentimiento_preprocess=use_pysentimiento_preprocess,
        )

        pred = 1 if prob_M >= threshold else 0
        all_preds.append(pred)
        all_probs.append(prob_M)

    # Métricas
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, all_preds, average="macro", zero_division=0.0
    )
    acc = accuracy_score(true_labels, all_preds)
    cm = confusion_matrix(true_labels, all_preds)

    print(
        f"\n  {classification_report(true_labels, all_preds, target_names=['NM', 'M'])}"
    )
    print(f"  Confusion matrix:\n{cm}\n")

    return {
        "F1-Macro": round(f1, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "preds": all_preds,
        "probs_M": all_probs,
    }


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────

results_list = []
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n--- INICIANDO INFERENCIA EN TEST ({device}) ---\n")

for name, inf_cfg in MODEL_INFERENCE_CFG.items():
    model_path = os.path.join(MODELS_DIR, name)

    print(f"\n{'=' * 50}")
    print(f">>> Modelo: {name}")
    print(f"{'=' * 50}")

    if not Path(model_path).exists():
        print(f"  ⚠ Modelo no encontrado en {model_path}, saltando.")
        results_list.append(
            {
                "Modelo": name,
                "F1-Macro": None,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
            }
        )
        continue

    try:
        metrics = run_inference(
            model_path=model_path,
            texts=df["text"].tolist(),
            true_labels=true_labels,
            max_len=inf_cfg.max_len,
            use_pysentimiento_preprocess=inf_cfg.use_pysentimiento_preprocess,
            threshold=best_thresholds.get(name, 0.5),
            device=device,
        )

        # Guardar predicciones individuales por modelo
        pred_path = os.path.join(MODELS_DIR, name, "test_predictions.csv")
        df_preds = df[["id", "label"]].copy()
        df_preds["pred"] = metrics.pop("preds")
        df_preds["prob_M"] = metrics.pop("probs_M")
        df_preds["pred_label"] = df_preds["pred"].map(ID2LABEL)
        df_preds["true_label"] = df_preds["label"].map(ID2LABEL)
        df_preds["correct"] = df_preds["pred"] == df_preds["label"]
        df_preds.to_csv(pred_path, index=False)
        print(f"  Predicciones guardadas en: {pred_path}")

        results_list.append({"Modelo": name, **metrics})
        print(
            f"  ✓ {name}: F1={metrics['F1-Macro']:.4f} | Acc={metrics['Accuracy']:.4f}"
        )

    except Exception as e:
        print(f"  ✗ Error con {name}: {e}")
        import traceback

        traceback.print_exc()
        results_list.append(
            {
                "Modelo": name,
                "F1-Macro": None,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
            }
        )

    finally:
        gc.collect()
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────────────────────

df_res = pd.DataFrame(results_list).sort_values("F1-Macro", ascending=False)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df_res.to_csv(RESULTS_FILE, index=False)

print(f"\n{'=' * 60}")
print("RESULTADOS EN TEST")
print(f"{'=' * 60}")
print(df_res.to_markdown(index=False))
print(f"\nGuardado en: {RESULTS_FILE}")
