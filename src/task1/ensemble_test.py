import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
from bert_model_configs import MODEL_CONFIGS, ModelConfig
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# seed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import DEFAULT_SEED, set_seed

set_seed(DEFAULT_SEED)


# Configuración del Ensemble
MODELS_TO_ENSEMBLE = [
    "LongFormer",
    "MarIA",
    "DistilBETO",
]  # Nombres en tu MODEL_CONFIGS
MODELS_BASE_DIR = "../../models/task1/comparison"
TEST_PATH = "../../data/task1/processed_test.csv"
RESULTS_FILE = "../../results/task1/submission_ensemble.csv"
BATCH_SIZE = 32
CHUNK_STRIDE = 256  # Overlap between chunks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# INFERENCE LOGIC
# ─────────────────────────────────────────────────────────────


def get_model_probabilities(name, df_test):
    cfg = MODEL_CONFIGS[name]
    model_path = os.path.join(MODELS_BASE_DIR, name)

    print(f"\n>> Inferencia con: {name} (chunking enabled, stride={CHUNK_STRIDE})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    texts = df_test["lyrics"].fillna("").astype(str).tolist()
    probs = []

    # Process each text with chunking
    for text in tqdm(texts, leave=False, desc=f"Chunked inference {name}"):
        prob_M = get_chunked_probability(
            text=text,
            tokenizer=tokenizer,
            model=model,
            max_len=cfg.max_len,
            stride=CHUNK_STRIDE,
            device=device,
            use_pysentimiento_preprocess=cfg.use_pysentimiento_preprocess,
        )
        probs.append(prob_M)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return np.array(probs)


def main():
    df_test = pd.read_csv(TEST_PATH)

    # 1. Definimos los modelos y sus pesos basados en su F1-Macro de Test
    # Pesos sugeridos según tus resultados: LongFormer(0.81), MarIA(0.79), DistilBETO(0.78)
    MODELS_TO_WEIGHT = {
        "LongFormer": 0.34,  # El mejor, le damos la mitad del voto
        "MarIA": 0.33,  # Muy estable en test
        "DistilBETO": 0.33,  # Un poco menos, pero ayuda a la diversidad
    }

    all_probs = []

    # Extraer probabilidades
    for name in MODELS_TO_WEIGHT.keys():
        p = get_model_probabilities(name, df_test)
        all_probs.append(p)

    # 2. Aplicar Ponderación y Power Averaging
    # Elevamos la probabilidad a una potencia (p=2 o p=3) para "premiar" la seguridad.
    # Esto hace que un 0.9 valga mucho más que dos 0.5.
    power = 1.5
    weighted_probs_sum = np.zeros(len(df_test))
    total_weight = sum(MODELS_TO_WEIGHT.values())

    for i, (name, weight) in enumerate(MODELS_TO_WEIGHT.items()):
        # Aplicamos el peso del modelo y la potencia de confianza
        weighted_probs_sum += (all_probs[i] ** power) * weight

    # Promediamos y devolvemos a la escala original (raíz de la potencia)
    avg_probs = (weighted_probs_sum / total_weight) ** (1 / power)

    # 3. Optimización del Threshold
    # Como el modelo tiende a ser conservador, un threshold ligeramente
    # más bajo que 0.5 suele dar mejor F1-Macro
    final_threshold = 0.5  # Valor sugerido basado en tus 'Best-Threshold'

    predictions = ["M" if p >= final_threshold else "NM" for p in avg_probs]

    # --- Guardado y Métricas ---
    df_test["label"] = predictions
    df_test[["id", "label"]].to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Submission guardada en: {RESULTS_FILE}")

    # Evaluación (si tienes las etiquetas)
    true_labels_df = pd.read_csv("../../data/task1/test_labels.csv")
    y_true = true_labels_df["label"].map({"NM": 0, "M": 1}).values
    y_pred = np.array([1 if p == "M" else 0 for p in predictions])

    print("\n" + "=" * 60)
    print("ENSEMBLE PONDERADO (POWER=1.5) - CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["NM", "M"]))
    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
