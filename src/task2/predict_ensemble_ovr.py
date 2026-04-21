"""
Paso 3: Generar la predicción final.
Aplica los thresholds exactos leídos de los JSON y promedia las predicciones de los modelos.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS

from bert_pooling import load_bert_like_classifier, predict_with_chunks


def get_ovr_probs(model_name, label, test_ds, device, augment=False):
    cfg = MODEL_CONFIGS[model_name]
    suffix = "_augmented" if augment else ""
    model_path = f"../../models/task2/final_OVR/{model_name}{suffix}_{label}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_bert_like_classifier(model_path, device)

    pred_output = predict_with_chunks(
        test_ds, tokenizer, model, device, cfg.max_len, aggregation="max"
    )
    probs = torch.sigmoid(torch.tensor(pred_output.predictions)).numpy().flatten()

    del model, tokenizer
    torch.cuda.empty_cache()
    return probs


def main(augment=False):
    models_to_ensemble = ["Robertuito", "BETO", "XLM-R"]

    # Cargar los thresholds descubiertos para cada modelo
    model_thresholds = {}
    for model_name in models_to_ensemble:
        suffix = "_augmented" if augment else ""
        json_path = f"../../models/task2/final_OVR/{model_name}{suffix}_ovr_params.json"
        with open(json_path, "r") as f:
            model_thresholds[model_name] = json.load(f)

    df = pd.read_csv("../../data/processed_test.csv")
    test_ds = Dataset.from_pandas(
        df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    all_preds = {}
    labels_map = {"sexualization": "S", "violence": "V", "hate": "H"}

    for label_full, label_short in labels_map.items():
        print(f"\nProcesando OVR Ensemble para: {label_full.upper()}")

        # Guardaremos preds binarias de cada modelo
        binary_votes = []

        for model_name in models_to_ensemble:
            probs = get_ovr_probs(
                model_name, label_full, test_ds, device, augment=augment
            )

            # Aplicar el threshold específico de ESTE modelo para ESTA etiqueta
            thr = model_thresholds[model_name][label_full]["best_threshold"]
            binary_pred = (probs >= thr).astype(int)
            binary_votes.append(binary_pred)
            print(f" -> {model_name} votó. (Threshold usado: {thr})")

        # Votación por Mayoría (Hard Voting). Ej: Si 2 de 3 dicen "1", es "1".
        # Hard voting suele ser mejor en OVR porque los espacios vectoriales de los modelos son muy distintos.
        sum_votes = np.sum(binary_votes, axis=0)
        final_preds = (sum_votes >= 2).astype(int)

        all_preds[label_short] = final_preds

    # Guardar
    for short_col in ["S", "V", "H"]:
        df[short_col] = all_preds[short_col]

    if "song_id" in df.columns:
        df_out = df[["song_id", "S", "V", "H"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "S", "V", "H"]]
    else:
        ids = [f"T2_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["S", "V", "H"]]
        df_out.insert(0, "id", ids)  # Insertar la columna 'id' al inicio

    df_out.to_csv("../../task_2_predictions.csv", index=False)
    print("\n¡Archivo final submission_OVR_ultimate.csv guardado con éxito!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OVR Ensemble Prediction for Task 2")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use augmented models trained with augmented data",
    )
    args = parser.parse_args()
    main(augment=args.augment)
