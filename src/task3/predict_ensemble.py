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


def get_binary_vote(model_name, threshold, test_ds, device):
    print(f"\n--- Generando votos para {model_name} (Threshold: {threshold}) ---")
    cfg = MODEL_CONFIGS[model_name]
    model_path = f"../../models/task3/final/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_bert_like_classifier(model_path, device)

    pred_output = predict_with_chunks(
        dataset=test_ds,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=cfg.max_len,
        batch_size=8,
        aggregation="max",
    )

    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # El modelo emite su voto binario (0 o 1) basado en SU propio umbral óptimo
    votes = (probs >= threshold).astype(int)

    del model, tokenizer
    torch.cuda.empty_cache()
    return votes


def main():
    test_file = "../../data/test.csv"
    output_file = "../../task_3_predictions.csv"

    print(f"Cargando datos de prueba desde {test_file}...")
    df = pd.read_csv(test_file)
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

    # 1. Obtener los votos de los 3 mejores modelos con sus umbrales óptimos
    votes_robertuito = get_binary_vote("Robertuito", 0.49, test_ds, device)
    votes_xlmr = get_binary_vote("XLM-R", 0.42, test_ds, device)
    votes_longformer = get_binary_vote("LongFormer", 0.51, test_ds, device)

    # 2. Hard Voting (Votación por Mayoría)
    # Sumamos los votos. El rango será de 0 a 3.
    # Si la suma es >= 2, la mayoría votó "1".
    sum_votes = votes_robertuito + votes_xlmr + votes_longformer
    final_preds = (sum_votes >= 2).astype(int)

    # 3. Mapear de vuelta a N / Y (Asegúrate de que Task 3 pide N y Y)
    label_map = {0: "N", 1: "Y"}
    df["preds"] = [label_map[p] for p in final_preds]

    # 4. Save submission
    if "song_id" in df.columns:
        df_out = df[["song_id", "preds"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "preds"]]
    else:
        df["id"] = [f"T3_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "preds"]]

    df_out.to_csv(output_file, index=False)
    print(f"\n¡Predicciones Hard-Voting guardadas exitosamente en {output_file}!")


if __name__ == "__main__":
    main()
