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


def load_threshold(model_path: str) -> float:
    results_path = os.path.join(model_path, "results.json")
    if not os.path.exists(results_path):
        return 0.5

    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        for key in ["Best_Threshold", "Best-Threshold", "best_threshold"]:
            if key in data:
                return float(data[key])
    except Exception:
        return 0.5

    return 0.5


def get_model_probs(model_name, test_ds, device, augment=False):
    print(f"\n--- Generando predicciones para {model_name} ---")
    cfg = MODEL_CONFIGS[model_name]
    suffix = "_augmented" if augment else ""
    model_path = f"../../models/task3/final/{model_name}{suffix}"

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

    del model, tokenizer
    torch.cuda.empty_cache()
    return probs, load_threshold(model_path)


def main(augment=False, voting_type="hard"):
    test_file = "../../data/processed_test.csv"
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

    # Top-3 ensemble: Robertuito + XLM-R + LongFormer
    probs_robertuito, thr_robertuito = get_model_probs(
        "Robertuito", test_ds, device, augment=augment
    )
    probs_xlmr, thr_xlmr = get_model_probs("XLM-R", test_ds, device, augment=augment)
    probs_longformer, thr_longformer = get_model_probs(
        "LongFormer", test_ds, device, augment=augment
    )

    if voting_type == "soft":
        # Average the probabilities (Soft Voting Ensemble)
        final_probs = (probs_robertuito + probs_xlmr + probs_longformer) / 3.0

        # Average optimized validation thresholds
        blended_threshold = float(np.mean([thr_robertuito, thr_xlmr, thr_longformer]))
        preds = (final_probs >= blended_threshold).astype(int)
        print(f"\nUsing Soft Voting with threshold {blended_threshold:.3f}")
    else:  # hard voting
        votes_robertuito = (probs_robertuito >= thr_robertuito).astype(int)
        votes_xlmr = (probs_xlmr >= thr_xlmr).astype(int)
        votes_longformer = (probs_longformer >= thr_longformer).astype(int)

        # Hard Voting (Majority voting)
        sum_votes = votes_robertuito + votes_xlmr + votes_longformer
        preds = (sum_votes >= 2).astype(int)
        print(f"\nUsing Hard Voting with majority threshold (>= 2 out of 3)")

    # 4. Mapear de vuelta a N / Y (Asegúrate de que Task 3 pide N y Y)
    label_map = {0: "N", 1: "Y"}
    df["preds"] = [label_map[p] for p in preds]

    # 5. Save submission
    if "song_id" in df.columns:
        df_out = df[["song_id", "preds"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "preds"]]
    else:
        df["id"] = [f"T3_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "preds"]]

    df_out.to_csv(output_file, index=False)
    print(f"\n¡Predicciones del ensemble guardadas exitosamente en {output_file}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Prediction for Task 3")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use augmented models trained with augmented data",
    )
    parser.add_argument(
        "--voting",
        type=str,
        choices=["soft", "hard"],
        default="hard",
        help="Voting method: 'soft' (average probabilities) or 'hard' (majority voting)",
    )
    args = parser.parse_args()
    main(augment=args.augment, voting_type=args.voting)
