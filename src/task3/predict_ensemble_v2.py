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

# ---------------------------------------------------------------------------
# Validation F1-Macro scores for each (model, augment, processed) combination.
# processed=True  -> model trained on processed data -> use processed_test.csv
# processed=False -> model trained on raw data (baseline) -> use test.csv
# ---------------------------------------------------------------------------
MODEL_REGISTRY = [
    # (model_name,   augment, processed, val_f1_macro, best_threshold)
    ("BETO", True, True, 0.7903, 0.53),
    ("Robertuito", True, True, 0.7879, 0.51),
    ("DistilBETO", True, True, 0.7874, 0.30),
    ("LongFormer", False, False, 0.7846, 0.51),  # baseline: raw data
    ("XLM-R", False, False, 0.7850, 0.42),  # baseline: raw data
    # MarIA is the weakest; uncomment to include in a 6-model ensemble
    # ("MarIA",      False,   False,     0.7609,       0.87),
]


def get_model_probs(model_name, test_ds, device, augment=False):
    suffix = "_augmented" if augment else ""
    model_path = f"../../models/task3/final/{model_name}{suffix}"
    aug_label = "augmented" if augment else "baseline"
    print(f"\n--- Generating predictions: {model_name} ({aug_label}) ---")

    cfg = MODEL_CONFIGS[model_name]
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

    return probs


def make_ds(path):
    return Dataset.from_pandas(
        pd.read_csv(path).rename(columns={"lyrics": "text"}),
        preserve_index=False,
    )


def main(voting_type="soft"):
    output_file = "../../task_3_predictions.csv"

    print("Loading test datasets...")
    # Keep processed df around just for the id/song_id column in the output
    df = pd.read_csv("../../data/processed_test.csv")

    # Load both variants: baseline models need raw text to avoid train/test mismatch
    test_ds_processed = make_ds("../../data/processed_test.csv")
    test_ds_raw = make_ds("../../data/test.csv")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Collect per-model probabilities and metadata
    # ------------------------------------------------------------------
    all_probs = []
    all_weights = []
    all_thresholds = []

    for model_name, augment, processed, val_f1, threshold in MODEL_REGISTRY:
        test_ds = test_ds_processed if processed else test_ds_raw
        probs = get_model_probs(model_name, test_ds, device, augment=augment)
        all_probs.append(probs)
        all_weights.append(val_f1)
        all_thresholds.append(threshold)

    all_probs = np.stack(all_probs, axis=0)  # shape: (n_models, n_samples)
    all_weights = np.array(all_weights)
    all_thresholds = np.array(all_thresholds)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------
    if voting_type == "soft":
        normalized_weights = all_weights / all_weights.sum()

        print("\nModel weights (normalized):")
        for (name, aug, proc, f1, thr), w in zip(MODEL_REGISTRY, normalized_weights):
            data_label = "processed" if proc else "raw"
            aug_label = "aug" if aug else "baseline"
            print(
                f"  {name} ({aug_label}, {data_label}): weight={w:.4f}  val_f1={f1:.4f}  thr={thr}"
            )

        final_probs = (all_probs * normalized_weights[:, None]).sum(axis=0)

        # Blended threshold: weighted average of individual best thresholds
        blended_threshold = float((all_thresholds * normalized_weights).sum())
        print(
            f"\nBlended threshold (weighted avg of individual bests): {blended_threshold:.4f}"
        )

        preds = (final_probs >= blended_threshold).astype(int)
        print(f"Using Soft Voting | threshold={blended_threshold:.4f}")

    else:  # hard voting
        votes = np.stack(
            [
                (all_probs[i] >= all_thresholds[i]).astype(int)
                for i in range(len(MODEL_REGISTRY))
            ],
            axis=0,
        )  # shape: (n_models, n_samples)

        n_models = len(MODEL_REGISTRY)
        majority = n_models // 2 + 1
        sum_votes = votes.sum(axis=0)
        preds = (sum_votes >= majority).astype(int)
        print(f"\nUsing Hard Voting | majority >= {majority}/{n_models}")

    # ------------------------------------------------------------------
    # Map predictions and save
    # ------------------------------------------------------------------
    label_map = {0: "N", 1: "Y"}
    df["preds"] = [label_map[p] for p in preds]

    if "song_id" in df.columns:
        df_out = df[["song_id", "preds"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "preds"]]
    else:
        df["id"] = [f"T3_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "preds"]]

    df_out.to_csv(output_file, index=False)
    print(f"\nEnsemble predictions saved to {output_file}!")

    n_positive = preds.sum()
    n_total = len(preds)
    print(f"Predicted Y: {n_positive}/{n_total} ({100 * n_positive / n_total:.1f}%)")
    print(f"Predicted N: {n_total - n_positive}/{n_total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mixed ensemble prediction for Task 3 (per-model best training regime)"
    )
    parser.add_argument(
        "--voting",
        type=str,
        choices=["soft", "hard"],
        default="soft",
        help="Voting method: 'soft' (weighted avg probabilities) or 'hard' (majority voting)",
    )
    args = parser.parse_args()
    main(voting_type=args.voting)
