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
# Validation F1-Macro scores for each (model, augment) combination.
# Used to compute normalized weights for weighted soft voting.
# Update these if you retrain or add models.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = [
    # (model_name, augment, val_f1_macro)
    ("DistilBETO", False, 0.8583),  # non-aug is better for DistilBETO
    ("Robertuito", False, 0.8544),  # non-aug is better for Robertuito
    ("BETO", True, 0.8731),  # aug is better for BETO
    ("XLM-R", True, 0.8539),  # aug is better for XLM-R
    # Uncomment to add MarIA (aug better: 0.8394 vs 0.8247)
    # ("MarIA",    True,  0.8394),
]


def get_model_probs(model_name, test_ds, device, augment=False):
    suffix = "_augmented" if augment else ""
    model_path = f"../../models/task1/final/{model_name}{suffix}"
    aug_label = "augmented" if augment else "non-augmented"
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
        aggregation="max",  # MAX aggregation: correct for presence-of-misogyny task
    )

    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # Free GPU/CPU memory before loading next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return probs


def main(voting_type="soft"):
    test_file = "../../data/processed_test.csv"
    output_file = "../../task_1_predictions.csv"

    print(f"Loading test data from {test_file}...")
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
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Collect per-model probabilities and weights
    # ------------------------------------------------------------------
    all_probs = []
    all_weights = []

    for model_name, augment, val_f1 in MODEL_REGISTRY:
        probs = get_model_probs(model_name, test_ds, device, augment=augment)
        all_probs.append(probs)
        all_weights.append(val_f1)

    all_probs = np.stack(all_probs, axis=0)  # shape: (n_models, n_samples)
    all_weights = np.array(all_weights)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------
    if voting_type == "soft":
        # Weighted average of probabilities
        normalized_weights = all_weights / all_weights.sum()
        print("\nModel weights (normalized):")
        for (name, aug, _), w in zip(MODEL_REGISTRY, normalized_weights):
            print(f"  {name} ({'aug' if aug else 'non-aug'}): {w:.4f}")

        final_probs = (all_probs * normalized_weights[:, None]).sum(axis=0)

        # Threshold: weighted average of each model's best individual threshold
        individual_thresholds = {
            ("DistilBETO", False): 0.41,
            ("Robertuito", False): 0.51,
            ("BETO", True): 0.52,
            ("XLM-R", True): 0.30,
            # ("MarIA",    True):  0.55,
        }
        thresholds = np.array(
            [individual_thresholds[(name, aug)] for name, aug, _ in MODEL_REGISTRY]
        )
        blended_threshold = float((thresholds * normalized_weights).sum())
        print(
            f"\nBlended threshold (weighted avg of individual bests): {blended_threshold:.4f}"
        )

        preds = (final_probs >= blended_threshold).astype(int)
        print(f"Using Soft Voting | threshold={blended_threshold:.4f}")

    else:  # hard voting
        individual_thresholds = {
            ("DistilBETO", False): 0.41,
            ("Robertuito", False): 0.51,
            ("BETO", True): 0.52,
            ("XLM-R", True): 0.30,
            # ("MarIA",    True):  0.55,
        }
        votes = np.stack(
            [
                (all_probs[i] >= individual_thresholds[(name, aug)]).astype(int)
                for i, (name, aug, _) in enumerate(MODEL_REGISTRY)
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
    label_map = {0: "NM", 1: "M"}
    df["preds"] = [label_map[p] for p in preds]

    if "song_id" in df.columns:
        df_out = df[["song_id", "preds"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "preds"]]
    else:
        df["id"] = [f"T1_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "preds"]]

    df_out.to_csv(output_file, index=False)
    print(f"\nEnsemble predictions saved to {output_file}!")

    # Summary
    n_misogynist = preds.sum()
    n_total = len(preds)
    print(
        f"Predicted M (misogynist): {n_misogynist}/{n_total} ({100 * n_misogynist / n_total:.1f}%)"
    )
    print(f"Predicted NM: {n_total - n_misogynist}/{n_total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mixed ensemble prediction for Task 1 (per-model best augmentation)"
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
