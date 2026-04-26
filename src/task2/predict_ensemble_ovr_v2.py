"""
Task 2 - OVR Ensemble Prediction
Per-label model registry: each label uses its best-performing models.
Thresholds are hardcoded from val JSON files (no runtime file dependency).
Supports both hard voting (default) and weighted soft voting.
"""

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
# Per-label model registry.
# Each entry: (model_name, augment, val_f1, best_threshold)
# Models are ordered best-first; adjust the list to control ensemble size.
# All models use processed_test.csv (all were trained on processed data).
# ---------------------------------------------------------------------------
LABEL_REGISTRY = {
    "sexualization": [
        ("BETO", True, 0.8302, 0.50),
        ("Robertuito", True, 0.8269, 0.96),
        ("LongFormer", False, 0.8239, 0.29),
        ("XLM-R", True, 0.8133, 0.13),
        ("DistilBETO", False, 0.8071, 0.47),
    ],
    "violence": [
        ("BETO", True, 0.5926, 0.50),
        ("DistilBETO", False, 0.5750, 0.38),
        ("Robertuito", False, 0.5443, 0.61),  # baseline slightly better for violence
        ("XLM-R", True, 0.5366, 0.41),
        ("LongFormer", False, 0.5135, 0.56),
    ],
    "hate": [
        ("DistilBETO", False, 0.6873, 0.47),
        ("Robertuito", False, 0.6776, 0.66),  # baseline better for hate
        ("XLM-R", True, 0.6740, 0.91),
        ("Robertuito", True, 0.6691, 0.86),
        ("BETO", True, 0.6535, 0.69),
    ],
}

# How many top models to use per label (must be odd for clean majority voting)
TOP_N = 3

LABEL_SHORT = {
    "sexualization": "S",
    "violence": "V",
    "hate": "H",
}


def get_ovr_probs(model_name, label_full, test_ds, device, augment=False):
    cfg = MODEL_CONFIGS[model_name]
    suffix = "_augmented" if augment else ""
    model_path = f"../../models/task2/final_OVR/{model_name}{suffix}_{label_full}"
    aug_label = "aug" if augment else "baseline"
    print(f"    Loading {model_name} ({aug_label}) for {label_full}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_bert_like_classifier(model_path, device)

    pred_output = predict_with_chunks(
        test_ds, tokenizer, model, device, cfg.max_len, aggregation="max"
    )
    probs = torch.sigmoid(torch.tensor(pred_output.predictions)).numpy().flatten()

    del model, tokenizer
    torch.cuda.empty_cache()

    return probs


def main(voting_type="hard", top_n=TOP_N):
    output_file = "../../task_2_predictions.csv"

    print("Loading test dataset...")
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
    print(f"Using device: {device}")
    print(f"Voting: {voting_type} | Top-N models per label: {top_n}\n")

    all_preds = {}

    for label_full, label_short in LABEL_SHORT.items():
        print(f"\n{'=' * 50}")
        print(f"  Processing label: {label_full.upper()}")
        print(f"{'=' * 50}")

        registry = LABEL_REGISTRY[label_full][:top_n]
        weights = np.array([entry[2] for entry in registry])  # val_f1 scores

        probs_list = []
        for model_name, augment, val_f1, threshold in registry:
            probs = get_ovr_probs(
                model_name, label_full, test_ds, device, augment=augment
            )
            probs_list.append((probs, threshold, val_f1, model_name, augment))

        if voting_type == "soft":
            normalized_weights = weights / weights.sum()
            print(f"\n  Soft voting weights for {label_full}:")
            for (probs, thr, f1, name, aug), w in zip(probs_list, normalized_weights):
                print(
                    f"    {name} ({'aug' if aug else 'base'}): weight={w:.4f}  f1={f1:.4f}  thr={thr}"
                )

            final_probs = sum(
                probs * w
                for (probs, _, _, _, _), w in zip(probs_list, normalized_weights)
            )
            # Blended threshold: weighted avg of individual best thresholds
            thresholds = np.array([thr for _, thr, _, _, _ in probs_list])
            blended_thr = float((thresholds * normalized_weights).sum())
            print(f"  Blended threshold: {blended_thr:.4f}")
            final_preds = (final_probs >= blended_thr).astype(int)

        else:  # hard voting (default — more robust for OVR with very different thresholds)
            binary_votes = []
            for probs, threshold, val_f1, model_name, augment in probs_list:
                vote = (probs >= threshold).astype(int)
                binary_votes.append(vote)
                aug_label = "aug" if augment else "base"
                print(
                    f"    {model_name} ({aug_label}): thr={threshold}  f1={val_f1:.4f}  "
                    f"positives={vote.sum()}/{len(vote)}"
                )

            majority = len(registry) // 2 + 1
            sum_votes = np.sum(binary_votes, axis=0)
            final_preds = (sum_votes >= majority).astype(int)
            print(
                f"  Hard voting majority >= {majority}/{len(registry)} -> "
                f"positives={final_preds.sum()}/{len(final_preds)}"
            )

        all_preds[label_short] = final_preds

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    for short_col in ["S", "V", "H"]:
        df[short_col] = all_preds[short_col]

    if "song_id" in df.columns:
        df_out = df[["song_id", "S", "V", "H"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "S", "V", "H"]]
    else:
        ids = [f"T2_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = pd.DataFrame(
            {"id": ids, "S": all_preds["S"], "V": all_preds["V"], "H": all_preds["H"]}
        )

    df_out.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}!")
    print(f"  S positives: {all_preds['S'].sum()}/{len(df)}")
    print(f"  V positives: {all_preds['V'].sum()}/{len(df)}")
    print(f"  H positives: {all_preds['H'].sum()}/{len(df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OVR Ensemble Prediction for Task 2")
    parser.add_argument(
        "--voting",
        type=str,
        choices=["soft", "hard"],
        default="hard",
        help="Voting method: 'hard' (majority, default) or 'soft' (weighted avg probabilities)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help=f"Number of top models to use per label (default: {TOP_N}, must be odd for hard voting)",
    )
    args = parser.parse_args()

    if args.voting == "hard" and args.top_n % 2 == 0:
        print(
            f"Warning: --top-n={args.top_n} is even; hard voting may produce ties. Consider using an odd number."
        )

    main(voting_type=args.voting, top_n=args.top_n)
