#!/usr/bin/env python3
"""
Apply LyricsAugmentor to a task's training CSV and optionally export the augmented
lyrics to other tasks that share the same `song_id` (keeping their labels).

Usage examples:
  # Augment task2 (default), multiplier x2, export to task1 and task3
  python src/apply_augmentation.py --source-task task2 --multiplier 2 --export-to task1,task3

  # Augment only specific class values in a label column
  python src/apply_augmentation.py --source-task task2 \
      --label-col sexualization --target-classes 1 \
      --multiplier 3

Notes:
- Default paths assume this script lives under `src/` and data under `data/{task}/train_df.csv`.
- Augmented rows keep the same `song_id`. When exported to other tasks, the script joins
  on `song_id` and copies the target-task labels onto the augmented lyrics rows.
"""

import argparse
import os
from typing import List, Optional

import pandas as pd

from augmentation_utils import LyricsAugmentor


def _try_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    # try alternate relative path
    alt = path.replace("../data/", "./data/")
    if os.path.exists(alt):
        return pd.read_csv(alt)
    raise FileNotFoundError(f"Could not find file at {path} or {alt}")


def _save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def apply_augmentation_to_task(
    source_task: str,
    processed_task: bool,
    multiplier: int = 1,
    seed: int = 42,
    method_probs: Optional[List[float]] = None,
    text_col: str = "lyrics",
    label_col: Optional[str] = None,
    target_classes: Optional[List[str]] = None,
    export_to: Optional[List[str]] = None,
    out_suffix: str = "augmented",
):
    """
    - source_task: e.g. "task2"
    - multiplier: factor for how many augmented copies per row
    - method_probs: list of 7 floats summing to 1.0 (synonym, back_en, back_fr, llm, rand_del, rand_char_ins, aeda)
    - text_col: column containing lyrics
    - label_col: column used for selecting target_classes (if provided)
    - target_classes: list of label values to augment (if provided). If None => augment all rows.
    - export_to: list of other task names to export augmented lyrics to (based on song_id)
    - out_suffix: suffix used in output filenames
    """
    pre = "processed_" if processed_task else ""
    src_path = os.path.join("..", "data", source_task, f"{pre}train_df.csv")
    print(f"Loading source data: {src_path}")
    df = _try_read_csv(src_path)

    aug = LyricsAugmentor(seed=seed, method_probs=method_probs)

    # If label_col and target_classes provided, pass them to augmentor; else augment all
    df_aug_all = aug.augment_dataframe(
        df,
        text_col=text_col,
        label_col=label_col if label_col is not None else "label",
        target_classes=target_classes,
        multiplier=multiplier,
    )

    # Save full augmented dataset for source task
    out_src_path = os.path.join("..", "data", source_task, f"train_df.{out_suffix}.csv")
    _save_csv(df_aug_all, out_src_path)

    # Extract only augmented rows (augmentation != 'original')
    if "augmentation" in df_aug_all.columns:
        augmented_only = (
            df_aug_all[df_aug_all["augmentation"] != "original"]
            .copy()
            .reset_index(drop=True)
        )
    else:
        # If augmentation column not present, try to infer by size difference
        augmented_only = df_aug_all.iloc[len(df) :].copy().reset_index(drop=True)

    print(f"Augmented rows (only): {len(augmented_only)}")

    if export_to:
        # For each target task, map augmented lyrics to that task's labels via song_id
        for tgt in export_to:
            tgt_path = os.path.join("..", "data", tgt, "train_df.csv")
            print(
                f"Exporting augmented lyrics to task '{tgt}' using labels from {tgt_path}"
            )
            try:
                df_tgt = _try_read_csv(tgt_path)
            except FileNotFoundError:
                print(f"  WARNING: target task file not found: {tgt_path} - skipping")
                continue

            # determine label columns in target task (all columns except common metadata)
            # We'll keep all columns from target except the lyrics column to avoid overwriting original lyrics
            # and then replace lyrics with augmented ones.
            # Identify columns to bring into new rows: those present in df_tgt but not the source lyrics text or augmentation
            # Basic join on 'song_id'
            if "song_id" not in df_tgt.columns:
                print(f"  WARNING: 'song_id' not found in target {tgt}; skipping")
                continue
            # Build a minimal df with song_id and label columns from target
            label_cols = [
                c
                for c in df_tgt.columns
                if c not in (text_col, "song_title", "artist_id", "artist_name")
            ]
            # ensure song_id included
            if "song_id" not in label_cols:
                label_cols = ["song_id"] + [c for c in label_cols if c != "song_id"]

            df_tgt_labels = (
                df_tgt[label_cols]
                .drop_duplicates(subset=["song_id"])
                .set_index("song_id")
            )

            # Merge augmented_only with target labels by song_id
            merged = augmented_only.merge(
                df_tgt_labels,
                how="inner",
                left_on="song_id",
                right_index=True,
                suffixes=("_src", "_tgt"),
            )

            if merged.empty:
                print(
                    f"  No shared song_id between augmented source and target '{tgt}' - nothing to export."
                )
                continue

            # Build new rows for the target: keep target's label columns and use augmented lyrics
            new_rows = []
            for _, row in merged.iterrows():
                # base_row: take target label columns values from merged
                new_row = {}
                # include metadata if present in target (e.g., song_title/artist_name) else take from augmented row
                for col in df_tgt.columns:
                    if col == text_col:
                        new_row[col] = row[text_col]
                    elif f"{col}_tgt" in merged.columns:
                        new_row[col] = row[f"{col}_tgt"]
                    elif col in merged.columns:
                        new_row[col] = row[col]
                    else:
                        # fallback: if column exists in augmented_only, use it; else NaN
                        new_row[col] = row.get(col, pd.NA)
                # ensure augmentation column exists/indicates origin
                new_row["augmentation"] = row.get(
                    "augmentation", "augmented_from_" + source_task
                )
                new_rows.append(new_row)

            df_new = pd.DataFrame(new_rows)

            # Append to existing target train df and save as new file (do not overwrite original by default)
            out_tgt_path = os.path.join("..", "data", tgt, f"train_df.{out_suffix}.csv")
            df_tgt_combined = pd.concat([df_tgt, df_new], ignore_index=True)
            _save_csv(df_tgt_combined, out_tgt_path)
            print(
                f"  Exported augmented lyrics to {out_tgt_path} (+{len(df_new)} rows)."
            )


def _parse_list_arg(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    return [item.strip() for item in s.split(",") if item.strip()]


def _parse_probs(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 7:
        raise argparse.ArgumentTypeError(
            "method_probs must have 7 comma-separated floats."
        )
    ssum = sum(vals)
    if abs(ssum - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError("method_probs must sum to 1.0")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Apply lyrics augmentation and export to other tasks."
    )
    parser.add_argument(
        "--source-task",
        type=str,
        default="task2",
        help="Source task name (folder under data/)",
    )
    parser.add_argument(
        "--processed-task",
        type=bool,
        default=False,
        help="Whether the source task is already processed",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="How many augmented copies per source example",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--method-probs",
        type=_parse_probs,
        default=None,
        help="Comma-separated 7 floats summing to 1.0: probs for [synonym, back_en, back_fr, llm, rand_del, rand_char, aeda]",
    )
    parser.add_argument(
        "--text-col", type=str, default="lyrics", help="Text column name"
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column used for filtering (optional)",
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        default=None,
        help="Comma-separated label values to augment",
    )
    parser.add_argument(
        "--export-to",
        type=str,
        default=None,
        help="Comma-separated target task names to export to",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="augmented",
        help="Suffix for output filenames",
    )

    args = parser.parse_args()

    target_classes = _parse_list_arg(args.target_classes)
    export_to = _parse_list_arg(args.export_to)
    method_probs = args.method_probs

    apply_augmentation_to_task(
        source_task=args.source_task,
        processed_task=args.processed_task,
        multiplier=args.multiplier,
        seed=args.seed,
        method_probs=method_probs,
        text_col=args.text_col,
        label_col=args.label_col,
        target_classes=target_classes,
        export_to=export_to,
        out_suffix=args.out_suffix,
    )


if __name__ == "__main__":
    main()
