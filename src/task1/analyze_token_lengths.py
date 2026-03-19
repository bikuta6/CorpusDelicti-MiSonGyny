import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def sanitize_tokenizer_name(name: str) -> str:
    """
    Convert tokenizer/model id into a filesystem-friendly suffix.
    Example: 'dccuchile/bert-base-spanish-wwm-cased' ->
             'dccuchile_bert-base-spanish-wwm-cased'
    """
    name = name.strip().replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name


def summarize_lengths(lengths: np.ndarray) -> dict:
    percentiles = [50, 75, 90, 95, 97, 98, 99, 100]
    summary = {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "max": int(np.max(lengths)),
    }
    for p in percentiles:
        summary[f"p{p}"] = int(np.percentile(lengths, p))
    return summary


def coverage_at_thresholds(lengths: np.ndarray, thresholds: list[int]) -> dict:
    n = len(lengths)
    return {t: float((lengths <= t).sum()) / n for t in thresholds}


def build_length_buckets(
    lengths: np.ndarray, thresholds: list[int]
) -> list[tuple[str, np.ndarray]]:
    """
    Build disjoint buckets:
      <=t1, (t1,t2], (t2,t3], ..., >t_last
    """
    thresholds = sorted(set(thresholds))
    buckets = []
    prev = None
    for t in thresholds:
        if prev is None:
            mask = lengths <= t
            label = f"<= {t}"
        else:
            mask = (lengths > prev) & (lengths <= t)
            label = f"{prev + 1}..{t}"
        buckets.append((label, mask))
        prev = t

    mask = lengths > thresholds[-1]
    label = f"> {thresholds[-1]}"
    buckets.append((label, mask))
    return buckets


def infer_label_type(series: pd.Series) -> str:
    """
    Infer label type from series values:
      - 'binary_numeric' for {0,1}
      - 'binary_text' for {'NM','M'} or any 2 unique strings
      - 'multiclass' for >2 unique values
      - 'unknown' otherwise
    """
    vals = series.dropna()
    if vals.empty:
        return "unknown"

    unique_vals = pd.unique(vals)
    if len(unique_vals) == 2:
        # numeric binary
        try:
            as_num = pd.to_numeric(vals, errors="raise")
            u = sorted(pd.unique(as_num))
            if set(u) == {0, 1}:
                return "binary_numeric"
        except Exception:
            pass
        return "binary_text"

    if len(unique_vals) > 2:
        return "multiclass"

    return "unknown"


def compute_positive_mask(labels: pd.Series) -> tuple[np.ndarray | None, str]:
    """
    Try to create a positive-class mask for binary labels.
    Returns (mask_or_none, positive_label_name)
    """
    lt = infer_label_type(labels)
    vals = labels.fillna("")

    if lt == "binary_numeric":
        arr = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int).to_numpy()
        return (arr == 1), "1"

    if lt == "binary_text":
        # Prefer common misogyny setup labels: NM/M
        normalized = vals.astype(str).str.strip()
        uniq = set(normalized.unique())
        if "M" in uniq and "NM" in uniq:
            return (normalized == "M").to_numpy(), "M"

        # Fallback: take the lexicographically larger as "positive"
        ordered = sorted(list(uniq))
        if len(ordered) == 2:
            pos = ordered[1]
            return (normalized == pos).to_numpy(), pos

    return None, ""


def print_summary(summary: dict, coverages: dict):
    print("\n=== Token Length Summary ===")
    print(f"count: {summary['count']}")
    print(f"min:   {summary['min']}")
    print(f"mean:  {summary['mean']:.2f}")
    print(f"std:   {summary['std']:.2f}")
    print(f"max:   {summary['max']}")
    print("\nPercentiles:")
    for k in ["p50", "p75", "p90", "p95", "p97", "p98", "p99", "p100"]:
        print(f"  {k}: {summary[k]}")

    print("\nCoverage by max_len:")
    for t, c in coverages.items():
        print(f"  <= {t:>4}: {c * 100:6.2f}%")

    print("\nSuggested candidates:")
    print("  - For long-lyric tasks, keep 512 as strong baseline")
    print("  - Evaluate 512 with smart truncation vs chunking for very long samples")


def print_label_bias_report(
    lengths: np.ndarray,
    labels: pd.Series,
    thresholds: list[int],
    label_col: str,
):
    print("\n=== Label-Length Bias Analysis ===")
    label_type = infer_label_type(labels)
    print(f"Label column: {label_col}")
    print(f"Inferred label type: {label_type}")

    # General per-bucket class distribution
    buckets = build_length_buckets(lengths, thresholds)
    temp = pd.DataFrame({label_col: labels, "n_tokens": lengths})

    for bucket_name, mask in buckets:
        bucket_df = temp[mask]
        n = len(bucket_df)
        if n == 0:
            continue

        counts = bucket_df[label_col].value_counts(dropna=False)
        shares = (counts / n * 100).round(2)
        dist_str = ", ".join([f"{k}: {counts[k]} ({shares[k]}%)" for k in counts.index])
        print(f"  [{bucket_name}] n={n} -> {dist_str}")

    # Binary-positive drift report (if possible)
    pos_mask, pos_name = compute_positive_mask(labels)
    if pos_mask is None:
        print(
            "Binary positive-class drift: not available (non-binary or unknown labels)."
        )
        return

    global_pos_rate = float(pos_mask.mean())
    print(
        f"\nBinary positive-class ({pos_name}) global rate: {global_pos_rate * 100:.2f}%"
    )

    # Cumulative: <= threshold
    print("\nPositive-class rate by cumulative max_len:")
    for t in sorted(set(thresholds)):
        m = lengths <= t
        if m.sum() == 0:
            continue
        rate = float(pos_mask[m].mean())
        delta = (rate - global_pos_rate) * 100
        print(f"  <= {t:>4}: {rate * 100:6.2f}%  (Δ {delta:+.2f} pp)")

    # Tail
    tail_t = max(thresholds)
    m_tail = lengths > tail_t
    if m_tail.sum() > 0:
        rate_tail = float(pos_mask[m_tail].mean())
        delta_tail = (rate_tail - global_pos_rate) * 100
        print(
            f"  >  {tail_t:>4}: {rate_tail * 100:6.2f}%  (Δ {delta_tail:+.2f} pp) "
            f"[n={int(m_tail.sum())}]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution and label-length bias for a CSV text column."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/task1/processed_train.csv",
        help="Path to CSV file.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="lyrics",
        help="Text column name in CSV.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Label column name in CSV for bias analysis.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="dccuchile/bert-base-spanish-wwm-cased",
        help="HF tokenizer/model id.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[64, 128, 192, 256, 384, 512, 768, 1024],
        help="Max length values for coverage and bias report.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for tokenization.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {csv_path}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    if len(texts) == 0:
        raise SystemExit("No texts found in dataset.")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    lengths = []
    print(f"Tokenizing {len(texts)} samples...")
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])

    lengths = np.array(lengths, dtype=np.int32)

    summary = summarize_lengths(lengths)
    coverages = coverage_at_thresholds(lengths, args.thresholds)
    print_summary(summary, coverages)

    if args.label_col in df.columns:
        print_label_bias_report(
            lengths, df[args.label_col], args.thresholds, args.label_col
        )
    else:
        print(
            f"\nLabel-length bias analysis skipped: column '{args.label_col}' not found."
        )

    # Save per-sample lengths with tokenizer-specific name
    tok_suffix = sanitize_tokenizer_name(args.tokenizer)
    out_path = csv_path.with_name(f"{csv_path.stem}_token_lengths__{tok_suffix}.csv")
    out_df = df.copy()
    out_df["n_tokens"] = lengths
    # out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-sample lengths to: {out_path}")


if __name__ == "__main__":
    main()
