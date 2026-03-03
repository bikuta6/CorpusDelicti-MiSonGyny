import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# ── AEDA punctuation marks ─────────────────────────────────────────────────────
AEDA_PUNCTS = [".", ",", "!", "?", ";", ":"]

# ── Back-translation language pairs ───────────────────────────────────────────
# For Spanish lyrics: ES → EN → ES  (or ES → FR → ES, ES → PT → ES)
BACKTRANSLATION_PAIRS = {
    "es-en": ("Helsinki-NLP/opus-mt-es-en", "Helsinki-NLP/opus-mt-en-es"),
    "es-fr": ("Helsinki-NLP/opus-mt-es-fr", "Helsinki-NLP/opus-mt-fr-es"),
    "es-pt": ("Helsinki-NLP/opus-mt-es-pt", "Helsinki-NLP/opus-mt-mul-es"),
}


# ══════════════════════════════════════════════════════════════════════════════
# AEDA  (An Easier Data Augmentation)
# Randomly inserts punctuation marks into the text.
# Paper: https://arxiv.org/abs/2108.13230
# ══════════════════════════════════════════════════════════════════════════════

def aeda_augment(text: str, insert_ratio: float = 0.15, seed: int = None) -> str:
    """Insert random punctuation marks into a text at random positions."""
    if seed is not None:
        random.seed(seed)
    words = text.split()
    if not words:
        return text
    n_inserts = max(1, int(len(words) * insert_ratio))
    for _ in range(n_inserts):
        pos = random.randint(0, len(words) - 1)
        punct = random.choice(AEDA_PUNCTS)
        words[pos] = words[pos] + punct
    return " ".join(words)


# ══════════════════════════════════════════════════════════════════════════════
# Back-translation
# ══════════════════════════════════════════════════════════════════════════════

def _load_backtranslation_pipeline(pair: str):
    """Lazy-load the two translation pipelines for a language pair."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise SystemExit("transformers is required for back-translation. Install with: pip install transformers sentencepiece")

    fwd_model, bwd_model = BACKTRANSLATION_PAIRS[pair]
    print(f"  Loading forward model  : {fwd_model}")
    fwd = hf_pipeline("translation", model=fwd_model, device=-1)
    print(f"  Loading backward model : {bwd_model}")
    bwd = hf_pipeline("translation", model=bwd_model, device=-1)
    return fwd, bwd


def backtranslate(text: str, fwd_pipe, bwd_pipe, max_length: int = 512) -> str:
    """Translate text forward then back to obtain a paraphrase."""
    if not text.strip():
        return text
    # Split into chunks to respect model max length
    lines = text.splitlines()
    translated_lines = []
    for line in lines:
        if not line.strip():
            translated_lines.append(line)
            continue
        try:
            interim = fwd_pipe(line, max_length=max_length)[0]["translation_text"]
            back   = bwd_pipe(interim, max_length=max_length)[0]["translation_text"]
            translated_lines.append(back)
        except Exception:
            translated_lines.append(line)   # fallback: keep original
    return "\n".join(translated_lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main augmentation loop
# ══════════════════════════════════════════════════════════════════════════════

def augment_file(
    input_csv: Path,
    output_csv: Path,
    text_col: str,
    label_col: str | None,
    method: str,
    bt_pair: str,
    aeda_ratio: float,
    n_augments: int,
    minority_only: bool,
):
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise SystemExit(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    # Optionally augment only the minority class
    if minority_only and label_col:
        if label_col not in df.columns:
            raise SystemExit(f"Label column '{label_col}' not found.")
        counts = df[label_col].value_counts()
        minority_label = counts.idxmin()
        augment_mask = df[label_col] == minority_label
        print(f"Minority class: '{minority_label}' ({augment_mask.sum()} samples)")
        source_df = df[augment_mask].copy()
    else:
        source_df = df.copy()

    # Load back-translation models once if needed
    fwd_pipe = bwd_pipe = None
    if method in ("bt", "both"):
        if bt_pair not in BACKTRANSLATION_PAIRS:
            raise SystemExit(f"Unknown bt-pair '{bt_pair}'. Choose from: {list(BACKTRANSLATION_PAIRS)}")
        print(f"Loading back-translation models for pair '{bt_pair}'...")
        fwd_pipe, bwd_pipe = _load_backtranslation_pipeline(bt_pair)

    new_rows = []
    for _ in range(n_augments):
        for _, row in tqdm(source_df.iterrows(), total=len(source_df), desc=f"Augmenting (pass {_ + 1}/{n_augments})"):
            text = str(row[text_col])
            augmented_texts = []

            if method == "aeda":
                augmented_texts.append(aeda_augment(text, insert_ratio=aeda_ratio))

            elif method == "bt":
                augmented_texts.append(backtranslate(text, fwd_pipe, bwd_pipe))

            elif method == "both":
                augmented_texts.append(aeda_augment(text, insert_ratio=aeda_ratio))
                augmented_texts.append(backtranslate(text, fwd_pipe, bwd_pipe))

            for aug_text in augmented_texts:
                new_row = row.copy()
                new_row[text_col] = aug_text
                new_row["augmentation"] = method
                new_rows.append(new_row)

    aug_df = pd.DataFrame(new_rows)

    # Add augmentation tag to original rows too
    df["augmentation"] = "original"
    combined = pd.concat([df, aug_df], ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nOriginal rows : {len(df)}")
    print(f"Augmented rows: {len(aug_df)}")
    print(f"Total rows    : {len(combined)}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Data augmentation for lyrics datasets (AEDA / back-translation)")
    p.add_argument("input_csv",                 help="Path to (preprocessed) input CSV")
    p.add_argument("--output",                  help="Output CSV path (default: augmented_{name} in same folder)")
    p.add_argument("--text-col",  default="lyrics",  help="Name of the text column")
    p.add_argument("--label-col", default=None,      help="Name of the label column (used with --minority-only)")
    p.add_argument("--method",    default="aeda",    choices=["aeda", "bt", "both"],
                   help="Augmentation method: aeda | bt (back-translation) | both")
    p.add_argument("--bt-pair",   default="es-en",   choices=list(BACKTRANSLATION_PAIRS),
                   help="Language pair for back-translation")
    p.add_argument("--aeda-ratio", type=float, default=0.15,
                   help="Fraction of words to insert punctuation into (AEDA)")
    p.add_argument("--n-augments", type=int,   default=1,
                   help="Number of augmented copies to generate per sample")
    p.add_argument("--minority-only", action="store_true",
                   help="Only augment samples belonging to the minority class")
    p.add_argument("--task",      default=None,
                   help="Resolve input under data/<task>/ if file not found directly")

    args = p.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists() and args.task:
        candidate = Path("data") / args.task / inp.name
        if candidate.exists():
            inp = candidate

    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    out = Path(args.output) if args.output else inp.parent / f"augmented_{inp.name}"

    augment_file(
        input_csv=inp,
        output_csv=out,
        text_col=args.text_col,
        label_col=args.label_col,
        method=args.method,
        bt_pair=args.bt_pair,
        aeda_ratio=args.aeda_ratio,
        n_augments=args.n_augments,
        minority_only=args.minority_only,
    )
    print(f"Saved augmented CSV to: {out}")


if __name__ == "__main__":
    main()