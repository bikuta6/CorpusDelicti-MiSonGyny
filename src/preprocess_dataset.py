import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

try:
    from sentence_transformers import SentenceTransformer

    from contraction_utils import normalize_contractions
    from lyric_utils import format_lyrics, remove_redundant_lyrics
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from sentence_transformers import SentenceTransformer

    from contraction_utils import normalize_contractions
    from lyric_utils import format_lyrics, remove_redundant_lyrics


def new_lyrics(df: pd.DataFrame) -> pd.Series:
    """
    Creates a new lyrics column by concatenating the title with the original lyrics.
    """
    return (
        "'"
        + df["song_title"]
        + "'"
        + " de "
        + df["artist_name"]
        + "\n\n"
        + df["lyrics"]
    )


def process_file(
    input_csv: Path,
    output_csv: Path,
    model_name: str,
    threshold: float,
    text_col: str,
    line_threshold: float = 0.95,
    skip_contractions: bool = False,
    linewise_processing: bool = False,
    skip_dedup: bool = False,
):
    print(f"Reading input CSV from: {input_csv}...")
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise SystemExit(f"Input CSV has no column '{text_col}'")

    texts = df[text_col].fillna("").astype(str).tolist()

    # Step 1: Normalize contractions
    if not skip_contractions:
        print("Normalizing contractions in lyrics...")
        texts = [
            normalize_contractions(t)
            for t in tqdm(texts, desc="Normalizing contractions")
        ]

    if not skip_dedup:
        print(
            f"Loading SentenceTransformer model '{model_name}' for redundancy removal..."
        )
        model = SentenceTransformer(model_name)

    # Step 3: Remove redundant lyrics or just format them
    processed = []
    mean_length = sum(len(t.split()) for t in texts) / len(texts)

    if skip_dedup:
        print(
            f"Formatting {len(texts)} lyrics without deduplication (average length {mean_length:.1f} words)..."
        )
        for txt in tqdm(texts, desc="Formatting lyrics"):
            try:
                out = format_lyrics(txt)
            except Exception:
                out = ""
            processed.append(out)
    else:
        print(
            f"Processing {len(texts)} lyrics with average length {mean_length:.1f} words using model '{model_name}' and threshold {threshold} (stanza) and {line_threshold} (line)..."
        )
        for txt in tqdm(texts, desc="Processing lyrics"):
            try:
                out = remove_redundant_lyrics(
                    model,
                    txt,
                    threshold=threshold,
                    line_threshold=line_threshold,
                    similarity_scope=(
                        "stanza_and_verse" if linewise_processing else "stanza"
                    ),
                )
            except Exception:
                out = ""
            processed.append(out)

    mean_length_out = sum(len(t.split()) for t in processed) / len(processed)
    print(
        f"Finished processing. Average length after processing: {mean_length_out:.1f} words."
    )
    df[text_col] = processed

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def main():
    p = argparse.ArgumentParser(
        description="Preprocess lyrics using lyric_utils.remove_redundant_lyrics"
    )
    p.add_argument("input_csv", help="Path to input CSV file with lyrics column")
    p.add_argument(
        "--output",
        help="Output CSV path (default: processed_{original_name} in same folder)",
    )
    p.add_argument(
        "--model",
        default="intfloat/multilingual-e5-large",
        help="SentenceTransformer model name",
    )
    p.add_argument(
        "--stanza-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for redundancy (0-1)",
    )
    p.add_argument(
        "--linewise-processing",
        action="store_true",
        help="If set, applies redundancy removal at line level as well as stanza level",
    )
    p.add_argument(
        "--line-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for line-level redundancy (0-1)",
    )
    p.add_argument(
        "--text-col", default="lyrics", help="Name of the lyrics/text column in CSV"
    )
    p.add_argument(
        "--task",
        help="If provided and input is a filename, resolve under data/<task>/ (e.g. task1, task2)",
    )
    p.add_argument(
        "--skip-contractions",
        action="store_true",
        help="Skip contraction normalization step",
    )
    p.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip deduplication and only apply semantic formatting (commas and periods)",
    )

    args = p.parse_args()

    inp = Path(args.input_csv)

    if not inp.exists() and args.task:
        candidate = Path("data") / args.task / inp.name
        if candidate.exists():
            inp = candidate

    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    if args.output:
        out = Path(args.output)
    else:
        out = inp.parent / f"processed_{inp.name}"

    process_file(
        inp,
        out,
        args.model,
        args.stanza_threshold,
        args.text_col,
        line_threshold=args.line_threshold,
        skip_contractions=args.skip_contractions,
        linewise_processing=args.linewise_processing,
        skip_dedup=args.skip_dedup,
    )
    print(f"Saved processed CSV to: {out}")


if __name__ == "__main__":
    main()
