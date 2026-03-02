import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

try:
    # try importing lyric_utils from src root
    from lyric_utils import remove_redundant_lyrics
    from sentence_transformers import SentenceTransformer
except Exception:
    # if running from this file's directory, ensure src/ is on path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from lyric_utils import remove_redundant_lyrics
    from sentence_transformers import SentenceTransformer


def process_file(input_csv: Path, output_csv: Path, model_name: str, threshold: float, text_col: str):
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise SystemExit(f"Input CSV has no column '{text_col}'")

    model = SentenceTransformer(model_name)

    # Ensure no NaNs
    texts = df[text_col].fillna("").astype(str).tolist()

    processed = []
    mean_length = sum(len(t.split()) for t in texts) / len(texts)
    print(f"Processing {len(texts)} lyrics with average length {mean_length:.1f} words using model '{model_name}' and threshold {threshold}...")
    for txt in tqdm(texts, desc="Processing lyrics"):
        try:
            out = remove_redundant_lyrics(model, txt, threshold=threshold)
        except Exception:
            out = ""
        processed.append(out)

    # Replace the original text column in the output CSV
    mean_length_out = sum(len(t.split()) for t in processed) / len(processed)
    print(f"Finished processing. Average length after processing: {mean_length_out:.1f} words.")
    df[text_col] = processed

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def main():
    p = argparse.ArgumentParser(description="Preprocess lyrics using lyric_utils.remove_redundant_lyrics")
    p.add_argument("input_csv", help="Path to input CSV file with lyrics column")
    p.add_argument("--output", help="Output CSV path (default: processed_{original_name} in same folder)")
    p.add_argument("--model", default="intfloat/multilingual-e5-large", help="SentenceTransformer model name")
    p.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold for redundancy (0-1)")
    p.add_argument("--text-col", default="lyrics", help="Name of the lyrics/text column in CSV")
    p.add_argument("--task", help="If provided and input is a filename, resolve under data/<task>/ (e.g. task1, task2)")

    args = p.parse_args()

    inp = Path(args.input_csv)

    # If the path doesn't exist but a task is given, try resolving under data/<task>/
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

    process_file(inp, out, args.model, args.threshold, args.text_col)
    print(f"Saved processed CSV to: {out}")


if __name__ == "__main__":
    main()
