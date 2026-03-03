import argparse
from pathlib import Path
from collections import Counter
import re
import pandas as pd


def find_contractions(texts: list[str]) -> Counter:
    """Find all words containing apostrophes in the texts."""
    counter = Counter()
    pattern = re.compile(r"\b\w*'\w*\b", re.IGNORECASE)
    for text in texts:
        matches = pattern.findall(text)
        counter.update(m.lower() for m in matches)
    return counter


def main():
    p = argparse.ArgumentParser(description="Check dataset for contractions (words with apostrophes)")
    p.add_argument("input_csv", help="Path to input CSV file")
    p.add_argument("--text-col", default="lyrics", help="Name of the text column in CSV")
    p.add_argument("--top", type=int, default=50, help="Show top N most frequent contractions")
    args = p.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists():
        raise SystemExit(f"File not found: {inp}")

    df = pd.read_csv(inp)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    print(f"Scanning {len(texts)} rows for contractions...\n")

    counter = find_contractions(texts)

    if not counter:
        print("No contractions found.")
        return

    print(f"Found {len(counter)} unique contractions | {sum(counter.values())} total occurrences\n")
    print(f"{'Rank':<6} {'Word':<20} {'Count':>8}")
    print("-" * 36)
    for rank, (word, count) in enumerate(counter.most_common(args.top), start=1):
        print(f"{rank:<6} {word:<20} {count:>8}")


if __name__ == "__main__":
    main()