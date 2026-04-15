import re
import unicodedata

import pandas as pd


def load_dataset(path, is_new=False):
    df = pd.read_csv(path)
    # Remove song name and artist name from lyrics column (structured like title: Tus Ojos, artist: Pantoja. ) if is_new is True
    if is_new and "lyrics" in df.columns:
        df["lyrics"] = (
            df["lyrics"]
            .astype(str)
            .str.replace(r"title:.*?, artist:.*?\.\s*", "", regex=True)
        )
    return df


def preprocess_lyrics(text):
    # Normalize and clean text:
    # - treat NaN as empty string
    # - unicode normalize (decompose accents) and remove combining marks
    # - lowercase
    # - remove punctuation (keep letters/numbers/whitespace)
    # - collapse whitespace
    if pd.isna(text):
        return ""
    s = str(text)
    # Unicode normalize and remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # Replace any non-word (letters/digits/underscore) characters with space, keep whitespace
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse multiple whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity_check(lyrics1, lyrics2):
    # Jaccard similarity on token sets (requires preprocessed lyrics)
    if not lyrics1 or not lyrics2:
        return 0.0
    set1 = set(lyrics1.split())
    set2 = set(lyrics2.split())
    if not set1 and not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


if __name__ == "__main__":
    # Load datasets
    train_df = load_dataset("../data/task1/processed_train_df.csv", is_new=True)
    val_df = load_dataset("../data/task1/processed_val_df.csv", is_new=True)
    test_df = load_dataset("../data/task1/processed_dev_df.csv", is_new=True)
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    prev_df = load_dataset("../prev_data/task1/processed_full.csv", is_new=False)

    # Preprocess lyrics into a new column used for similarity comparisons
    for d in (df, prev_df):
        if "lyrics" not in d.columns:
            raise KeyError(f"DataFrame is missing 'lyrics' column: {d.columns}")
        d["sim_lyrics"] = d["lyrics"].apply(preprocess_lyrics)

    # Determine which id column new dataset uses
    new_id_col = (
        "song_id" if "song_id" in df.columns else ("id" if "id" in df.columns else None)
    )
    if new_id_col is None:
        raise KeyError("new dataset has no 'song_id' or 'id' column")

    # Collect unmatched previous rows in a list, then build DataFrame once
    unmatched = []

    # For each previous lyric, search for a matching new lyric (first match kept)
    for idx, row in prev_df.iterrows():
        lyrics = row["sim_lyrics"]
        found = False
        for idx2, row2 in df.iterrows():
            sim = similarity_check(lyrics, row2["sim_lyrics"])
            if sim > 0.6:
                print(
                    f"Found similar lyrics (sim={sim:.2f}) in new dataset for old id {row['song_id']} -> new id {row2[new_id_col]}"
                )
                print(f"  Old lyrics (preprocessed): {lyrics[:100]}...")
                print(f"  New lyrics (preprocessed): {row2['sim_lyrics'][:100]}...\n")
                found = True
                break
        if not found:
            print(f"No match found for old id {row['song_id']} in new dataset.\n")
            unmatched.append(
                {
                    "song_id": row["song_id"],
                    "lyrics": row.get("lyrics", ""),
                    "label": row.get("label", None),
                }
            )

    id_store = pd.DataFrame(unmatched, columns=["song_id", "lyrics", "label"])

    print(f"\nTotal unmatched old entries: {len(id_store)}")
    print("Sample of unmatched entries:")
    print(id_store.head())
    id_store.to_csv("../data/task1/unmatched_old_entries.csv", index=False)
