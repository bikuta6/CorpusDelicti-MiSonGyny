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
    # replace newline and double new lines with space
    s = s.replace("\n", " ").replace("\n\n", " ")
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

    prev_train_df = load_dataset("../prev_data/task1/train.csv", is_new=False)
    prev_test_df_text = load_dataset("../prev_data/task1/test.csv", is_new=False)
    prev_test_labels = load_dataset("../prev_data/task1/test_labels.csv", is_new=False)
    prev_test_df = prev_test_df_text.merge(
        prev_test_labels, on="song_id", how="left"
    )
    prev_df = pd.concat([prev_train_df, prev_test_df], ignore_index=True)

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

    # Collect matched and unmatched previous rows
    unmatched = []
    matched_same_label = []
    matched_changed_label = []

    # For each previous lyric, search for a matching new lyric (first match kept)
    for idx, row in prev_df.iterrows():
        lyrics = row["sim_lyrics"]
        found = False
        for idx2, row2 in df.iterrows():
            sim = similarity_check(lyrics, row2["sim_lyrics"])
            if sim > 0.6:
                old_label = row.get("label", None)
                new_label = row2.get("label", None)
                
                print(
                    f"Found similar lyrics (sim={sim:.2f}) in new dataset for old id {row['song_id']} -> new id {row2[new_id_col]}"
                )
                print(f"  Old label: {old_label}")
                print(f"  New label: {new_label}")
                
                # Check if label changed
                if old_label != new_label:
                    print(f"  ⚠️  LABEL CHANGED from '{old_label}' to '{new_label}'")
                    matched_changed_label.append(
                        {
                            "old_song_id": row["song_id"],
                            "new_song_id": row2[new_id_col],
                            "old_label": old_label,
                            "new_label": new_label,
                            "similarity": sim,
                            "old_lyrics": row.get("lyrics", ""),
                            "new_lyrics": row2.get("lyrics", ""),
                        }
                    )
                else:
                    matched_same_label.append(
                        {
                            "old_song_id": row["song_id"],
                            "new_song_id": row2[new_id_col],
                            "label": old_label,
                            "similarity": sim,
                            "old_lyrics": row.get("lyrics", ""),
                            "new_lyrics": row2.get("lyrics", ""),
                        }
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

    # Create DataFrames for each category
    unmatched_df = pd.DataFrame(unmatched, columns=["song_id", "lyrics", "label"]) if unmatched else pd.DataFrame()
    matched_same_df = pd.DataFrame(matched_same_label) if matched_same_label else pd.DataFrame()
    matched_changed_df = pd.DataFrame(matched_changed_label) if matched_changed_label else pd.DataFrame()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"Total songs in previous dataset: {len(prev_df)}")
    print(f"Total songs in new dataset: {len(df)}")
    print(f"\nMatches found with SAME label: {len(matched_same_df)}")
    print(f"Matches found with CHANGED label: {len(matched_changed_df)}")
    print(f"Unmatched old entries: {len(unmatched_df)}")
    print("="*80)
    
    # Show changed labels
    if not matched_changed_df.empty:
        print("\n⚠️  SONGS WITH LABEL CHANGES:")
        print("-"*80)
        for idx, row in matched_changed_df.iterrows():
            print(f"{idx+1}. Old ID {row['old_song_id']} -> New ID {row['new_song_id']}")
            print(f"   Label change: '{row['old_label']}' -> '{row['new_label']}'")
            print(f"   Similarity: {row['similarity']:.2f}")
            print()
    
    # Save results to CSV files
    if not unmatched_df.empty:
        unmatched_df.to_csv("../prev_data/task1/unmatched_old_entries.csv", index=False)
        print(f"\nUnmatched entries saved to: ../prev_data/task1/unmatched_old_entries.csv")
    
    if not matched_same_df.empty:
        matched_same_df.to_csv("../prev_data/task1/matched_same_label.csv", index=False)
        print(f"Matched (same label) entries saved to: ../prev_data/task1/matched_same_label.csv")
    
    if not matched_changed_df.empty:
        matched_changed_df.to_csv("../prev_data/task1/matched_changed_label.csv", index=False)
        print(f"Matched (changed label) entries saved to: ../prev_data/task1/matched_changed_label.csv")
