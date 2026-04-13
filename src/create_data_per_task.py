import argparse
import os

import pandas as pd


def new_lyrics(df: pd.DataFrame) -> pd.Series:
    """
    Creates a new lyrics column by concatenating the title with the original lyrics.
    """
    return (
        "Título: "
        + df["song_title"]
        + "\n\n"
        + df["lyrics"]
    )


def main(filename="train.csv"):
    try:
        path = os.path.join("data", filename)
        df = pd.read_csv(path)

    except FileNotFoundError:
        path = os.path.join("..", "data", filename)
        df = pd.read_csv(path)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    print(f"Loaded {len(df)} rows from {filename}")
    print(df.columns)
    print(df.head())

    new_lyrics_column = new_lyrics(df)
    df["lyrics"] = new_lyrics_column
    common_cols = [
        "song_id",
        "song_title",
        "artist_id",
        "artist_name",
        "lyrics",
        "language",
    ]

    task1_df = df[common_cols + ["is_misogynistic"]].copy()
    task1_df.rename(columns={"is_misogynistic": "label"}, inplace=True)
    task1_df["label"] = task1_df["label"].fillna("NM")

    task2_df = df[
        common_cols + ["type_sexualization", "type_violence", "type_hate"]
    ].copy()
    task2_df.rename(
        columns={
            "type_sexualization": "sexualization",
            "type_violence": "violence",
            "type_hate": "hate",
        },
        inplace=True,
    )
    task2_df[["sexualization", "violence", "hate"]] = task2_df[
        ["sexualization", "violence", "hate"]
    ].fillna(0)

    task3_df = df[common_cols + ["has_gender_stereotype"]].copy()
    task3_df.rename(columns={"has_gender_stereotype": "label"}, inplace=True)
    task3_df["label"] = task3_df["label"].fillna("N")

    task1_df.to_csv(os.path.join("data", "task1", "train.csv"), index=False)
    task2_df.to_csv(os.path.join("data", "task2", "train.csv"), index=False)
    task3_df.to_csv(os.path.join("data", "task3", "train.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a CSV file and print its contents."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="train.csv",
        help="The name of the CSV file to load (default: train.csv)",
    )

    args = parser.parse_args()

    main(args.filename)
