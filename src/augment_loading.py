import pandas as pd


def augment_df(
    df: pd.DataFrame, aug_path: str = "../../data/processed_train_augmented.csv"
) -> pd.DataFrame:
    """
    Augment the given DataFrame with additional data from a CSV file.

    Parameters:
    df (pd.DataFrame): The original DataFrame to be augmented.
    aug_path (str): The file path to the CSV file containing the additional data.

    Returns:
    pd.DataFrame: The augmented DataFrame.
    """
    # Load the additional data from the CSV file
    aug_df = pd.read_csv(aug_path)
    # check the song_id column exists in both dataframes
    if "song_id" not in df.columns or "song_id" not in aug_df.columns:
        raise ValueError("Both DataFrames must contain a 'song_id' column for merging.")
    # now extract the augmented columns where id mathce sthe ones in the original dataframe and augmentation column is different form 'original'

    src_id = df["song_id"].unique()
    aug_df = aug_df[
        aug_df["song_id"].isin(src_id) & (aug_df["augmentation"] != "original")
    ]
    # now from the new rows, include the labels from the original dataframe
    irreleveant_cols = [
        "song_title",
        "artist_name",
        "artist_id",
        "language",
        "augmentation_idx",
        "source_row_idx",
        "augmentation_slot",
        "label",
    ]
    for col in irreleveant_cols[:4]:
        if col in df.columns:
            df = df.drop(columns=col)
    df["augmentation"] = "original"

    for col in irreleveant_cols:
        if col in aug_df.columns:
            aug_df = aug_df.drop(columns=col)
    print(df.columns)
    label_cols = list(df.columns.difference(["song_id", "augmentation", "lyrics"]))

    if len(label_cols) == 3:
        label_cols = ["sexualization", "violence", "hate"]
    print(label_cols)
    labels = df[["song_id"] + label_cols]
    aug_df = aug_df.merge(labels, on="song_id", how="left")

    # now merge the original dataframe with the augmented dataframe on song_id
    final_df = pd.concat([df, aug_df], ignore_index=True)

    print(final_df.columns)

    return final_df


if __name__ == "__main__":
    # Example usage
    original_df = pd.read_csv("../data/task2/processed_train_df.csv")
    print("Original DataFrame size:", original_df.shape)
    path_aug = "../data/processed_train_augmented.csv"
    augmented_df = augment_df(original_df, aug_path=path_aug)
    print("Augmented DataFrame size:", augmented_df.shape)
    print(augmented_df)
    # now check for all augmented rows, the labels are the same as the original rows
    label_cols = list(
        original_df.columns.difference(
            [
                "song_id",
                "song_title",
                "artist_name",
                "artist_id",
                "language",
                "augmentation_idx",
                "source_row_idx",
                "augmentation_slot",
                "augmentation",
                "lyrics",
            ]
        )
    )
    for idx, row in augmented_df.iterrows():
        if row["augmentation"] != "original":
            original_row = augmented_df[
                (augmented_df["song_id"] == row["song_id"])
                & (augmented_df["augmentation"] == "original")
            ]
            if not original_row.empty:
                for col in label_cols:
                    assert row[col] == original_row.iloc[0][col], (
                        f"Label mismatch for song_id {row['song_id']} in column {col}"
                    )
    print("All augmented rows have the same labels as the original rows.")
