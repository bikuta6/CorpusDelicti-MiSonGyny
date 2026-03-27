import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import set_seed


def get_stratify_col(task: str, df: pd.DataFrame) -> pd.Series:
    if task in ["task1", "task3"]:
        return df["label"]
    elif task == "task2":
        # Combines multiple columns to ensure balanced representation across all categories
        return (
            df[["sexualization", "violence", "hate"]].astype(str).agg("_".join, axis=1)
        )
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'task1', 'task2', or 'task3'.")


def main(task: str = "task1", train_ratio: float = 0.8):
    set_seed(42)

    # 1. Path Handling: Search for the file in possible locations
    paths_to_check = [
        f"../data/{task}/processed_train.csv",
        f"./data/{task}/processed_train.csv",
    ]
    path = next((p for p in paths_to_check if os.path.exists(p)), None)

    if not path:
        print(f"Error: Could not find processed_train.csv for {task}")
        return

    df = pd.read_csv(path)
    stratify_col = get_stratify_col(task, df)

    # 2. First Split: Separate Train from the rest (Val + Dev)
    # If train_ratio is 0.8, temp_size is 0.2
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=42, stratify=stratify_col
    )

    # 3. Second Split: Split the remainder equally into Val and Dev (Test)
    # We recalculate the stratification column for the subset
    temp_stratify = get_stratify_col(task, temp_df)
    val_df, dev_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_stratify
    )

    # 4. Save the files
    base_dir = os.path.dirname(path)
    train_df.to_csv(os.path.join(base_dir, "train_df.csv"), index=False)
    val_df.to_csv(os.path.join(base_dir, "val_df.csv"), index=False)
    dev_df.to_csv(os.path.join(base_dir, "dev_df.csv"), index=False)

    print(f"Splits completed for {task}:")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Dev: {len(dev_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Train, Val, and Dev splits.")
    parser.add_argument("--task", type=str, default="task1", help="Task name")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8 for an 80/10/10 split)",
    )
    args = parser.parse_args()
    main(task=args.task, train_ratio=args.ratio)
