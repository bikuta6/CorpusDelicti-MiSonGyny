import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import set_seed


def get_stratify_col(task: str, df: pd.DataFrame) -> pd.Series:
    if task in ["task1", "task3"]:
        return df["label"]
    elif task == "task2":
        return (
            df[["sexualization", "violence", "hate"]].astype(str).agg("_".join, axis=1)
        )
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'task1', 'task2', or 'task3'.")


def main(task: str = "task1", ratio: float = 0.8):

    set_seed(42)  # Establecer seed para reproducibilidad
    try:
        path = f"../data/{task}/processed_train.csv"
        df = pd.read_csv(path)
        stratify_col = get_stratify_col(task, df)
        train_df, val_df = train_test_split(
            df, test_size=1 - ratio, random_state=42, stratify=stratify_col
        )

        # overwrite train and save train and val splits
        train_df.to_csv(f"../data/{task}/train_df.csv", index=False)
        val_df.to_csv(f"../data/{task}/val_df.csv", index=False)

    except FileNotFoundError:
        path = f"./data/{task}/processed_train.csv"
        df = pd.read_csv(path)
        stratify_col = get_stratify_col(task, df)
        train_df, val_df = train_test_split(
            df, test_size=1 - ratio, random_state=42, stratify=stratify_col
        )

        # overwrite train and save train and val splits
        train_df.to_csv(f"./data/{task}/train_df.csv", index=False)
        val_df.to_csv(f"./data/{task}/val_df.csv", index=False)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a validation split from a training dataset."
    )
    parser.add_argument(
        "--task", type=str, default="task1", help="Task name (e.g., task1, task2)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to keep (default: 0.85)",
    )
    args = parser.parse_args()
    main(task=args.task, ratio=args.ratio)
