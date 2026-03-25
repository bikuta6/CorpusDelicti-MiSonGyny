import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def main(task: str = "task1", ratio: float = 0.85):
    try:
        path = f"../data/{task}/train.csv"
        df = pd.read_csv(path)
        train_df, val_df = train_test_split(
            df, test_size=1 - ratio, random_state=42, stratify=df["label"]
        )

        # overwrite train and save train and val splits
        train_df.to_csv(f"../data/{task}/train_df.csv", index=False)
        val_df.to_csv(f"../data/{task}/val_df.csv", index=False)

    except FileNotFoundError:
        path = f"./data/{task}/train.csv"
        df = pd.read_csv(path)
        train_df, val_df = train_test_split(
            df, test_size=1 - ratio, random_state=42, stratify=df["label"]
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
