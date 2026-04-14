#!/usr/bin/env bash

# Check if --skip-dedup is passed as an argument
SKIP_DEDUP=""
for arg in "$@"; do
    if [ "$arg" == "--skip-dedup" ]; then
        SKIP_DEDUP="--skip-dedup"
        echo "Info: Skipping deduplication step."
        break
    fi
done

# Check if --skip-contractions is passed as an argument
SKIP_CONTRACTIONS=""
for arg in "$@"; do
    if [ "$arg" == "--skip-contractions" ]; then
        SKIP_CONTRACTIONS="--skip-contractions"
        echo "Info: Skipping contractions step."
        break
    fi
done

echo "Running preprocessing for all tasks..."

uv run src/preprocess_dataset.py train.csv --task task1 $SKIP_DEDUP $SKIP_CONTRACTIONS
uv run src/preprocess_dataset.py train.csv --task task2 $SKIP_DEDUP $SKIP_CONTRACTIONS
uv run src/preprocess_dataset.py train.csv --task task3 $SKIP_DEDUP $SKIP_CONTRACTIONS

echo "Preprocessing complete!"
