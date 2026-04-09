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

echo "Running preprocessing for all tasks..."

uv run src/preprocess_dataset.py train.csv --task task1 $SKIP_DEDUP
uv run src/preprocess_dataset.py train.csv --task task2 $SKIP_DEDUP
uv run src/preprocess_dataset.py train.csv --task task3 $SKIP_DEDUP

echo "Preprocessing complete!"
