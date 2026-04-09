#!/usr/bin/env bash

echo "Running splitting for all tasks..."

uv run src/create_val_test_split.py --task task1 --baseline
uv run src/create_val_test_split.py --task task1
uv run src/create_val_test_split.py --task task2 --baseline
uv run src/create_val_test_split.py --task task2
uv run src/create_val_test_split.py --task task3 --baseline
uv run src/create_val_test_split.py --task task3

echo "Splitting complete!"
