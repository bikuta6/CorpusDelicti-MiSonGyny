#!/usr/bin/env bash
set -euo pipefail

# Reproducible end-to-end pipeline as requested:
# - Preprocess (no dedup, no contractions) for Task 1 & 2
# - Task 3 uses raw data (no preprocessing)
# - Augmentation is done on the base dataset using Task 1 labels
# - Task 1: train final models on preprocessed data, soft-voting ensemble
# - Task 3: raw data, hard-voting ensemble
# - Task 2: OVR pipeline with augmented data

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is required. Install it first." >&2
  exit 1
fi

if [ ! -f data/train.csv ] || [ ! -f data/test.csv ]; then
  echo "Error: missing data/train.csv or data/test.csv" >&2
  exit 1
fi

export PYTHONHASHSEED=42

echo "[1/10] Generating per-task datasets"
uv run src/create_data_per_task.py --filename train.csv

echo "[2/10] Preprocessing Task 1 & 2 (skip dedup + contractions)"
uv run src/preprocess_dataset.py train.csv --task task1 --skip-dedup --skip-contractions
uv run src/preprocess_dataset.py train.csv --task task2 --skip-dedup --skip-contractions

# Task 3 stays raw (no preprocessing)

echo "[3/10] Preprocessing test set for Task 1 & 2"
uv run src/preprocess_dataset.py data/test.csv \
  --output data/processed_test.csv \
  --skip-dedup --skip-contractions

echo "[4/10] Creating train/val/dev splits (baseline + processed)"
bash split_all_data.sh

echo "[5/10] Data augmentation for Task 2 (processed data, targeted)"
uv run src/augment_data.py --config augmentation_config.yaml --tasks task2 --base-dir data --processed --seed 42
# The training scripts expect this global file path
cp data/task2/processed_train_augmented.csv data/processed_train_augmented.csv

echo "[6/10] Benchmarks (Task 1 & Task 3)"
uv run src/task1/bert_based_comparison.py
uv run src/task3/bert_based_comparison.py --baseline

echo "[7/10] Train final models (Task 1, preprocessed, no augmentation)"
uv run src/task1/train_final_model.py --model DistilBETO --processed
uv run src/task1/train_final_model.py --model Robertuito --processed
uv run src/task1/train_final_model.py --model BETO --processed

echo "[8/10] Train final models (Task 3, raw data)"
uv run src/task3/train_final_model.py --model Robertuito
uv run src/task3/train_final_model.py --model XLM-R
uv run src/task3/train_final_model.py --model LongFormer

echo "[9/10] Task 1 ensemble prediction (soft voting)"
uv run src/task1/predict_ensemble.py --voting soft

echo "[10/10] Task 2 OVR pipeline (augmented)"
for model in DistilBETO LongFormer Robertuito; do
  uv run src/task2/find_ovr_params.py --model "$model" --augment
  uv run src/task2/train_final_ovr.py --model "$model" --augment
done
uv run src/task2/predict_ensemble_ovr.py --augment

# Task 3 uses raw test set. Swap processed_test.csv temporarily.
PREPROCESSED_TEST_BACKUP="data/processed_test.preprocessed.csv"
restore_processed_test() {
  if [ -f "$PREPROCESSED_TEST_BACKUP" ]; then
    mv -f "$PREPROCESSED_TEST_BACKUP" data/processed_test.csv
  fi
}

if [ -f data/processed_test.csv ]; then
  cp data/processed_test.csv "$PREPROCESSED_TEST_BACKUP"
  trap restore_processed_test EXIT
fi
cp data/test.csv data/processed_test.csv

echo "[Task 3] Ensemble prediction (hard voting, raw test)"
uv run src/task3/predict_ensemble.py --voting hard

restore_processed_test
trap - EXIT


echo "Done. Predictions saved to task_1_predictions.csv, task_2_predictions.csv, task_3_predictions.csv"
