# CorpusDelicti-MiSonGyny

Project for **misogyny detection in songs** (IberLEF MiSonGyny). It includes a full pipeline: data preparation, semantic preprocessing, data augmentation, BERT/ML training, and prediction generation for the **three competition tasks**.

## Tasks

- **Task 1 (binary)**: misogyny in lyrics → `M` / `NM`.
- **Task 2 (multi‑label)**: misogyny types → `sexualization`, `violence`, `hate` (output `S/V/H`).
- **Task 3 (binary)**: gender stereotype → `Y` / `N`.

## Repository structure

```
.
├── data/                      # raw/processed data and splits per task
├── models/                    # checkpoints and trained models
├── results/                   # result tables (csv)
├── src/                       # main scripts
│   ├── task1/                 # training/inference task1
│   ├── task2/                 # training/inference task2 (+ OVR)
│   ├── task3/                 # training/inference task3
│   ├── preprocess_dataset.py  # lyrics preprocessing
│   ├── create_data_per_task.py
│   ├── create_val_test_split.py
│   ├── augment_data.py        # data augmentation
│   └── augment_loading.py     # merge augmented data
├── run_bert.sh                # BERT comparisons (tasks 1‑3)
├── run_bert_aug.sh            # BERT comparisons with augmentation
├── run_bert_baselines.sh      # BERT baselines
├── run_ml.sh                  # ML baselines (TF‑IDF + classic models)
├── run_ml_baselines.sh        # ML baselines without preprocessing
└── README.md
```

## Requirements

- **Python 3.11** (see `.python-version`).
- **uv** is recommended for reproducibility (see `uv.lock`).

### Install with `uv` (recommended)

```bash
uv sync
```

### Install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Extra (ML baselines with spaCy)

The `ml_comparison.py` scripts require the Spanish spaCy model:

```bash
uv run python -m spacy download es_core_news_sm
```

## Data

Expected files: `data/train.csv` and `data/test.csv` with at least these columns:

- `song_id`, `song_title`, `artist_id`, `artist_name`, `lyrics`, `language`
- Labels:
  - Task 1: `is_misogynistic`
  - Task 2: `type_sexualization`, `type_violence`, `type_hate`
  - Task 3: `has_gender_stereotype`

`src/create_data_per_task.py` generates:

- `data/task1/train.csv`
- `data/task2/train.csv`
- `data/task3/train.csv`

## Methodology (summary)

This is a condensed, practical summary of the methodology described in `paper.tex`:

- **Long sequence handling**
  - *Training:* randomized window sampling (random crop up to `MAX_LEN`) to expose different lyric segments per epoch.
  - *Inference:* overlapping sliding windows with **max aggregation** across chunks (stride `MAX_LEN // 2`).
- **ML baselines**
  - TF‑IDF (max 5k features) over lemmatized Spanish text (spaCy `es_core_news_sm`), tuned via genetic search.
- **Transformer benchmarking**
  - BETO, DistilBETO, RoBERTuito, MarIA, XLM‑R, Longformer with standard AdamW‑based training and early stopping.
- **Validation threshold tuning**
  - Per‑model threshold search on validation set; soft‑voting ensembles average these thresholds.
- **Text preprocessing**
  - Tasks 1–2: punctuation mapping to preserve verse/stanza structure; contractions/dedup *not* used in final setup.
  - Task 3: raw text performs best.
- **Data augmentation (Task 2)**
  - Targeted only to positive labels; 80% LLM paraphrase + 20% AEDA punctuation noise.
- **Ensembles**
  - *Task 1:* DistilBETO + BETO + RoBERTuito (soft voting).
  - *Task 2 (OVR):* DistilBETO + Longformer + RoBERTuito (hard voting per label).
  - *Task 3:* RoBERTuito + XLM‑R + Longformer (hard voting).
- **Final training for submissions**
  - Train on full data; use a tiny dummy eval split to satisfy the Trainer API.

## Recommended pipeline (end‑to‑end)

1) **Split by task**
```bash
./generate_task_data.sh
```

2) **Preprocess lyrics** (normalize contractions + semantic deduplication)
```bash
./preprocess_all_data.sh
# Options: --skip-dedup / --skip-contractions
```

3) **Create train/val/dev splits**
```bash
./split_all_data.sh
```

4) **Data augmentation (optional)**
```bash
uv run src/augment_data.py --config augmentation_config.yaml
# Use --processed to start from processed_train.csv
```
> Note: `llm_paraphrase` uses a local **Ollama** server at `http://localhost:11434`.

5) **Train and compare**

- BERT (comparisons):
```bash
./run_bert.sh
./run_bert_aug.sh
./run_bert_baselines.sh
```

- ML baselines:
```bash
./run_ml.sh
./run_ml_baselines.sh
```

## Reproducible run (full pipeline)

If you want to run everything in reproducible mode with this configuration (Task1/2 preprocessed without dedup/contr, Task3 unprocessed, **augmentation in Task 2** on processed data, ensembles as in the paper), use:

```bash
./run_reproducible.sh
```

> Note: augmentation uses `llm_paraphrase` with Ollama at `http://localhost:11434`.
>
> Ensemble configurations:
> - **Task 1:** DistilBETO + BETO + Robertuito (soft voting, averaged validation threshold).
> - **Task 2 (OVR):** DistilBETO + LongFormer + Robertuito (hard voting, per‑model/label thresholds).
> - **Task 3:** Robertuito + XLM-R + LongFormer (hard voting, per‑model thresholds).

## Per‑task training

- **Task 1**
```bash
uv run src/task1/train_model.py --model BETO
```

- **Task 2 (multi‑label)**
```bash
uv run src/task2/train_model.py --model BETO
```

- **Task 2 (One‑Vs‑Rest / OVR)**
```bash
uv run src/task2/find_ovr_params.py --model Robertuito
uv run src/task2/train_final_ovr.py --model Robertuito
```

- **Task 3**
```bash
uv run src/task3/train_model.py --model BETO
```

## Prediction / Submission

- **Task 1 (single model)**
```bash
uv run src/task1/predict.py \
  --model_path models/task1/single/BETO \
  --test_file data/processed_test.csv \
  --output_file task_1_predictions.csv
```

- **Task 1 (ensemble)**
```bash
uv run src/task1/predict_ensemble.py --voting soft
```

- **Task 2 (OVR ensemble)**
```bash
uv run src/task2/predict_ensemble_ovr.py
```

- **Task 3 (ensemble)**
```bash
uv run src/task3/predict_ensemble.py --voting hard
```

Output files are saved at the repo root:

- `task_1_predictions.csv`
- `task_2_predictions.csv`
- `task_3_predictions.csv`

## Results and models

- **Results**: `results/task*/` (CSV tables for paper/comparisons).
- **Models**: `models/task*/` (checkpoints and final models).

## Useful notes

- Preprocessing uses `SentenceTransformer` (`intfloat/multilingual-e5-large`), which can be expensive on CPU.
- If you don’t have a GPU, consider `--skip-dedup` during preprocessing or reduce batch sizes.
- Global seed is configured in `src/utils.py`.

---
