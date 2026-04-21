"""
Paso 2: Entrenamiento OVR Final.
Lee los epochs exactos descubiertos y entrena en 100% de los datos sin Early Stopping.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS

from augment_loading import augment_df
from bert_pooling import build_bert_like_classifier
from random_crop_collator import RandomCropDataCollator
from trainer import WeightedTrainer
from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED
set_seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Usar datos aumentados",
    )
    args = parser.parse_args()

    # Cargar parámetros descubiertos
    augmented = "_augmented" if args.augment else ""
    params_path = (
        f"../../models/task2/final_OVR/{args.model}{augmented}_ovr_params.json"
    )
    print(f"Cargando parámetros OVR desde: {params_path}")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Ejecuta primero find_ovr_params.py para {args.model}")

    with open(params_path, "r") as f:
        params = json.load(f)

    cfg = MODEL_CONFIGS[args.model]
    df = pd.read_csv("../../data/task2/processed_train.csv")
    if args.augment:
        print("Aplicando data augmentation...")
        df = augment_df(df, aug_path="../../data/processed_train_augmented.csv")
    print(df.shape)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    for label in ["sexualization", "violence", "hate"]:
        epochs = params[label]["best_epoch"]
        print(
            f"\n--- ENTRENANDO 100% DATA | {args.model} -> {label.upper()} | EPOCHS: {epochs} ---"
        )

        df_label = df.copy()
        df_label["label"] = df_label[label].astype(float).values.reshape(-1, 1).tolist()

        train_df = df_label.copy()  # 100% DE LOS DATOS
        val_df = df_label.head(10).copy()  # Dummy para evitar errores del Trainer

        n_pos = sum([x[0] for x in train_df["label"]])
        n_neg = len(train_df) - n_pos
        weights_tensor = torch.tensor([float(np.sqrt(n_neg / max(1, n_pos)))]).float()

        train_ds = Dataset.from_pandas(
            train_df[["lyrics", "label"]].rename(columns={"lyrics": "text"}),
            preserve_index=False,
        )
        val_ds = Dataset.from_pandas(
            val_df[["lyrics", "label"]].rename(columns={"lyrics": "text"}),
            preserve_index=False,
        )

        def tokenize_fn(batch, training=False):
            if training:
                return tokenizer(batch["text"], padding=False, truncation=False)
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=cfg.max_len,
            )

        train_tok = train_ds.map(
            lambda b: tokenize_fn(b, True), batched=True, remove_columns=["text"]
        ).rename_column("label", "labels")
        val_tok = val_ds.map(
            lambda b: tokenize_fn(b, False), batched=True, remove_columns=["text"]
        ).rename_column("label", "labels")
        train_tok.set_format("torch")
        val_tok.set_format("torch")

        model = build_bert_like_classifier(cfg, device, num_labels=1)
        suffix = ""
        if args.augment:
            suffix += "_augmented"
        SAVE_DIR = f"../../models/task2/final_OVR/{args.model}{suffix}_{label}"

        training_args = TrainingArguments(
            output_dir=os.path.join(SAVE_DIR, "checkpoints"),
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            num_train_epochs=epochs,  # EXACT EPOCHS FOUND IN STEP 1
            eval_strategy="no",  # NO EARLY STOPPING
            save_strategy="no",
            load_best_model_at_end=False,  # FORCE LAST EPOCH
            report_to="none",
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=RandomCropDataCollator(
                tokenizer=tokenizer, max_length=cfg.max_len
            ),
            class_weights=weights_tensor,
            loss_type="focal",
            focal_gamma=cfg.focal_gamma,
            is_multilabel=True,
        )

        trainer.train()
        os.makedirs(SAVE_DIR, exist_ok=True)
        trainer.save_model(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"✓ Modelo OVR guardado en: {SAVE_DIR}")


if __name__ == "__main__":
    main()
