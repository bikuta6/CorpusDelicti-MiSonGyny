"""
Paso 1: Búsqueda de hiperparámetros OVR.
Encuentra el mejor Epoch y Threshold para cada etiqueta (S, V, H) usando un split 80/20 y Focal Loss.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS

from augment_loading import augment_df
from bert_pooling import build_bert_like_classifier, predict_with_chunks
from random_crop_collator import RandomCropDataCollator
from trainer import WeightedTrainer
from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED
set_seed(SEED)


def find_best_binary_threshold(true_labels, probs):
    best_t, best_f = 0.5, 0.0
    for thr in np.arange(0.0, 1.01, 0.01):
        preds = (probs >= thr).astype(int)
        f = f1_score(true_labels, preds, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = thr
    return round(best_t, 3), round(best_f, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ej. Robertuito")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Usar datos aumentados",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    df = pd.read_csv("../../data/task2/processed_train.csv")

    print(df.columns)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    params_dict = {}

    for label in ["sexualization", "violence", "hate"]:
        print(
            f"\n{'=' * 50}\nBuscando params para {args.model} -> {label.upper()}\n{'=' * 50}"
        )

        # 1. Preparar Datos (80/20 split)
        df_label = df.copy()
        df_label["label"] = df_label[label].astype(float).values.reshape(-1, 1).tolist()
        df_label = df_label[["song_id", "lyrics", "label"]]

        train_df, val_df = train_test_split(
            df_label, test_size=0.2, random_state=SEED, stratify=df_label["label"]
        )

        if args.augment:
            print("Aplicando data augmentation...")
            train_df = augment_df(
                train_df, aug_path="../../data/processed_train_augmented.csv"
            )

        # Pesos Focal Loss
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
        temp_dir = f"../../models/task2/temp/{args.model}{suffix}_{label}"

        def compute_metrics(pred):
            preds = (
                torch.sigmoid(torch.tensor(pred.predictions)).numpy() >= 0.5
            ).astype(int)
            return {"eval_f1_macro": f1_score(pred.label_ids, preds, zero_division=0)}

        training_args = TrainingArguments(
            output_dir=temp_dir,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            num_train_epochs=cfg.num_train_epochs,  # Hasta 10, cortará antes
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            save_total_limit=1,
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
            compute_metrics=compute_metrics,
            class_weights=weights_tensor,
            loss_type="focal",
            focal_gamma=cfg.focal_gamma,
            is_multilabel=True,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()

        # Extraer Best Epoch
        eval_logs = [l for l in trainer.state.log_history if "eval_f1_macro" in l]
        best_epoch = max(eval_logs, key=lambda x: x["eval_f1_macro"])["epoch"]

        # Encontrar Best Threshold en Validación
        pred_output = predict_with_chunks(
            val_ds, tokenizer, model, device, cfg.max_len, aggregation="max"
        )
        probs = torch.sigmoid(torch.tensor(pred_output.predictions)).numpy().flatten()
        true_labels = pred_output.label_ids

        best_thr, best_f1 = find_best_binary_threshold(true_labels, probs)

        print(
            f" -> Best Epoch: {int(best_epoch)} | Best Threshold: {best_thr} | Val F1: {best_f1}"
        )
        params_dict[label] = {
            "best_epoch": int(best_epoch),
            "best_threshold": best_thr,
            "val_f1": best_f1,
        }

    # Guardar parametros descubiertos
    os.makedirs("../../models/task2/final_OVR", exist_ok=True)
    with open(
        f"../../models/task2/final_OVR/{args.model}{suffix}_ovr_params.json", "w"
    ) as f:
        json.dump(params_dict, f, indent=4)
    print(f"\nParámetros guardados en {args.model}{suffix}_ovr_params.json")


if __name__ == "__main__":
    main()
