"""
Unified checker to validate that configured loss mode and class weights are being used
for tasks 1, 2, and 3.

Usage examples:
    python src/check_loss_and_weights.py --task 1 --model BETO
    python src/check_loss_and_weights.py --task 2 --model BETO --baseline
    python src/check_loss_and_weights.py --task 3 --model MarIA --batch-size 4
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoTokenizer, TrainingArguments

# Make src imports work when launched from project root
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from bert_pooling import build_bert_like_classifier  # noqa: E402
from random_crop_collator import RandomCropDataCollator  # noqa: E402
from trainer import WeightedTrainer  # noqa: E402
from utils import DEFAULT_SEED, set_seed  # noqa: E402

set_seed(DEFAULT_SEED)


def _import_task_configs(task: int):
    mod = importlib.import_module(f"task{task}.bert_model_configs")
    return mod.MODEL_CONFIGS, mod.apply_baseline_settings, mod.ModelConfig


def _paths_for_task(task: int, baseline: bool) -> tuple[str, list[str]]:
    pre = "" if baseline else "processed_"
    train_path = os.path.join("data", f"task{task}", f"{pre}train_df.csv")
    if task == 2:
        label_cols = ["sexualization", "violence", "hate"]
    else:
        label_cols = ["label"]
    return train_path, label_cols


def _map_binary_labels(df: pd.DataFrame, task: int) -> pd.DataFrame:
    out = df.copy()
    if task == 1:
        out["label"] = out["label"].map({"NM": 0, "M": 1})
    elif task == 3:
        out["label"] = out["label"].map({"N": 0, "Y": 1})
    return out


def _create_task2_label_column(df: pd.DataFrame) -> pd.Series:
    return df[["sexualization", "violence", "hate"]].astype(int).values.tolist()


def _compute_weights(train_df: pd.DataFrame, task: int) -> torch.Tensor:
    if task in (1, 3):
        train_originals_labels = (
            train_df[train_df["augmentation"] == "original"]["label"]
            if "augmentation" in train_df.columns
            else train_df["label"]
        )
        n_pos = int((train_originals_labels == 1).sum())
        n_neg = int((train_originals_labels == 0).sum())
        total = n_neg + n_pos
        # same formula used by train scripts
        w0 = total / (2 * max(1, n_neg))
        w1 = total / (2 * max(1, n_pos))
        weights_tensor = torch.sqrt(torch.tensor([w0, w1]).float())
        return weights_tensor

    # task 2 multilabel
    train_originals_labels = (
        train_df[train_df["augmentation"] == "original"]["label"]
        if "augmentation" in train_df.columns
        else train_df["label"]
    )
    n_samples = len(train_originals_labels)
    n_pos = np.array(
        [train_originals_labels.apply(lambda x: int(x[i])).sum() for i in range(3)]
    )
    n_neg = n_samples - n_pos
    weights_tensor = torch.sqrt(torch.tensor(n_neg / np.maximum(1, n_pos)).float())
    return weights_tensor


def _make_tokenize_fn(cfg: Any, task: int):
    def tokenize_fn(batch: dict[str, list[str]], tokenizer):
        texts = batch["text"]
        if getattr(cfg, "use_pysentimiento_preprocess", False):
            texts = [preprocess_tweet(t, lang="es") for t in texts]
        # mimic train behavior: no truncation during train
        return tokenizer(texts, padding=False, truncation=False)

    return tokenize_fn


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _safe_cfg_dump(cfg: Any) -> dict[str, Any]:
    try:
        return asdict(cfg)
    except Exception:
        # fallback for non-dataclass objects
        keys = [k for k in dir(cfg) if not k.startswith("_")]
        out = {}
        for k in keys:
            v = getattr(cfg, k, None)
            if not callable(v):
                out[k] = v
        return out


def _label_distribution(train_df: pd.DataFrame, task: int) -> str:
    if task in (1, 3):
        n0 = int((train_df["label"] == 0).sum())
        n1 = int((train_df["label"] == 1).sum())
        total = n0 + n1
        p1 = (n1 / total) if total else 0.0
        return f"binary labels -> neg={n0}, pos={n1}, pos_ratio={p1:.4f}"

    arr = np.array(train_df["label"].tolist(), dtype=int)
    totals = arr.shape[0]
    pos = arr.sum(axis=0).tolist()
    ratio = (arr.mean(axis=0)).tolist()
    return (
        "multilabel -> "
        f"samples={totals}, "
        f"pos_counts={{sexualization:{pos[0]}, violence:{pos[1]}, hate:{pos[2]}}}, "
        f"pos_ratios={{sexualization:{ratio[0]:.4f}, violence:{ratio[1]:.4f}, hate:{ratio[2]:.4f}}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check configured loss mode and class weights are active for tasks 1-3."
    )
    parser.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--model", type=str, default="BETO")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="How many train samples to keep for quick check.",
    )
    parser.add_argument(
        "--compare-losses",
        action="store_true",
        help="Additionally compute standard, weighted, and focal losses on the same probe batch.",
    )
    args = parser.parse_args()

    MODEL_CONFIGS, apply_baseline_settings, _ModelConfig = _import_task_configs(
        args.task
    )
    if args.baseline:
        apply_baseline_settings()

    if args.model not in MODEL_CONFIGS:
        raise ValueError(
            f"Model '{args.model}' not found for task {args.task}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    cfg = MODEL_CONFIGS[args.model]

    train_path, _label_cols = _paths_for_task(args.task, args.baseline)
    project_root = os.path.abspath(os.path.join(SRC_DIR, ".."))
    abs_train_path = os.path.abspath(os.path.join(project_root, train_path))
    if not os.path.exists(abs_train_path):
        raise FileNotFoundError(f"Train CSV not found: {abs_train_path}")

    _print_header(f"Task {args.task} | Model {args.model} | baseline={args.baseline}")
    print(f"Train path: {abs_train_path}")

    train_df = pd.read_csv(abs_train_path)
    train_df = _map_binary_labels(train_df, args.task)
    if args.task == 2:
        train_df["label"] = _create_task2_label_column(train_df)

    if "lyrics" not in train_df.columns:
        raise ValueError("Expected 'lyrics' column in train dataframe.")

    # keep a subset for speed
    if args.max_samples > 0 and len(train_df) > args.max_samples:
        train_df = train_df.sample(
            args.max_samples, random_state=DEFAULT_SEED
        ).reset_index(drop=True)

    print(_label_distribution(train_df, args.task))

    weights_tensor = _compute_weights(train_df, args.task)
    print(f"Computed class weights tensor: {weights_tensor.tolist()}")

    ds = Dataset.from_pandas(
        train_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    tokenize_fn = _make_tokenize_fn(cfg, args.task)
    tok = ds.map(
        lambda batch: tokenize_fn(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )
    tok = tok.rename_column("label", "labels")
    tok.set_format("torch")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    num_labels = 3 if args.task == 2 else 2
    model = build_bert_like_classifier(cfg, device=device, num_labels=num_labels)

    collator = RandomCropDataCollator(tokenizer=tokenizer, max_length=cfg.max_len)

    out_dir = os.path.abspath(os.path.join(project_root, "results", "tmp_loss_check"))
    os.makedirs(out_dir, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=1,
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
    )

    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=tok,
        eval_dataset=tok.select(range(min(len(tok), args.batch_size))),
        data_collator=collator,
        compute_metrics=None,
        class_weights=weights_tensor,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        is_multilabel=(args.task == 2),
    )

    _print_header("Configuration and Trainer Wiring")
    cfg_dump = _safe_cfg_dump(cfg)
    print(f"cfg.loss_type: {cfg_dump.get('loss_type')}")
    print(f"cfg.focal_gamma: {cfg_dump.get('focal_gamma')}")
    print(f"cfg.focal_alpha: {cfg_dump.get('focal_alpha')}")
    print(f"trainer.loss_type: {trainer.loss_type}")
    print(f"trainer.is_multilabel: {trainer.is_multilabel}")
    print(
        "trainer.class_weights: "
        f"{trainer.class_weights.detach().cpu().tolist() if trainer.class_weights is not None else None}"
    )
    print(f"model.config.num_labels: {getattr(model.config, 'num_labels', None)}")
    print(f"model.config.problem_type: {getattr(model.config, 'problem_type', None)}")

    _print_header("Single-batch Loss Probe")
    train_loader = trainer.get_train_dataloader()
    first_batch = next(iter(train_loader))
    first_batch = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in first_batch.items()
    }
    if args.task == 2:
        first_batch["labels"] = first_batch["labels"].float()

    model.eval()
    with torch.no_grad():
        loss, outputs = trainer.compute_loss(model, first_batch, return_outputs=True)

    logits = outputs.logits
    labels = first_batch["labels"]

    print(f"batch keys: {list(first_batch.keys())}")
    print(f"labels shape: {tuple(labels.shape)} | dtype={labels.dtype}")
    print(f"logits shape: {tuple(logits.shape)} | dtype={logits.dtype}")
    print(f"computed loss ({trainer.loss_type}): {float(loss.detach().cpu()):.6f}")

    if args.task in (1, 3):
        preds = logits.argmax(dim=-1)
        pos_rate = float((preds == 1).float().mean().detach().cpu())
        print(f"argmax positive-rate (probe batch): {pos_rate:.4f}")
    else:
        probs = torch.sigmoid(logits)
        pos_rate = probs.mean(dim=0).detach().cpu().numpy().tolist()
        print(
            "mean sigmoid per label (probe batch): "
            f"sexualization={pos_rate[0]:.4f}, violence={pos_rate[1]:.4f}, hate={pos_rate[2]:.4f}"
        )

    if args.compare_losses:
        _print_header("Compare Losses on Same Probe Batch")
        loss_results: dict[str, float] = {}

        for loss_name in ("standard", "weighted", "focal"):
            cmp_trainer = WeightedTrainer(
                model=model,
                args=train_args,
                train_dataset=tok,
                eval_dataset=tok.select(range(min(len(tok), args.batch_size))),
                data_collator=collator,
                compute_metrics=None,
                class_weights=weights_tensor,
                loss_type=loss_name,
                focal_gamma=cfg.focal_gamma,
                focal_alpha=cfg.focal_alpha,
                is_multilabel=(args.task == 2),
            )
            with torch.no_grad():
                cmp_loss = cmp_trainer.compute_loss(
                    model, first_batch, return_outputs=False
                )
            loss_results[loss_name] = float(cmp_loss.detach().cpu())

        print(
            "loss(standard)={:.6f} | loss(weighted)={:.6f} | loss(focal)={:.6f}".format(
                loss_results["standard"],
                loss_results["weighted"],
                loss_results["focal"],
            )
        )
        print(
            "These were computed on the exact same batch so differences come only from the loss formulation/weights."
        )

    _print_header("DONE")
    print(
        "If trainer.loss_type and trainer.class_weights match cfg/computed values, loss wiring is correct."
    )


if __name__ == "__main__":
    main()
