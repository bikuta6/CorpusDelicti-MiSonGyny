from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.trainer_utils import PredictionOutput


def _apply_dropout_overrides(
    cfg: AutoConfig,
    classifier_dropout: float,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
) -> AutoConfig:
    """
    Apply common dropout attributes across different transformer families.
    """
    if hasattr(cfg, "classifier_dropout"):
        cfg.classifier_dropout = classifier_dropout
    if hasattr(cfg, "hidden_dropout_prob"):
        cfg.hidden_dropout_prob = hidden_dropout_prob
    if hasattr(cfg, "attention_probs_dropout_prob"):
        cfg.attention_probs_dropout_prob = attention_probs_dropout_prob

    # RoBERTa / DeBERTa / others
    if hasattr(cfg, "seq_classif_dropout"):
        cfg.seq_classif_dropout = classifier_dropout
    if hasattr(cfg, "dropout"):
        cfg.dropout = hidden_dropout_prob
    if hasattr(cfg, "attention_dropout"):
        cfg.attention_dropout = attention_probs_dropout_prob
    if hasattr(cfg, "cls_dropout"):
        cfg.cls_dropout = classifier_dropout

    return cfg


@dataclass
class BertLikeBuildConfig:
    model_id: str
    classifier_dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    ignore_mismatched_sizes: bool = False
    problem_type: str | None = None


def build_bert_like_classifier(
    cfg: Any,
    device: torch.device,
    num_labels: int = 2,
) -> nn.Module:
    """
    Builds a standard Hugging Face AutoModelForSequenceClassification from config values.
    Any former pooling-related fields in cfg are ignored.
    """
    model_cfg = AutoConfig.from_pretrained(cfg.model_id)
    model_cfg.num_labels = num_labels

    problem_type = getattr(cfg, "problem_type", None)
    if problem_type:
        model_cfg.problem_type = problem_type

    model_cfg = _apply_dropout_overrides(
        model_cfg,
        classifier_dropout=getattr(cfg, "classifier_dropout", 0.1),
        hidden_dropout_prob=getattr(cfg, "hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=getattr(cfg, "attention_probs_dropout_prob", 0.1),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        config=model_cfg,
        ignore_mismatched_sizes=getattr(cfg, "ignore_mismatched_sizes", False),
    )
    return model.to(device)


def load_bert_like_classifier(
    model_path_or_id: str,
    device: torch.device,
) -> nn.Module:
    """
    Loads a standard HF sequence classifier from a local path or Hub model id.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_id)
    model.to(device)
    model.eval()
    return model


def predict_with_chunks(
    dataset,
    tokenizer,
    model,
    device,
    max_len=512,
    batch_size=8,
    stride: int | None = None,
    aggregation="mean",
    is_multilabel=False,
):
    """
    Tokenizes texts without truncation and splits them into chunks using a sliding window.

    Window size = max_len tokens (including special tokens).
    Stride defaults to max_len // 2 (50% overlap), configurable via `stride`.

    Args:
        dataset:        HuggingFace Dataset or dict with a "text"/"lyrics" column
                        and optionally a "labels"/"label" column.
        tokenizer:      HuggingFace tokenizer matching the model.
        model:          Model with a `.logits` output (AutoModelForSequenceClassification).
        device:         torch.device to run inference on.
        max_len:        Maximum sequence length in tokens (including special tokens).
        batch_size:     Number of chunks to process per forward pass.
        stride:         Sliding window stride in tokens. Defaults to max_len // 2.
                        Smaller values → more overlap → more chunks → slower but
                        better coverage of chunk boundaries.
        aggregation:    How to combine logits across chunks: "mean" (default,
                        recommended for multiclass) or "max" (suited for multilabel).
        is_multilabel:  If True and aggregation != "max", emits a warning. Kept for
                        API compatibility; does not change aggregation automatically.

    Returns:
        PredictionOutput(predictions=np.ndarray, label_ids=np.ndarray|None, metrics=None)
    """
    import warnings

    model.eval()

    # --- is_multilabel advisory -----------------------------------------------
    if is_multilabel and aggregation != "max":
        warnings.warn(
            "is_multilabel=True was passed but aggregation is not 'max'. "
            "For multilabel tasks, aggregation='max' is typically preferred. "
            "Note: sigmoid must be applied post-hoc to the returned logits.",
            UserWarning,
            stacklevel=2,
        )

    # --- Stride default -------------------------------------------------------
    if stride is None:
        stride = max_len // 2

    if stride <= 0 or stride > max_len:
        raise ValueError(f"stride must be in (0, max_len], got {stride}")

    # --- Detect whether this tokenizer uses token_type_ids --------------------
    # BERT / DistilBERT need them; RoBERTa / Longformer / XLM-R do not.
    uses_token_type_ids = "token_type_ids" in tokenizer.model_input_names

    # --- Column resolution ----------------------------------------------------
    cols = (
        dataset.column_names
        if hasattr(dataset, "column_names")
        else list(dataset.keys())
    )
    texts = dataset["text"] if "text" in cols else dataset["lyrics"]
    label_ids = (
        dataset["labels"]
        if "labels" in cols
        else (dataset["label"] if "label" in cols else None)
    )

    all_logits = []

    for text in texts:
        # Encode without special tokens or truncation to get raw subword IDs
        # for the sliding window.
        raw_ids: list[int] = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        # Reserve room for 2 special tokens (safe default for BERT family).
        max_content = max_len - 2
        if max_content <= 0:
            raise ValueError(
                f"max_len={max_len} is too small to fit any content tokens"
            )

        # --- Handle empty text ------------------------------------------------
        if len(raw_ids) == 0:
            warnings.warn(
                "Encountered an empty text (0 tokens after encoding). "
                "Returning zero logits for this example.",
                UserWarning,
                stacklevel=2,
            )
            encoded = tokenizer(
                "",
                add_special_tokens=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=uses_token_type_ids,
                return_tensors="pt",
            )
            with torch.no_grad():
                kwargs = {
                    k: v.to(device)
                    for k, v in encoded.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                }
                outputs = model(**kwargs)
            all_logits.append(np.zeros_like(outputs.logits.cpu().numpy()[0]))
            continue

        # --- Build chunk boundaries -------------------------------------------
        chunk_starts = list(range(0, max(1, len(raw_ids) - max_content + 1), stride))
        last_start = max(0, len(raw_ids) - max_content)
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)

        chunks_input_ids = []
        chunks_attention_mask = []
        chunks_token_type_ids = []

        for start in chunk_starts:
            content = raw_ids[start : start + max_content]
            if len(content) == 0:
                continue

            cls_id = (
                tokenizer.cls_token_id
                if tokenizer.cls_token_id is not None
                else tokenizer.bos_token_id
            )
            sep_id = (
                tokenizer.sep_token_id
                if tokenizer.sep_token_id is not None
                else tokenizer.eos_token_id
            )
            chunk_ids = [cls_id] + content + [sep_id]

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            n_real = len(chunk_ids)
            n_pad = max_len - n_real
            input_ids_chunk = chunk_ids + [pad_id] * n_pad
            attention_mask_chunk = [1] * n_real + [0] * n_pad

            chunks_input_ids.append(input_ids_chunk)
            chunks_attention_mask.append(attention_mask_chunk)
            if uses_token_type_ids:
                chunks_token_type_ids.append([0] * max_len)

        # --- Batched forward passes -------------------------------------------
        doc_logits = []
        with torch.no_grad():
            for i in range(0, len(chunks_input_ids), batch_size):
                b_ids = torch.tensor(
                    chunks_input_ids[i : i + batch_size], device=device
                )
                b_mask = torch.tensor(
                    chunks_attention_mask[i : i + batch_size], device=device
                )

                kwargs = dict(input_ids=b_ids, attention_mask=b_mask)
                if uses_token_type_ids:
                    kwargs["token_type_ids"] = torch.tensor(
                        chunks_token_type_ids[i : i + batch_size], device=device
                    )

                outputs = model(**kwargs)
                doc_logits.append(outputs.logits.cpu().numpy())

        doc_logits = np.concatenate(doc_logits, axis=0)  # (n_chunks, num_labels)

        # --- Aggregate ---------------------------------------------------------
        if aggregation == "mean":
            agg_logits = doc_logits.mean(axis=0)
        elif aggregation == "max":
            agg_logits = doc_logits.max(axis=0)
        else:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. Choose 'mean' or 'max'."
            )

        all_logits.append(agg_logits)

    predictions = np.array(all_logits)
    labels = np.array(label_ids) if label_ids is not None else None

    return PredictionOutput(predictions=predictions, label_ids=labels, metrics=None)
