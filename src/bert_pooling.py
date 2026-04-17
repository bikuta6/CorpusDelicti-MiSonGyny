import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import PredictionOutput


class LyricsPoolingConfig(PretrainedConfig):
    """Configuration for a BERT-like sequence classifier with configurable pooling."""

    model_type = "lyrics_pooling_classifier"

    def __init__(
        self,
        base_model_name_or_path: str = "",
        pooling_strategy: str = "cls",
        classifier_dropout: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.pooling_strategy = pooling_strategy
        self.classifier_dropout = classifier_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class LyricsPoolingClassifier(PreTrainedModel):
    """Classifier head over an AutoModel backbone with cls/mean/cls_mean pooling."""

    config_class = LyricsPoolingConfig
    base_model_prefix = "backbone"

    def __init__(self, config: LyricsPoolingConfig):
        super().__init__(config)
        if not config.base_model_name_or_path:
            raise ValueError(
                "base_model_name_or_path must be set in LyricsPoolingConfig"
            )
        backbone_cfg = AutoConfig.from_pretrained(config.base_model_name_or_path)
        backbone_cfg.num_labels = config.num_labels

        if hasattr(backbone_cfg, "classifier_dropout"):
            backbone_cfg.classifier_dropout = config.classifier_dropout
        if hasattr(backbone_cfg, "hidden_dropout_prob"):
            backbone_cfg.hidden_dropout_prob = config.hidden_dropout_prob
        if hasattr(backbone_cfg, "attention_probs_dropout_prob"):
            backbone_cfg.attention_probs_dropout_prob = (
                config.attention_probs_dropout_prob
            )
        if hasattr(backbone_cfg, "seq_classif_dropout"):
            backbone_cfg.seq_classif_dropout = config.classifier_dropout
        if hasattr(backbone_cfg, "dropout"):
            backbone_cfg.dropout = config.hidden_dropout_prob
        if hasattr(backbone_cfg, "attention_dropout"):
            backbone_cfg.attention_dropout = config.attention_probs_dropout_prob
        if hasattr(backbone_cfg, "cls_dropout"):
            backbone_cfg.cls_dropout = config.classifier_dropout

        self.backbone = AutoModel.from_config(backbone_cfg)

        hidden_size = getattr(backbone_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(backbone_cfg, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config")

        strategy = config.pooling_strategy
        if strategy not in {"cls", "mean", "cls_mean"}:
            raise ValueError(
                "pooling_strategy must be one of: 'cls', 'mean', 'cls_mean'"
            )

        in_features = hidden_size * 2 if strategy == "cls_mean" else hidden_size
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(in_features, config.num_labels)

        self.post_init()

    @classmethod
    def from_backbone_pretrained(
        cls,
        base_model_name_or_path: str,
        pooling_strategy: str = "cls",
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        ignore_mismatched_sizes: bool = False,
    ) -> "LyricsPoolingClassifier":
        config = LyricsPoolingConfig(
            base_model_name_or_path=base_model_name_or_path,
            pooling_strategy=pooling_strategy,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        model = cls(config)

        pretrained_backbone = AutoModel.from_pretrained(
            base_model_name_or_path,
            config=model.backbone.config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        model.backbone = pretrained_backbone
        return model

    @staticmethod
    def _masked_mean(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.size(), dtype=torch.long, device=input_ids.device
            )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden = outputs.last_hidden_state

        cls_repr = last_hidden[:, 0]

        mean_repr = self._masked_mean(last_hidden, attention_mask)

        if self.config.pooling_strategy == "cls":
            pooled = cls_repr
        elif self.config.pooling_strategy == "mean":
            pooled = mean_repr
        else:
            pooled = torch.cat([cls_repr, mean_repr], dim=-1)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        # We bypass internal loss calculation to allow the Trainer (WeightedTrainer)
        # to correctly apply focal or multi-label BCE loss using compute_loss()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class BertLikeBuildConfig:
    model_id: str
    pooling_strategy: str = "cls"
    classifier_dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    ignore_mismatched_sizes: bool = False


def build_bert_like_classifier(
    cfg: Any,
    device: torch.device,
    num_labels: int = 2,
) -> nn.Module:
    """Builds the configurable pooling classifier from a model config object."""
    model = LyricsPoolingClassifier.from_backbone_pretrained(
        base_model_name_or_path=cfg.model_id,
        pooling_strategy=getattr(cfg, "pooling_strategy", "cls"),
        num_labels=num_labels,
        classifier_dropout=cfg.classifier_dropout,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        ignore_mismatched_sizes=cfg.ignore_mismatched_sizes,
    )
    return model.to(device)


def load_bert_like_classifier(
    model_path_or_id: str,
    device: torch.device,
) -> nn.Module:
    """Loads either the custom pooling model or a standard HF sequence classifier."""
    model = None

    if os.path.isdir(model_path_or_id):
        try:
            model = LyricsPoolingClassifier.from_pretrained(model_path_or_id)
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(model_path_or_id)
    else:
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
        model:          Model with a `.logits` output (LyricsPoolingClassifier or
                        AutoModelForSequenceClassification).
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
        # for the sliding window. Special tokens and padding are added per chunk
        # below via prepare_for_model, which avoids the lossy round-trip of
        # convert_ids_to_tokens → re-tokenize.
        raw_ids: list[int] = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        # How many content tokens fit after reserving space for special tokens.
        # -2 is conservative and correct for BERT-family (CLS + SEP); models
        # that add only one special token will simply have one extra padding
        # token — harmless.
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
            # Run a single forward pass on an all-padding input so the output
            # shape (num_labels) is inferred from the model rather than hardcoded.
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

        # Guarantee the tail is always covered even when len(raw_ids) is not a
        # multiple of stride.
        last_start = max(0, len(raw_ids) - max_content)
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)

        chunks_input_ids = []
        chunks_attention_mask = []
        chunks_token_type_ids = []

        for start in chunk_starts:
            content = raw_ids[start : start + max_content]

            if len(content) == 0:
                # Should never happen after the last_start guard above.
                continue

            # Manually assemble input_ids with special tokens from the raw ID
            # slice.
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

            # Pad to max_len with the tokenizer pad token id.
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            n_real = len(chunk_ids)
            n_pad = max_len - n_real
            input_ids_chunk = chunk_ids + [pad_id] * n_pad
            # Attention mask: 1 for real tokens, 0 for padding.
            attention_mask_chunk = [1] * n_real + [0] * n_pad

            chunks_input_ids.append(input_ids_chunk)
            chunks_attention_mask.append(attention_mask_chunk)
            if uses_token_type_ids:
                # BERT-family: all zeros for single-sequence input.
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

        # --- Aggregate --------------------------------------------------------
        # "mean" is the correct default for multiclass: averaging logits across
        # chunks is equivalent to voting and produces a coherent softmax.
        # "max" is appropriate for multilabel: it selects the most confident
        # positive signal across chunks for each label independently.
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
