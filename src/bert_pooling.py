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
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden = outputs.last_hidden_state

        cls_repr = last_hidden[:, 0]
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.size(), dtype=torch.long, device=input_ids.device
            )
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
    aggregation="max",
    is_multilabel=False,
):
    """
    Tokenizes texts without truncation and splits them into chunks using a sliding window.
    Window size = max_len, stride = max_len // 4.
    Returns PredictionOutput with aggregated logits for each text.
    """
    model.eval()
    stride = max_len // 4

    all_logits = []

    # Safely get column names whether it's a HuggingFace Dataset or a dict
    cols = dataset.column_names if hasattr(dataset, "column_names") else dataset.keys()

    texts = dataset["text"] if "text" in cols else dataset["lyrics"]
    label_ids = (
        dataset["labels"]
        if "labels" in cols
        else (dataset["label"] if "label" in cols else None)
    )

    # Process text by text to keep track of chunks per document easily
    for text in texts:
        tokens = tokenizer(
            text, add_special_tokens=False, truncation=False, padding=False
        )
        input_ids = tokens["input_ids"]

        chunks_input_ids = []
        chunks_attention_mask = []

        cls_token = tokenizer.cls_token_id
        sep_token = tokenizer.sep_token_id
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Room for CLS and SEP
        max_content_len = max_len - 2

        if len(input_ids) <= max_content_len:
            # Single chunk
            chunk_ids = [cls_token] + input_ids + [sep_token]
            chunk_mask = [1] * len(chunk_ids)

            # Pad
            pad_len = max_len - len(chunk_ids)
            if pad_len > 0:
                chunk_ids += [pad_token] * pad_len
                chunk_mask += [0] * pad_len

            chunks_input_ids.append(chunk_ids)
            chunks_attention_mask.append(chunk_mask)
        else:
            # Sliding window
            for i in range(0, len(input_ids), stride):
                content = input_ids[i : i + max_content_len]
                chunk_ids = [cls_token] + content + [sep_token]
                chunk_mask = [1] * len(chunk_ids)

                pad_len = max_len - len(chunk_ids)
                if pad_len > 0:
                    chunk_ids += [pad_token] * pad_len
                    chunk_mask += [0] * pad_len

                chunks_input_ids.append(chunk_ids)
                chunks_attention_mask.append(chunk_mask)

                if i + max_content_len >= len(input_ids):
                    break

        # Batch predict for this document's chunks
        doc_logits = []
        with torch.no_grad():
            for i in range(0, len(chunks_input_ids), batch_size):
                b_input_ids = torch.tensor(chunks_input_ids[i : i + batch_size]).to(
                    device
                )
                b_attention_mask = torch.tensor(
                    chunks_attention_mask[i : i + batch_size]
                ).to(device)

                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
                logits = outputs.logits
                doc_logits.append(logits.cpu().numpy())

        doc_logits = np.concatenate(doc_logits, axis=0)

        # Aggregate predictions for the document
        if aggregation == "mean":
            agg_logits = np.mean(doc_logits, axis=0)
        elif aggregation == "max":
            agg_logits = np.max(doc_logits, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        all_logits.append(agg_logits)

    predictions = np.array(all_logits)
    labels = np.array(label_ids) if label_ids is not None else None

    return PredictionOutput(predictions=predictions, label_ids=labels, metrics=None)
