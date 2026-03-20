import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput


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
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

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
) -> nn.Module:
    """Builds the configurable pooling classifier from a model config object."""
    model = LyricsPoolingClassifier.from_backbone_pretrained(
        base_model_name_or_path=cfg.model_id,
        pooling_strategy=getattr(cfg, "pooling_strategy", "cls"),
        num_labels=2,
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
