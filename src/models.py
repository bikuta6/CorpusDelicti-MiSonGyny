import os
import threading

os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"

# Silenciar errores de threads de conversión safetensors
_original_excepthook = threading.excepthook
def _silent_excepthook(args):
    if "safetensors" in str(args.exc_value) or "JSONDecodeError" in str(type(args.exc_value)):
        pass
    else:
        _original_excepthook(args)
threading.excepthook = _silent_excepthook

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class FocalLoss(nn.Module):
    """Focal Loss para clases desbalanceadas"""
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# Modelos que NO soportan add_pooling_layer
MODELS_WITHOUT_POOLING_ARG = {"deberta", "deberta-v2", "distilbert"}

# Modelos que NO usan token_type_ids
MODELS_WITHOUT_TOKEN_TYPE_IDS = {"deberta", "deberta-v2", "xlm-roberta", "roberta", "distilbert"}


class MisogynyClassifier(nn.Module):
    def __init__(self, model_name_or_path, num_labels=2, dropout_rate=0.2, 
                 class_weights=None, is_multilabel=False, 
                 use_focal_loss=False, focal_gamma=2.0, label_smoothing=0.0,
                 pooling_strategy="cls"):  # Nuevo parámetro
        super(MisogynyClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.num_labels = num_labels
        self.is_multilabel = is_multilabel
        
        # Detectar tipo de modelo
        model_type = self.config.model_type.lower()
        self.skip_token_type_ids = any(t in model_type for t in MODELS_WITHOUT_TOKEN_TYPE_IDS)
        
        # Cargar backbone con parámetros correctos según arquitectura
        if any(t in model_type for t in MODELS_WITHOUT_POOLING_ARG):
            self.backbone = AutoModel.from_pretrained(
                model_name_or_path,
                ignore_mismatched_sizes=True
            )
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name_or_path,
                ignore_mismatched_sizes=True,
                add_pooling_layer=False
            )
        
        self.hidden_size = self.config.hidden_size
        
        # Clasificador propio (no depende del pooler del modelo)
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_labels)
        )

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Selección de Loss
        if self.is_multilabel:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        elif use_focal_loss:
            self.loss_fct = FocalLoss(gamma=focal_gamma, weight=self.class_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights, 
                label_smoothing=label_smoothing
            )

        self.pooling_strategy = pooling_strategy

    def _pool(self, last_hidden_state, attention_mask):
        """Diferentes estrategias de pooling"""
        if self.pooling_strategy == "cls":
            # CLS token (default, más común)
            return last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling (mejor para textos largos como canciones)
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == "max":
            # Max pooling
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state[mask == 0] = -1e9
            return torch.max(last_hidden_state, dim=1)[0]
        
        elif self.pooling_strategy == "cls_mean":
            # Combinación CLS + Mean (robusto)
            cls = last_hidden_state[:, 0, :]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            mean = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            return (cls + mean) / 2
        
        else:
            return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Preparar argumentos según arquitectura
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs
        }
        
        # Solo pasar token_type_ids si el modelo lo soporta
        if not self.skip_token_type_ids and token_type_ids is not None:
            forward_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.backbone(**forward_kwargs)
        
        # Usar estrategia de pooling
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        
        logits = self.classifier_head(pooled)
        
        loss = None
        if labels is not None:
            if self.is_multilabel:
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )