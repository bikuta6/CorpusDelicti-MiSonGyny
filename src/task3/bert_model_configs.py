from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuración de arquitectura y entrenamiento para un modelo."""

    model_id: str

    # --- Arquitectura del clasificador ---
    classifier_dropout: float = 0.1  # Dropout en la capa de clasificación final
    attention_probs_dropout_prob: float = 0.1  # Dropout en atención (BERT/RoBERTa)
    hidden_dropout_prob: float = 0.1  # Dropout en capas ocultas (BERT/RoBERTa)

    # --- Tokenización ---
    max_len: int = 512
    use_pysentimiento_preprocess: bool = False  # Solo para Robertuito

    # --- Entrenamiento ---
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.0
    lr_scheduler_type: str = "linear"
    early_stopping_patience: int = 3
    optim: str = "adamw_torch_fused"  # "adamw_torch" o "adamw_hf"

    # --- Focal Loss ---
    loss_type: str = "standard"  # "standard", "weighted" o "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None

    # --- Carga del modelo ---
    ignore_mismatched_sizes: bool = (
        False  # True para modelos con cabeza de clasificación preentrenada
    )


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "DistilBETO": ModelConfig(
        model_id="dccuchile/distilbert-base-spanish-uncased",
        classifier_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=2e-5,
    ),
    "BETO": ModelConfig(
        model_id="dccuchile/bert-base-spanish-wwm-cased",
        classifier_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=5e-6,
    ),
    "MarIA": ModelConfig(
        model_id="IsGarrido/roberta-base-bne",
        classifier_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=3e-5,
    ),
    "XLM-R": ModelConfig(
        model_id="xlm-roberta-base",
        classifier_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=2e-5,  # XLM-R es más sensible a lr altos
    ),
    "Robertuito": ModelConfig(
        model_id="pysentimiento/robertuito-base-uncased",
        classifier_dropout=0.1,  # ↑ más regularización
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=128,  # Tweets → contexto corto
        use_pysentimiento_preprocess=True,
        learning_rate=2e-5,  # ↑ ligeramente más alto
    ),
    "LongFormer": ModelConfig(
        model_id="mrm8488/longformer-base-4096-spanish",  # or  Buzzeitor/longformer-base-4096-bne-es
        classifier_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=1024,  # Para textos largos (canciones)
        learning_rate=2e-5,
    ),
}


def apply_baseline_settings():
    """Modifica todas las configuraciones para usar los parámetros del baseline."""
    for config in MODEL_CONFIGS.values():
        config.loss_type = "standard"
