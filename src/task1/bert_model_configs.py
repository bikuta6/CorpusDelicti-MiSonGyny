from attr import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuración de arquitectura y entrenamiento para un modelo."""
    model_id: str

    # --- Arquitectura del clasificador ---
    classifier_dropout: float = 0.1         # Dropout en la capa de clasificación final
    attention_probs_dropout_prob: float = 0.1  # Dropout en atención (BERT/RoBERTa)
    hidden_dropout_prob: float = 0.1        # Dropout en capas ocultas (BERT/RoBERTa)

    # --- Tokenización ---
    max_len: int = 512
    use_pysentimiento_preprocess: bool = False  # Solo para Robertuito

    # --- Entrenamiento ---
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    early_stopping_patience: int = 5
    optim: str = "adamw_torch_fused"  # "adamw_torch" o "adamw_hf"

    # --- Focal Loss ---
    loss_type: str = "focal" # "standard", "weighted" o "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None

    # --- Carga del modelo ---
    ignore_mismatched_sizes: bool = False   # True para modelos con cabeza de clasificación preentrenada


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "DistilBETO": ModelConfig(
        model_id="dccuchile/distilbert-base-spanish-uncased",
        classifier_dropout=0.1,          
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=5e-6,              
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        weight_decay=0.01,               
    ),
    "BETO": ModelConfig(
        model_id="dccuchile/bert-base-spanish-wwm-cased",
        classifier_dropout=0.1,          
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=5e-6,
        warmup_ratio=0.06,
    ),
    "BETO-sentiment": ModelConfig(
        model_id="finiteautomata/beto-sentiment-analysis",
        classifier_dropout=0.1,          
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=5e-6,
        warmup_ratio=0.06,
        ignore_mismatched_sizes=True,    # ← pretrained with 3 sentiment labels
    ),
    "MarIA": ModelConfig(
        model_id="IsGarrido/roberta-base-bne",
        classifier_dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        warmup_ratio=0.0,
        max_len=512,
        weight_decay=0.01,
        learning_rate=5e-6,
    ),
    "BERT-multilingual": ModelConfig(
        model_id="nlptown/bert-base-multilingual-uncased-sentiment",
        classifier_dropout=0.1,          
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        weight_decay=0.01,
    ),
    "XLM-R": ModelConfig(
        model_id="xlm-roberta-base",
        classifier_dropout=0.15,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=512,
        learning_rate=5e-6,         # XLM-R es más sensible a lr altos
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.05,               # ↑ ayuda con estabilidad
    ),
    "Robertuito": ModelConfig(
        model_id="pysentimiento/robertuito-base-uncased",
        classifier_dropout=0.2,          # ↑ más regularización
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_len=128,                # Tweets → contexto corto
        use_pysentimiento_preprocess=True,
        learning_rate=1e-5,              # ↑ ligeramente más alto
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        warmup_ratio=0.06,              # ← añadir warmup
    ),
}
