from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class LLMConfig:
    """Configuración de arquitectura y entrenamiento para un LLM con QLoRA."""
    model_id: str

    # --- Tokenización ---
    max_len: int = 2048

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Union[str, list] = "all-linear"
    lora_bias: str = "none"

    # --- Entrenamiento ---
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 2
    optim = "adamw_torch_fused"  # "adamw_torch", "adamw_hf" o "paged_adamw_8bit"

    # --- Focal Loss ---
    loss_type: str = "focal"
    focal_gamma: float = 2.0


LLM_CONFIGS: dict[str, LLMConfig] = {
    # 8B models: batch=2, accum=8 → effective batch=16
    "Qwen-2.5-7B": LLMConfig(
        model_id="Qwen/Qwen2.5-7B",
        max_len=1024,
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    ),
    "Llama-3-8B": LLMConfig(
        model_id="meta-llama/Meta-Llama-3-8B",
        max_len=1024,
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    ),
    # 3B models: batch=4, accum=4 → effective batch=16
    "Qwen-2.5-3B": LLMConfig(
        model_id="Qwen/Qwen2.5-3B",
        max_len=1024,
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    ),
    "Llama-3.2-3B": LLMConfig(
        model_id="meta-llama/Llama-3.2-3B",
        max_len=1024,
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    ),
}
