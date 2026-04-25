"""
Entrenamiento final del LLM con todos los datos disponibles (train + val) para Task 3.
"""

import argparse
import gc
import json
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED
set_seed(SEED)


SYSTEM_PROMPT = (
    "You are a classifier for gender-based stereotypes in Spanish song lyrics. "
    "It is a binary classification task:\n"
    "- Y (Stereotypical): Lyrics represent or reinforce generalized, oversimplified, or biased beliefs about gender roles, traits, or behaviors. "
    "These stereotypes can promote assumptions about how women or men should behave, their capabilities, social roles, or emotional characteristics "
    "(e.g., women as submissive or emotional; men as dominant or emotionless).\n"
    "- N (Non-Stereotypical): Lyrics do not include generalized or biased beliefs about gender. "
    "They may mention men or women, or deal with gender-related topics, but without reinforcing restrictive roles, assumptions, or attributes linked to a specific gender.\n\n"
    "Analyze the following Spanish song lyric and answer ONLY with 'Y' or 'N'."
)


def format_instruction(example, tokenizer):
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nLyric: {example['text']}"},
        {"role": "model", "content": example["label_str"]},
    ]
    return {"text_formatted": tokenizer.apply_chat_template(messages, tokenize=False)}


class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()


def main(model_id, baseline=False):
    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task3/{pre}train_df.csv"
    VAL_PATH = f"../../data/task3/{pre}val_df.csv"

    model_name = model_id.split("/")[-1]
    suffix = "_baseline" if baseline else ""
    SAVE_DIR = f"../../models/task3/llms/{model_name}{suffix}_final"

    print("Cargando y combinando train + val...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    for df in [full_df]:
        df["label_str"] = df["label"]

    full_ds = Dataset.from_pandas(
        full_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Formateando dataset...")
    full_ds = full_ds.map(lambda x: format_instruction(x, tokenizer))

    print("Cargando modelo en 4-bits (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Use the best epoch count found during validation training
    # (e.g. if early stopping fired at epoch 2, set num_train_epochs=2)
    checkpoints_path = os.path.join(SAVE_DIR, "checkpoints")
    training_args = SFTConfig(
        output_dir=checkpoints_path,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,  # <-- set from best val run
        logging_steps=10,
        save_strategy="no",  # no checkpoints needed for final
        completion_only_loss=True,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text_formatted",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=full_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[ClearCacheCallback()],
    )

    print(f"--- ENTRENAMIENTO FINAL PARA {model_name} ---")
    trainer.train()

    print(f"Guardando modelo en {SAVE_DIR}...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Listo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    main(model_id=args.model_id, baseline=args.baseline)
