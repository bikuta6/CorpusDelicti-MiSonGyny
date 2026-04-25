"""
Entrenamiento y evaluación para un modelo generativo (LLM) con Instruction-Tuning en Task 3.
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import DEFAULT_SEED, set_seed

SEED = DEFAULT_SEED
set_seed(SEED)


def predict_chunks(model, tokenizer, full_text, device, max_length=512, overlap=256):
    """
    Predict 'Y'/'N' for a long text by chunking it with overlapping windows.

    Args:
        model: The language model
        tokenizer: The tokenizer
        full_text: The input text to classify
        device: The device to run inference on
        max_length: Maximum length of each chunk (in characters)
        overlap: Overlap between consecutive chunks (stride = max_length - overlap)

    Returns:
        Binary prediction: 0 ('N') or 1 ('Y') based on majority voting
    """
    label_map = {0: "N", 1: "Y"}

    # Create overlapping chunks
    stride = max_length - overlap
    chunks = []
    for i in range(0, len(full_text), stride):
        chunk = full_text[i : i + max_length]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        if i + max_length >= len(full_text):
            break
    chunk_preds = []

    for chunk in chunks:
        prompt = format_inference_instruction(chunk, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        if "Y" in generated_text.upper():
            chunk_preds.append(1)
        else:
            chunk_preds.append(0)

    # Majority vote across chunks
    return max(chunk_preds)


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


def format_inference_instruction(text, tokenizer):
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nLyric: {text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


from transformers import TrainerCallback


class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()


def main(model_id, baseline=False):
    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task3/{pre}train_df.csv"
    VAL_PATH = f"../../data/task3/{pre}val_df.csv"
    DEV_PATH = f"../../data/task3/{pre}dev_df.csv"

    model_name = model_id.split("/")[-1]
    suffix = "_baseline" if baseline else ""
    SAVE_DIR = f"../../models/task3/llms/{model_name}{suffix}"

    print(f"Cargando datos de entrenamiento desde {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    dev_df = pd.read_csv(DEV_PATH)

    # Mantener labels como strings para generación
    for df in [train_df, val_df, dev_df]:
        if "label" in df.columns:
            df["label_str"] = df["label"]
            df["label_id"] = df["label"].map({"N": 0, "Y": 1})

    train_ds = Dataset.from_pandas(
        train_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    val_ds = Dataset.from_pandas(
        val_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )
    dev_ds = Dataset.from_pandas(
        dev_df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Formateando datasets...")
    train_ds = train_ds.map(lambda x: format_instruction(x, tokenizer))
    val_ds = val_ds.map(lambda x: format_instruction(x, tokenizer))

    print("Cargando modelo en 4-bits (QLoRA) para optimizar 12GB VRAM...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
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

    checkpoints_path = os.path.join(SAVE_DIR, "checkpoints")

    training_args = SFTConfig(
        output_dir=checkpoints_path,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,  # reduce eval batch size
        eval_accumulation_steps=8,  # accumulate eval in smaller chunks
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        completion_only_loss=True,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        dataset_text_field="text_formatted",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=1),
            ClearCacheCallback(),
        ],
    )

    print(f"--- INICIANDO ENTRENAMIENTO SFT PARA {model_name} ---")
    trainer.train()

    print("Guardando adaptador...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Evaluando en Dev Set...")
    model.eval()
    model.config.use_cache = True

    preds = []
    true_labels = dev_ds["label_id"]

    for item in dev_ds:
        prompt = format_inference_instruction(item["text"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
            )

        # Decode only the generated response
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        # Parse output
        if "Y" in generated_text.upper():
            preds.append(1)
        else:
            preds.append(0)

    precision, recall, f1_opt, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0.0
    )
    acc_opt = accuracy_score(true_labels, preds)

    results = {
        "Modelo": model_id,
        "F1-Macro": f1_opt,
        "Accuracy": acc_opt,
        "Precision": precision,
        "Recall": recall,
    }

    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nModelo guardado en: {SAVE_DIR}")
    print(f"Resultados en Dev: {results}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Entrenamiento de un LLM mediante Instruction Finetuning (QLoRA) para Task 3"
    )
    arg_parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-2b-it",
        help="ID del modelo generativo de HuggingFace",
    )
    arg_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usar datos crudos en lugar de procesados",
    )
    args = arg_parser.parse_args()
    main(model_id=args.model_id, baseline=args.baseline)
