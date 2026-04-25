"""
Script de predicción para LLMs fine-tuneados con QLoRA (Task 3).
Uso equivalente al predict script de BERT.
Utiliza estrategia de chunking con solapamiento para textos largos.
"""

import argparse
import os
import sys

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

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


def format_inference_instruction(text, tokenizer):
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nLyric: {text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def predict_chunks(model, tokenizer, full_text, device, max_length=512, overlap=256):
    """Predict 'Y'/'N' for a long text by chunking it with overlapping windows."""
    chunks = []

    # Create overlapping chunks
    stride = max_length - overlap
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


def predict(model, tokenizer, texts, device, batch_size=8, max_length=512, overlap=256):
    """
    Batch inference with overlapping chunking strategy for long texts.
    Returns list of 'Y'/'N' strings.

    Args:
        model: The loaded model
        tokenizer: The model's tokenizer
        texts: List of texts to predict on
        device: Device to use (cuda, mps, cpu)
        batch_size: Not used with chunking strategy (kept for API compatibility)
        max_length: Maximum chunk length in characters
        overlap: Overlap between consecutive chunks in characters
    """
    preds = []

    for i, text in enumerate(texts):
        # Use chunk prediction strategy with overlapping windows for each text
        pred_label = predict_chunks(
            model, tokenizer, text, device, max_length=max_length, overlap=overlap
        )
        preds.append("Y" if pred_label == 1 else "N")

        if (i + 1) % 10 == 0:
            print(f"  Procesados {i + 1}/{len(texts)} ejemplos...")

    return preds


def main(args):
    print(f"Cargando datos de prueba desde {args.test_file}...")
    df = pd.read_csv(args.test_file)
    if "lyrics" not in df.columns:
        raise ValueError("El archivo CSV debe contener la columna 'lyrics'.")

    print(f"Cargando modelo desde {args.model_path}...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # AutoPeftModelForCausalLM handles LoRA adapter loading automatically
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = True

    print(
        f"Realizando predicciones con estrategia de chunking solapado "
        f"(chunk_size={args.max_length}, overlap={args.overlap})..."
    )
    preds = predict(
        model=model,
        tokenizer=tokenizer,
        texts=df["lyrics"].tolist(),
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        overlap=args.overlap,
    )

    df["preds"] = preds

    # Build output with id column — same logic as your BERT predict script
    if "song_id" in df.columns:
        df_out = df[["song_id", "preds"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "preds"]]
    else:
        df["id"] = [f"T3_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "preds"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df_out.to_csv(args.output_file, index=False)
    print(f"\nPredicciones guardadas en: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción con LLM QLoRA para Task 3")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo fine-tuneado (ej. ../../models/task3/llms/gemma-2b-it_final)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="CSV con columnas 'song_id' y 'lyrics'",
    )
    parser.add_argument(
        "--output_file", type=str, default="../../task_3_predictions.csv"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (no usado con estrategia de chunking, mantenido por compatibilidad)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Longitud máxima de cada chunk en caracteres",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Solapamiento entre chunks consecutivos en caracteres (default 50% de max_length)",
    )
    args = parser.parse_args()
    main(args)
