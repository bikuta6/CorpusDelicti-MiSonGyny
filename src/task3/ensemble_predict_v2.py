import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS

from bert_pooling import load_bert_like_classifier, predict_with_chunks


def get_model_probs(model_name, test_ds, device, augment=False):
    print(f"\n--- Generando predicciones para {model_name} ---")
    cfg = MODEL_CONFIGS[model_name]
    suffix = "_augmented" if augment else ""
    model_path = f"../../models/task3/final/{model_name}{suffix}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_bert_like_classifier(model_path, device)
    pred_output = predict_with_chunks(
        dataset=test_ds,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=cfg.max_len,
        batch_size=8,
        aggregation="max",
    )
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    del model, tokenizer
    torch.cuda.empty_cache()
    return probs


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


def get_llm_probs(model_path, texts, device, max_chars=1000, overlap=200):
    print(f"\n--- Generando predicciones para LLM ({model_path}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = True

    def predict_single(text):
        # Split into overlapping chunks if text is long
        stride = max_chars - overlap
        chunks = []
        for i in range(0, len(text), stride):
            chunk = text[i : i + max_chars]
            if chunk:
                chunks.append(chunk)
            if i + max_chars >= len(text):
                break

        chunk_preds = []
        for chunk in chunks:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nLyric: {chunk}"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

            generated = (
                tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                .strip()
                .upper()
            )

            chunk_preds.append(1.0 if "Y" in generated else 0.0)

        # Majority vote across chunks — return proportion as soft "prob"
        return max(chunk_preds)

    probs = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Procesados {i}/{len(texts)}...")
        probs.append(predict_single(text))

    del model, tokenizer
    torch.cuda.empty_cache()
    return np.array(probs)


def main(augment=False, voting_type="hard", use_llm=False, llm_path=None):
    test_file = "../../data/test.csv"
    output_file = "../../task_3_predictions.csv"

    print(f"Cargando datos de prueba desde {test_file}...")
    df = pd.read_csv(test_file)
    test_ds = Dataset.from_pandas(
        df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # BERT models — load one at a time to save VRAM
    probs_robertuito = get_model_probs("Robertuito", test_ds, device, augment=augment)
    probs_distilbeto = get_model_probs("XLM-R", test_ds, device, augment=augment)
    probs_beto = get_model_probs("LongFormer", test_ds, device, augment=augment)

    all_probs = [probs_robertuito, probs_distilbeto, probs_beto]
    # Weights based on individual dev F1 — update these with your actual scores
    weights = [0.7861405801947493, 0.7849637144195741, 0.7846249610955494]

    # Optionally add LLM
    if use_llm and llm_path:
        probs_llm = get_llm_probs(llm_path, df["lyrics"].tolist(), device)
        all_probs.append(probs_llm)
        weights.append(0.7)  # lower weight since LLM is weaker — adjust as needed
        print("LLM añadido al ensemble.")

    if voting_type == "soft":
        weights_arr = np.array(weights)
        final_probs = sum(p * w for p, w in zip(all_probs, weights_arr))
        final_probs /= weights_arr.sum()
        preds = (final_probs >= 0.5).astype(int)
        print(f"\nUsing Weighted Soft Voting")

    else:  # hard voting
        n_models = len(all_probs)
        votes = [(p >= 0.5).astype(int) for p in all_probs]
        sum_votes = sum(votes)
        # Majority: more than half must agree
        preds = (sum_votes > n_models / 2).astype(int)
        print(f"\nUsing Hard Voting ({n_models} models, threshold > {n_models / 2})")

    label_map = {0: "N", 1: "Y"}
    df["preds"] = [label_map[p] for p in preds]

    if "song_id" in df.columns:
        df_out = df[["song_id", "preds"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "preds"]]
    else:
        df["id"] = [f"T3_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "preds"]]

    df_out.to_csv(output_file, index=False)
    print(f"\n¡Predicciones guardadas en {output_file}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Prediction for Task 3")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--voting", type=str, choices=["soft", "hard"], default="hard")
    parser.add_argument("--use_llm", action="store_true", help="Add LLM to ensemble")
    parser.add_argument(
        "--llm_path",
        type=str,
        default=None,
        help="Path to fine-tuned LLM (e.g. ../../models/task3/llms/gemma-4-E2B-it_final)",
    )
    args = parser.parse_args()
    main(
        augment=args.augment,
        voting_type=args.voting,
        use_llm=args.use_llm,
        llm_path=args.llm_path,
    )
