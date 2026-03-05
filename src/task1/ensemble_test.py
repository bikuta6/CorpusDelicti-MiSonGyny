import os
import sys
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from bert_model_configs import MODEL_CONFIGS

# Reutilizamos la función make_tokenize_fn para consistencia total
from train_kfold import make_tokenize_fn

# Configuración del Ensemble
MODELS_TO_ENSEMBLE = ["BETO", "MarIA", "Robertuito"] # Nombres en tu MODEL_CONFIGS
MODELS_BASE_DIR = "../../models/task1/comparison"
TEST_PATH = "../../data/task1/processed_test.csv"
RESULTS_FILE = "../../results/task1/submission_ensemble.csv"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_probabilities(name, df_test):
    cfg = MODEL_CONFIGS[name]
    model_path = os.path.join(MODELS_BASE_DIR, name)
    
    print(f"\n>> Inferencia con: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Preparar Dataset con Truncamiento Inteligente
    ds = Dataset.from_pandas(df_test.rename(columns={"lyrics": "text"}), preserve_index=False)
    tok_fn = make_tokenize_fn(tokenizer, cfg)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds_tok.set_format("torch")
    
    loader = DataLoader(ds_tok, batch_size=BATCH_SIZE)
    probs = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            # Manejar token_type_ids si existen
            inputs = {"input_ids": input_ids, "attention_mask": mask}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)
            
            outputs = model(**inputs)
            p = torch.softmax(outputs.logits, dim=-1)[:, 1] # Probabilidad clase Misoginia
            probs.extend(p.cpu().numpy())
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return np.array(probs)

def main():
    df_test = pd.read_csv(TEST_PATH)
    all_probs = []

    for name in MODELS_TO_ENSEMBLE:
        p = get_model_probabilities(name, df_test)
        all_probs.append(p)

    # Promedio de probabilidades (Soft Voting)
    avg_probs = np.mean(all_probs, axis=0)
    
    # Threshold: Puedes usar el promedio de los thresholds de tabla_paper.csv
    # o un valor estándar de 0.5 ajustado según validación.
    final_threshold = 0.5 
    
    predictions = ["M" if p >= final_threshold else "NM" for p in avg_probs]
    
    df_test["label"] = predictions
    df_test[["id", "label"]].to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Submission guardada en: {RESULTS_FILE}")

if __name__ == "__main__":
    main()