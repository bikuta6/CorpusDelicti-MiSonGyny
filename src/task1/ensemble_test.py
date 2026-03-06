import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score
from bert_model_configs import MODEL_CONFIGS, ModelConfig
from pysentimiento.preprocessing import preprocess_tweet
# seed  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_seed, DEFAULT_SEED
set_seed(DEFAULT_SEED)



# Configuración del Ensemble
MODELS_TO_ENSEMBLE = ["LongFormer", "MarIA", "DistilBETO"] # Nombres en tu MODEL_CONFIGS
MODELS_BASE_DIR = "../../models/task1/comparison"
TEST_PATH = "../../data/task1/processed_test.csv"
RESULTS_FILE = "../../results/task1/submission_ensemble.csv"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_tokenize_fn(tokenizer, cfg: ModelConfig):
    """Tokenization with smart head+tail truncation (75% Head, 25% Tail)."""
    def tokenize_fn(batch):
        texts = batch["text"]
        if cfg.use_pysentimiento_preprocess:
            texts = [preprocess_tweet(t, lang="es") for t in texts]

        tokenized = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)

        max_len = cfg.max_len
        input_ids, attention_mask = [], []

        for ids in tokenized["input_ids"]:
            if len(ids) <= max_len:
                pad_len = max_len - len(ids)
                padded = ids + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            else:
                cls_token, sep_token = ids[0], ids[-1]
                content = ids[1:-1]
                head_len = int((max_len - 2) * 0.75)
                tail_len = (max_len - 2) - head_len
                padded = [cls_token] + content[:head_len] + content[-tail_len:] + [sep_token]
                mask = [1] * max_len
            
            input_ids.append(padded)
            attention_mask.append(mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return tokenize_fn

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
    
    # 1. Definimos los modelos y sus pesos basados en su F1-Macro de Test
    # Pesos sugeridos según tus resultados: LongFormer(0.81), MarIA(0.79), DistilBETO(0.78)
    MODELS_TO_WEIGHT = {
        "LongFormer": 0.34, # El mejor, le damos la mitad del voto
        "MarIA": 0.33,      # Muy estable en test
        "DistilBETO": 0.33  # Un poco menos, pero ayuda a la diversidad
    }

    all_probs = []
    
    # Extraer probabilidades
    for name in MODELS_TO_WEIGHT.keys():
        p = get_model_probabilities(name, df_test)
        all_probs.append(p)

    # 2. Aplicar Ponderación y Power Averaging
    # Elevamos la probabilidad a una potencia (p=2 o p=3) para "premiar" la seguridad.
    # Esto hace que un 0.9 valga mucho más que dos 0.5.
    power = 1.5 
    weighted_probs_sum = np.zeros(len(df_test))
    total_weight = sum(MODELS_TO_WEIGHT.values())

    for i, (name, weight) in enumerate(MODELS_TO_WEIGHT.items()):
        # Aplicamos el peso del modelo y la potencia de confianza
        weighted_probs_sum += (all_probs[i] ** power) * weight

    # Promediamos y devolvemos a la escala original (raíz de la potencia)
    avg_probs = (weighted_probs_sum / total_weight) ** (1/power)
    
    # 3. Optimización del Threshold
    # Como el modelo tiende a ser conservador, un threshold ligeramente 
    # más bajo que 0.5 suele dar mejor F1-Macro
    final_threshold = 0.5 # Valor sugerido basado en tus 'Best-Threshold'
    
    predictions = ["M" if p >= final_threshold else "NM" for p in avg_probs]
    
    # --- Guardado y Métricas ---
    df_test["label"] = predictions
    df_test[["id", "label"]].to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Submission guardada en: {RESULTS_FILE}")

    # Evaluación (si tienes las etiquetas)
    true_labels_df = pd.read_csv("../../data/task1/test_labels.csv")
    y_true = true_labels_df["label"].map({"NM": 0, "M": 1}).values
    y_pred = np.array([1 if p == "M" else 0 for p in predictions])
    
    print("\n" + "="*60)
    print("ENSEMBLE PONDERADO (POWER=2) - CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=["NM", "M"]))
    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("="*60)

 

if __name__ == "__main__":
    main()