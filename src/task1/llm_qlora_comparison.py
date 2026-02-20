"""
Comparativa de modelos para Task 1: Clasificación Binaria de Misoginia en Canciones usando LLMs con QLoRA.
Genera una tabla para el paper con métricas de cada modelo.
"""

import os
import sys
import gc
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Añadimos la carpeta padre al path para poder importar utils y trainer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED
from trainer import WeightedTrainer  # Importamos tu trainer personalizado

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
DATA_PATH = "../../data/task1/train.csv"
OUTPUT_DIR = "../../models/task1/comparativa_llm"
RESULTS_FILE = "../../results/task1/tabla_paper_llms.csv"
SAVE_DIR = "../../models/task1/comparison_llm"

# Los LLMs soportan contextos mucho más largos. 2048 suele cubrir canciones enteras (intro, estribillos, outro).
MAX_LEN = 2048 

# Modelos LLM a comparar
MODELS = {
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B",
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
}

# --- CARGA DE DATOS ---
print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds = Dataset.from_pandas(val_df, preserve_index=False)

# Calcular pesos para clase desbalanceada
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
ratio = n_neg / (n_pos + 1e-5)
weights_tensor = torch.tensor([1.0, ratio]).float()
print(f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 1: {ratio:.2f}")


def compute_metrics(pred):
    """Calcula métricas para evaluación"""
    labels = pred.label_ids
    # En modelos PEFT de clasificación, logits a veces viene encapsulado.
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0.0
        )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(acc, 4),
        "eval_f1_macro": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

results_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- INICIANDO COMPARATIVA LLM EN {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ---")

for name, model_id in MODELS.items():
    print(f"\n{'='*50}")
    print(f">>> Evaluando: {name}")
    print(f"{'='*50}")
    
    try:
        # 1. Configuración de Cuantización a 4-bit (Obligatorio para 12GB VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        # 2. Tokenizador
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Los LLMs no tienen pad_token por defecto. Asignamos el eos_token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 3. Función de tokenización (Sin smart truncate, pasamos el texto completo)
        def tokenize_fn(batch):
            return tokenizer(
                batch["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LEN
            )
        
        # 4. Tokenizar datasets
        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
        val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)
        train_tok = train_tok.rename_column("label", "labels")
        val_tok = val_tok.rename_column("labels" if "labels" in val_tok.column_names else "label", "labels")
        train_tok.set_format("torch")
        val_tok.set_format("torch")
        
        # 5. Crear modelo base cuantizado
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            quantization_config=bnb_config,
            device_map={"": 0}, # Forza a cargar todo en la GPU 0 para evitar conflictos con Trainer
            problem_type="single_label_classification"
        )
        # Asignar explícitamente el token de padding a la configuración del modelo
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # 6. Preparar para LoRA
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear", # Aplica LoRA a todas las capas lineales, crítico para modelos grandes
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS" # Crítico para Sequence Classification
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 7. Configurar entrenamiento (Ajuste extremo para 12GB VRAM)
        args = TrainingArguments(
            output_dir=f"{SAVE_DIR}/{name}/temp_checkpoints",
            learning_rate=2e-4, # Los modelos LoRA necesitan un LR más alto que los BERTs
            per_device_train_batch_size=1, # Obligatorio 1
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, # Compensamos el batch size de 1
            num_train_epochs=5, # 5 es suficiente con LLMs, el Early Stopping actuará rápido
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit", # Optimizador que ahorra muchísima memoria
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            gradient_checkpointing=True,
        )
        
        # 8. Instanciar tu Trainer personalizado
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            compute_metrics=compute_metrics,
            class_weights=weights_tensor,  # Usa tus propios pesos
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # 9. Entrenar y evaluar
        trainer.train()
        metrics = trainer.evaluate()
        
        results_list.append({
            "Modelo": name,
            "F1-Macro": metrics["eval_f1_macro"],
            "Accuracy": metrics["eval_accuracy"],
            "Precision": metrics["eval_precision"],
            "Recall": metrics["eval_recall"],
        })
        
        print(f" {name}: F1={metrics['eval_f1_macro']:.4f}")
        
    except Exception as e:
        print(f" Error con {name}: {e}")
        results_list.append({
            "Modelo": name,
            "F1-Macro": None,
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
        })
        
    finally:
        # Limpieza de Memoria Rigurosa (Vital para poder entrenar el segundo modelo)
        if 'trainer' in locals():
            del trainer
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

# --- GUARDAR RESULTADOS ---
df_res = pd.DataFrame(results_list)
df_res = df_res.sort_values("F1-Macro", ascending=False)

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df_res.to_csv(RESULTS_FILE, index=False)

print(f"\n{'='*60}")
print("RESULTADOS COMPARATIVA LLMs QLoRA")
print(f"{'='*60}")
print(df_res.to_markdown(index=False))
print(f"\n Guardado en: {RESULTS_FILE}")