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
from trainer import WeightedTrainer
from llm_model_configs import LLMConfig, LLM_CONFIGS

# --- REPRODUCIBILIDAD ---
SEED = DEFAULT_SEED
set_seed(SEED)

# --- CONFIGURACIÓN ---
DATA_PATH = "../../data/task1/processed_train.csv"
OUTPUT_DIR = "../../models/task1/comparativa_llm"
RESULTS_FILE = "../../results/task1/tabla_paper_llms.csv"
SAVE_DIR = "../../models/task1/comparison_llm"



# --- CARGA DE DATOS ---
print(f"Cargando datos desde {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].map({"NM": 0, "M": 1})

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
# column lyrics -> text, label -> label
train_ds = Dataset.from_pandas(train_df.rename(columns={"lyrics": "text", "label": "label"}), preserve_index=False)
val_ds = Dataset.from_pandas(val_df.rename(columns={"lyrics": "text", "label": "label"}), preserve_index=False)

# Calcular pesos para clase desbalanceada
n_pos = sum(df["label"] == 1)
n_neg = sum(df["label"] == 0)
total = n_neg + n_pos
w0 = total / (2 * n_neg)
w1 = total / (2 * n_pos)
weights_tensor = torch.tensor([w0, w1]).float()
print(f"Desbalance: Neg={n_neg}, Pos={n_pos} -> Peso clase 0: {w0:.2f}, clase 1: {w1:.2f}")


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

def make_tokenize_fn(tokenizer, max_len: int):
    """
    Factoría de funciones de tokenización.
    Evita el bug de closure en bucles Python.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
    return tokenize_fn

results_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"--- INICIANDO COMPARATIVA LLM EN {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ---")

for name, cfg in LLM_CONFIGS.items():
    if name[-2:] in  {"7B", "8B"}:
        continue
    model_id = cfg.model_id
    print(f"\n{'='*50}")
    print(f">>> Evaluando: {name} ({model_id})")
    print(f"    lr={cfg.learning_rate}, max_len={cfg.max_len}, lora_r={cfg.lora_r}")
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
        tokenize_fn = make_tokenize_fn(tokenizer, cfg.max_len)
        
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
            problem_type="single_label_classification",
            attn_implementation="sdpa"
        )
        # Asignar explícitamente el token de padding a la configuración del modelo
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # 6. Preparar para LoRA
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 7. Configurar entrenamiento
        checkpoints_path = os.path.join(SAVE_DIR, name, "checkpoints")
        args = TrainingArguments(
            output_dir=checkpoints_path,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim=cfg.optim,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler_type,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        
        # 8. Instanciar tu Trainer personalizado
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            compute_metrics=compute_metrics,
            loss_type=cfg.loss_type,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
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
        
        print(f"✓ {name}: F1={metrics['eval_f1_macro']:.4f}")

        # Guardar adaptadores LoRA + tokenizador
        model_save_path = os.path.join(SAVE_DIR, name)
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)   # Solo guarda los pesos LoRA
        tokenizer.save_pretrained(model_save_path)
        print(f"Modelo guardado en: {model_save_path}")
        
    except Exception as e:
        print(f"✗ Error con {name}: {e}")
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