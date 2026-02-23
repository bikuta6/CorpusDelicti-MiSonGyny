import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, DEFAULT_SEED

# --- REPRODUCIBILIDAD ---
set_seed(DEFAULT_SEED)

# --- MODELOS A PROBAR ---
MODELS_TO_CHECK = {
    "DistilBETO": "dccuchile/distilbert-base-spanish-uncased",  # Español eficiente
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",  # Español clásico
    "MarIA": "IsGarrido/roberta-base-bne",  # SOTA en español
    "XLM-R": "xlm-roberta-base",  # Clásico multilingual
    "mDeBERTa": "microsoft/mdeberta-v3-base",  # SOTA multilingual
    "Robertuito": "pysentimiento/robertuito-base-uncased",  # Especializado en slang en español
    "LongFormer": "markussagen/xlm-roberta-longformer-base-4096",  # Para textos largos (canciones)
    # "Qwen-2.5-7B": "Qwen/Qwen2.5-7B",
    # "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"--- INICIANDO DIAGNÓSTICO EN: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'} ---\n")

def test_model(name, model_id):
    print(f" Probando: {name} ({model_id})...")
    
    try:
        # 1. CARGA DE TOKENIZER (Con fallback a versión lenta si falla)
        print("   [1/4] Cargando Tokenizer...", end=" ")
        if "roberta-base-bne" in model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(" OK")

        # 2. CARGA DEL MODELO
        print("   [2/4] Cargando Modelo...", end=" ")
        dummy_weights = torch.tensor([1.0, 5.0]).to(DEVICE)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
        ).to(DEVICE)
        model.train()
        print(" OK")

        # 3. DATOS DUMMY
        texts = ["Esto es una prueba", "Otra prueba de texto"]
        labels = torch.tensor([0, 1]).to(DEVICE)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        inputs["labels"] = labels

        # 4. FORWARD PASS (Con Autocast para arreglar mDeBERTa)
        print("   [3/4] Forward Pass (Autocast)...", end=" ")
        
        # EL SECRETO: Usar autocast para gestionar FP16/FP32 automáticamente
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            print(f"Logits: {logits.detach().cpu().numpy()}")
        
        if loss is None: raise ValueError("Loss is None")
        print(f" OK (Loss: {loss.item():.4f})")

        # 5. BACKWARD PASS
        print("   [4/4] Verificando Gradientes...", end=" ")
        
        # Backward necesita escalar el loss si usamos float16, pero para este test simple 
        # basta con llamarlo directo, ya que no estamos optimizando pesos reales.
        loss.backward()
        

        # Determine the correct classifier weight for gradient checking
        if hasattr(model.classifier, "out_proj"):  # RoBERTa / XLM-R / Robertuito
            param_check = model.classifier.out_proj.weight
        elif isinstance(model.classifier, torch.nn.Linear):  # DistilBERT / BERT / DeBERTa
            param_check = model.classifier.weight
        else:
            raise ValueError(f"Unknown classifier type: {type(model.classifier)}")

        # Check gradients
        if param_check.grad is None:
            raise ValueError("Gradientes None")
        grad_step_1 = param_check.grad.abs().sum().item()
        if grad_step_1 == 0:
            raise ValueError("Gradientes CERO")

        # Simular Acumulación (Autocast)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            outputs_2 = model(**inputs)
            loss_2 = outputs_2.loss
        loss_2.backward()

        # Check accumulation
        grad_step_2 = param_check.grad.abs().sum().item()
        if grad_step_2 <= grad_step_1:
            raise ValueError(f"No acumula (G1={grad_step_1} -> G2={grad_step_2})")

        print(f" OK (Acumula: {grad_step_1:.2f} -> {grad_step_2:.2f})")
        
        # Limpieza
        del model, tokenizer, inputs, labels, outputs
        torch.cuda.empty_cache()
        print(f" {name}: LISTO\n")
        return True

    except Exception as e:
        print(f"\n ERROR EN {name}: {str(e)}\n")
        return False

# --- EJECUCIÓN ---
passed = 0
total = len(MODELS_TO_CHECK)
for name, mid in MODELS_TO_CHECK.items():
    if test_model(name, mid): passed += 1

print("="*40)
print(f"RESUMEN: {passed}/{total} Modelos listos.")
print("="*40)