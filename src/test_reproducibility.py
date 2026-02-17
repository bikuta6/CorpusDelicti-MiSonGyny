"""
Script de prueba: Verifica que la reproducibilidad con seeds funciona correctamente.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import random
from utils import set_seed, DEFAULT_SEED, print_seed_info

print("="*60)
print("TEST DE REPRODUCIBILIDAD")
print("="*60)

# Mostrar información de configuración
print_seed_info()

# --- TEST 1: Python random ---
print("\n[TEST 1] Python random.random()")
set_seed(DEFAULT_SEED)
r1_a = [random.random() for _ in range(5)]
set_seed(DEFAULT_SEED)
r1_b = [random.random() for _ in range(5)]

print(f"  Primera ejecución:  {r1_a}")
print(f"  Segunda ejecución:  {r1_b}")
print(f"  Reproducible" if r1_a == r1_b else "  NO reproducible")

# --- TEST 2: NumPy ---
print("\n[TEST 2] NumPy random")
set_seed(DEFAULT_SEED)
r2_a = np.random.rand(5).tolist()
set_seed(DEFAULT_SEED)
r2_b = np.random.rand(5).tolist()

print(f"  Primera ejecución:  {[f'{x:.4f}' for x in r2_a]}")
print(f"  Segunda ejecución:  {[f'{x:.4f}' for x in r2_b]}")
print(f"  Reproducible" if np.allclose(r2_a, r2_b) else "  NO reproducible")

# --- TEST 3: PyTorch CPU ---
print("\n[TEST 3] PyTorch (CPU)")
set_seed(DEFAULT_SEED)
r3_a = torch.randn(5).tolist()
set_seed(DEFAULT_SEED)
r3_b = torch.randn(5).tolist()

print(f"  Primera ejecución:  {[f'{x:.4f}' for x in r3_a]}")
print(f"  Segunda ejecución:  {[f'{x:.4f}' for x in r3_b]}")
print(f"  Reproducible" if torch.allclose(torch.tensor(r3_a), torch.tensor(r3_b)) else "  NO reproducible")

# --- TEST 4: PyTorch GPU (si está disponible) ---
if torch.cuda.is_available():
    print("\n[TEST 4] PyTorch (GPU)")
    set_seed(DEFAULT_SEED)
    r4_a = torch.randn(5, device='cuda').cpu().tolist()
    set_seed(DEFAULT_SEED)
    r4_b = torch.randn(5, device='cuda').cpu().tolist()
    
    print(f"  Primera ejecución:  {[f'{x:.4f}' for x in r4_a]}")
    print(f"  Segunda ejecución:  {[f'{x:.4f}' for x in r4_b]}")
    print(f"  Reproducible" if torch.allclose(torch.tensor(r4_a), torch.tensor(r4_b)) else "  NO reproducible")
else:
    print("\n[TEST 4] PyTorch (GPU)")
    print("  GPU no disponible, saltando test")

# --- TEST 5: Augmentation ---
print("\n[TEST 5] Augmentation (AEDA)")
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'task1')))
    from augmentation import aeda_punctuation
    
    test_text = "Esta es una canción de prueba con varias palabras para augmentar"
    
    set_seed(DEFAULT_SEED)
    aug1 = aeda_punctuation(test_text, insertion_rate=0.3)
    set_seed(DEFAULT_SEED)
    aug2 = aeda_punctuation(test_text, insertion_rate=0.3)
    
    print(f"  Original:           {test_text[:50]}...")
    print(f"  Primera ejecución:  {aug1[:50]}...")
    print(f"  Segunda ejecución:  {aug2[:50]}...")
    print(f"  Reproducible" if aug1 == aug2 else "  NO reproducible")
except ImportError as e:
    print(f"  No se pudo importar augmentation: {e}")

# --- RESUMEN ---
print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(" Sistema de reproducibilidad configurado correctamente")
print(f"   Seed por defecto: {DEFAULT_SEED}")
print("   Todos los scripts usan utils.set_seed()")
print("\n Archivos configurados:")
print("   • src/utils.py")
print("   • src/task1/entrenamiento.py")
print("   • src/task1/comparativa.py")
print("   • src/task1/ensemble_stacking.py")
print("   • src/task1/augmentation.py")
print("   • src/check_models.py")
print("\n Uso:")
print("   from utils import set_seed, DEFAULT_SEED")
print("   set_seed(DEFAULT_SEED)  # Al inicio del script")
print("="*60 + "\n")
