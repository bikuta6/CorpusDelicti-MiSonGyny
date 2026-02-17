"""
Utilidades para reproducibilidad y configuración general del proyecto.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Establece seeds para reproducibilidad completa en:
    - Python random
    - NumPy
    - PyTorch (CPU y GPU)
    - CUDA (si está disponible)
    
    Args:
        seed (int): Seed para reproducibilidad (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para multi-GPU
    
    # Configuraciones adicionales para reproducibilidad total
    # Nota: Estos pueden reducir el rendimiento ligeramente
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Establecer seed para workers de DataLoader
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f" Seed establecido: {seed} (reproducibilidad activada)")


def set_seed_worker(worker_id):
    """
    Función para establecer seed en workers de DataLoader.
    Usar con: DataLoader(..., worker_init_fn=set_seed_worker)
    
    Args:
        worker_id: ID del worker (proporcionado automáticamente por DataLoader)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed=42):
    """
    Crea un generador de PyTorch con seed específico.
    Útil para DataLoader: DataLoader(..., generator=get_generator(42))
    
    Args:
        seed (int): Seed para el generador
    
    Returns:
        torch.Generator: Generador con seed establecido
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def print_seed_info():
    """
    Imprime información sobre el estado actual de seeds y reproducibilidad.
    """
    print("\n" + "="*60)
    print("CONFIGURACIÓN DE REPRODUCIBILIDAD")
    print("="*60)
    print(f"PyTorch deterministic:  {torch.backends.cudnn.deterministic}")
    print(f"PyTorch benchmark:      {torch.backends.cudnn.benchmark}")
    print(f"CUDA disponible:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:            {torch.cuda.get_device_name(0)}")
    print(f"PYTHONHASHSEED:         {os.environ.get('PYTHONHASHSEED', 'No establecido')}")
    print("="*60 + "\n")


# Configuración global por defecto
DEFAULT_SEED = 42
