"""
Augmentation específico para letras de canciones.

Técnicas diseñadas para preservar el contenido semántico mientras se generan
variaciones que ayudan al modelo a generalizar mejor.

Uso:
    from augmentation import augment_song, create_augmented_dataset
"""

import random
import re
import pandas as pd
from collections import Counter


def detect_chorus_lines(text):
    """
    Detecta posibles líneas de estribillo buscando repeticiones.
    En canciones, el estribillo suele contener el mensaje central.
    
    Returns:
        set: Líneas que aparecen 2+ veces (probablemente estribillo)
    """
    lines = [l.strip().lower() for l in text.split('\n') if l.strip()]
    line_counts = Counter(lines)
    return {line for line, count in line_counts.items() if count >= 2}


def shuffle_stanzas(text):
    """
    Mezcla estrofas manteniendo primera y última (intro/outro).
    El orden de estrofas intermedias no suele afectar el significado global.
    """
    lines = text.split('\n')
    stanzas = []
    current = []
    
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            stanzas.append(current)
            current = []
    if current:
        stanzas.append(current)
    
    if len(stanzas) <= 2:
        return text
    
    # Mantener primera y última, mezclar el resto
    middle = stanzas[1:-1]
    random.shuffle(middle)
    stanzas = [stanzas[0]] + middle + [stanzas[-1]]
    
    return '\n\n'.join(['\n'.join(s) for s in stanzas])


def line_dropout(text, drop_ratio=0.15):
    """
    Elimina un porcentaje de líneas aleatorias.
    Simula letras incompletas o mal transcritas.
    """
    lines = text.split('\n')
    if len(lines) <= 5:
        return text
    
    n_drop = max(1, int(len(lines) * drop_ratio))
    indices_to_drop = set(random.sample(range(len(lines)), n_drop))
    
    return '\n'.join([l for i, l in enumerate(lines) if i not in indices_to_drop])


def word_dropout(text, drop_ratio=0.1):
    """
    Elimina palabras aleatorias (no líneas completas).
    Más sutil que line_dropout.
    """
    words = text.split()
    if len(words) <= 10:
        return text
    
    n_drop = max(1, int(len(words) * drop_ratio))
    indices_to_drop = set(random.sample(range(len(words)), n_drop))
    
    return ' '.join([w for i, w in enumerate(words) if i not in indices_to_drop])


def normalize_whitespace(text):
    """
    Normaliza espacios y saltos de línea.
    Variación muy suave.
    """
    # Normalizar múltiples espacios
    text = re.sub(r' +', ' ', text)
    # Normalizar múltiples saltos de línea
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def augment_song(text, augmentation_type=None, p=0.5):
    """
    Aplica una augmentación aleatoria a la letra de una canción.
    
    Args:
        text: Letra de la canción
        augmentation_type: Tipo específico de augmentación (None = aleatorio)
            - 'shuffle': Mezclar estrofas
            - 'line_drop': Eliminar líneas
            - 'word_drop': Eliminar palabras
            - 'normalize': Normalizar espacios
        p: Probabilidad de aplicar augmentación (default 0.5)
    
    Returns:
        str: Texto augmentado (o original si random > p)
    """
    if random.random() > p:
        return text
    
    augmentations = {
        'shuffle': shuffle_stanzas,
        'line_drop': line_dropout,
        'word_drop': word_dropout,
        'normalize': normalize_whitespace,
    }
    
    if augmentation_type is None:
        augmentation_type = random.choice(list(augmentations.keys()))
    
    return augmentations.get(augmentation_type, lambda x: x)(text)


def create_augmented_dataset(df, augment_positive_only=True, n_augments=2, p=1.0):
    """
    Crea dataset aumentado, enfocándose en la clase minoritaria.
    
    Args:
        df: DataFrame con columnas 'text' y 'label'
        augment_positive_only: Si True, solo aumenta clase positiva (misógino)
        n_augments: Número de augmentaciones por muestra
        p: Probabilidad de augmentación (1.0 = siempre)
    
    Returns:
        pd.DataFrame: Dataset aumentado
    """
    augmented_rows = []
    
    for _, row in df.iterrows():
        # Siempre incluir original
        augmented_rows.append(row.to_dict())
        
        # Augmentar solo positivos (misóginos) si se especifica
        if augment_positive_only and row['label'] == 0:
            continue
        
        # Generar n_augments variaciones
        aug_types = ['shuffle', 'line_drop', 'word_drop', 'normalize']
        for i in range(n_augments):
            new_row = row.to_dict()
            aug_type = aug_types[i % len(aug_types)]
            new_row['text'] = augment_song(row['text'], augmentation_type=aug_type, p=p)
            
            # Solo añadir si es diferente al original
            if new_row['text'] != row['text']:
                augmented_rows.append(new_row)
    
    return pd.DataFrame(augmented_rows)


# --- Para testing ---
if __name__ == "__main__":
    sample = """Verso uno de la canción
aquí empieza la historia
con palabras que riman

Este es el estribillo
que se repite mucho
Este es el estribillo

Otro verso diferente
con más contenido aquí
siguiendo la melodía

Este es el estribillo
que se repite mucho
Este es el estribillo"""

    print("=== ORIGINAL ===")
    print(sample)
    print("\n=== SHUFFLE ===")
    print(augment_song(sample, 'shuffle', p=1.0))
    print("\n=== LINE DROP ===")
    print(augment_song(sample, 'line_drop', p=1.0))
    print("\n=== WORD DROP ===")
    print(augment_song(sample, 'word_drop', p=1.0))
