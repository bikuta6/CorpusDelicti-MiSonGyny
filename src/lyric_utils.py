import re
from sentence_transformers import SentenceTransformer, util
from pysentimiento.preprocessing import preprocess_tweet
import torch
import pandas as pd

def remove_redundant_lyrics(model: SentenceTransformer, text: str, threshold: float = 0.82) -> str:
    """
    Optimized for Spanglish lyrics and nuanced semantic redundancy using E5-Large.
    Threshold 0.82 is the sweet spot for E5 to prevent over-deduplication.
    """
    if not text.strip():
        return ""

    # 1. Structural cleaning
    lines = text.split("\n\n")
    # Only remove square brackets [Chorus], keep parentheses (ad-libs) for misogyny context
    lines = [re.sub(r'\[.*?\]', '', line).strip() for line in lines]
    lines = [line.replace('(', '').replace(')', '') for line in lines]
    # Drop empty lines and very short noise (single characters/grunts)
    lines = [line for line in lines if len(line) > 2]
    
    if not lines:
        return ""
    
    # 2. Embedding Generation (with E5 prefix)
    # E5 models perform best when instructions are prepended
    processed_lines = [f"passage: {preprocess_tweet(line)}" for line in lines]
    
    # 3. Generación de Embeddings
    # BAAI/bge-m3 no necesita prefijos para similitud de oraciones
    embeddings = model.encode(processed_lines, convert_to_tensor=True)
    
    kept_indices = [0] # Siempre mantenemos la primera línea

    for i in range(1, len(processed_lines)):
        # Comparación semántica contra las líneas que ya hemos guardado
        similarities = util.cos_sim(embeddings[i], embeddings[kept_indices])
        
        # Si la similitud es menor al umbral, aporta un significado nuevo
        if torch.max(similarities).item() < threshold:
            kept_indices.append(i)

    # 4. Devolver las líneas ORIGINALES (lines, no processed_lines)
    # Esto mantiene la jerga, faltas de ortografía o insultos intactos para el clasificador final
    return "\n".join([lines[idx] for idx in kept_indices])

def remove_redundant_lyrics_hierarchical(model: SentenceTransformer, text: str, 
                                         stanza_threshold: float = 0.88, 
                                         line_threshold: float = 0.92) -> str:
    """
    Deduplicación en dos fases para letras de canciones:
    1. Elimina estrofas/coros repetidos (threshold más permisivo).
    2. Elimina líneas repetidas dentro de las estrofas únicas (threshold más estricto).
    """
    if not text.strip():
        return ""

    # --- 0. LIMPIEZA ESTRUCTURAL GLOBAL ---
    # Quitar [Chorus], [Verse], etc.
    text_clean = re.sub(r'\[.*?\]', '', text)
    # Quitar los caracteres de paréntesis ( ) pero dejar su contenido
    text_clean = text_clean.replace('(', '').replace(')', '')
    
    # --- 1. FASE MACRO: DEDUPLICACIÓN POR ESTROFAS ---
    # Separamos por doble salto de línea
    stanzas = [s.strip() for s in text_clean.split("\n\n") if len(s.strip()) > 5]
    if not stanzas:
        return ""

    # Procesamos la estrofa entera (reemplazando saltos de línea por espacios para el embedding)
    stanzas = [s.replace("\n", " ") for s in stanzas]
    processed_stanzas = [f"passage: {preprocess_tweet(s)}" for s in stanzas]
    stanza_embeddings = model.encode(processed_stanzas, convert_to_tensor=True)
    
    kept_stanza_indices = [0]
    for i in range(1, len(stanzas)):
        similarities = util.cos_sim(stanza_embeddings[i], stanza_embeddings[kept_stanza_indices])
        if torch.max(similarities).item() < stanza_threshold:
            kept_stanza_indices.append(i)

    unique_stanzas = [stanzas[idx] for idx in kept_stanza_indices]

    # --- 2. FASE MICRO: DEDUPLICACIÓN INTRA-ESTROFA ---
    final_stanzas = []
    
    for stanza in unique_stanzas:
        # Separamos la estrofa por líneas individuales
        lines = [line.strip() for line in stanza.split("\n") if len(line.strip()) > 2]
        if not lines:
            continue
            
        # Si la estrofa tiene solo 1 línea, la guardamos directamente
        if len(lines) == 1:
            final_stanzas.append(lines[0])
            continue

        processed_lines = [f"passage: {preprocess_tweet(line)}" for line in lines]
        line_embeddings = model.encode(processed_lines, convert_to_tensor=True)
        
        kept_line_indices = [0]
        for i in range(1, len(lines)):
            similarities = util.cos_sim(line_embeddings[i], line_embeddings[kept_line_indices])
            # Usamos el umbral de línea (más estricto)
            if torch.max(similarities).item() < line_threshold:
                kept_line_indices.append(i)
                
        # Reconstruimos la estrofa con sus líneas únicas
        final_stanza_text = "\n".join([lines[idx] for idx in kept_line_indices])
        final_stanzas.append(final_stanza_text)

    # Reconstruimos la canción entera uniendo las estrofas limpias
    return "\n".join(final_stanzas)

if __name__ == "__main__":
    # Test with Spanglish and similar meanings (nuance check)
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    sample_lyrics = pd.read_csv("../data/task1/train.csv")["lyrics"]
    # Find the longest lyrics to test the function on a complex case
    longest_idx = sample_lyrics.apply(lambda x: len(x.split())).idxmax()
    sample_lyrics = sample_lyrics.iloc[longest_idx:longest_idx+1].tolist()
    print("--- Original Lyrics ---")
    print(sample_lyrics[0])
    print("Number of words:", len(sample_lyrics[0].split()))
    
    # 0.80 is usually the "sweet spot" for multilingual deduplication
    cleaned_text = [remove_redundant_lyrics(model, lyrics, threshold=0.90) for lyrics in sample_lyrics]
    print("--- Cleaned Lyrics ---")
    print(cleaned_text[0])
    print("Number of words after cleaning:", len(cleaned_text[0].split()))
    cleaned_text_hierarchical = [remove_redundant_lyrics_hierarchical(model, lyrics, stanza_threshold=0.88, line_threshold=0.92) for lyrics in sample_lyrics]
    print("--- Cleaned Lyrics (Hierarchical) ---")
    print(cleaned_text_hierarchical[0])
    print("Number of words after hierarchical cleaning:", len(cleaned_text_hierarchical[0].split()))