from sentence_transformers import SentenceTransformer, util
import torch

def remove_redundant_lyrics(text: str, threshold: float = 0.85, model_name: str = 'paraphrase-multilingual-mpnet-base-v2') -> str:
    """
    Optimized for Spanglish lyrics and nuanced semantic redundancy.
    """
    if not text.strip():
        return ""

    # In a production pipeline, load this model globally/once outside the function
    model = SentenceTransformer(model_name)
    
    # Clean and split lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return ""

    # Generate embeddings
    embeddings = model.encode(lines, convert_to_tensor=True)
    
    kept_indices = [0] # Always keep the first line

    for i in range(1, len(lines)):
        # We compare the current line against all lines already kept using a vector dot product
        # util.cos_sim can compare one vector against a list of vectors simultaneously
        similarities = util.cos_sim(embeddings[i], embeddings[kept_indices])
        
        # If the highest similarity found is below the threshold, it's a "new" unique line
        if torch.max(similarities).item() < threshold:
            kept_indices.append(i)

    return "\n".join([lines[idx] for idx in kept_indices])

if __name__ == "__main__":
    # Test with Spanglish and similar meanings (nuance check)
    sample_lyrics = """
    Ella no quiere a nadie, she just wants to dance.
    Solo quiere bailar y no le importa nada.
    She just wants to dance.                  
    Today is a beautiful day.
    Hoy es un día hermoso.                 
    """
    print("--- Original Lyrics ---")
    print(sample_lyrics)
    # 0.80 is usually the "sweet spot" for multilingual deduplication
    cleaned_text = remove_redundant_lyrics(sample_lyrics, threshold=0.80)
    print("--- Cleaned Lyrics ---")
    print(cleaned_text)