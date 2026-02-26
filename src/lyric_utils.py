import re
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

def remove_redundant_lyrics(model: SentenceTransformer, text: str, threshold: float = 0.85) -> str:
    """
    Optimized for Spanglish lyrics and nuanced semantic redundancy.
    """
    if not text.strip():
        return ""

    # In a production pipeline, load this model globally/once outside the function
    
    # Clean and split lines
    lines = text.split("\n")
    # Remove text between brackets (e.g., [Chorus], (Verse 1))
    lines = [re.sub(r'[\[\(].*?[\]\)]', '', line).strip() for line in lines]
    lines = [line for line in lines if line != ""]  # Remove empty lines
    lines = [line for line in lines if len(line.split()) > 2]  # Keep lines with more than 2 words
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

    return ". ".join([lines[idx] for idx in kept_indices])

if __name__ == "__main__":
    # Test with Spanglish and similar meanings (nuance check)
    sample_lyrics = pd.read_csv("../data/task1/train.csv")["lyrics"]
    # Find the longest lyrics to test the function on a complex case
    longest_idx = sample_lyrics.apply(lambda x: len(x.split())).idxmax()
    sample_lyrics = sample_lyrics.iloc[412:413].tolist()  # Get the longest lyrics as a list
    print("--- Original Lyrics ---")
    print(sample_lyrics[0])
    print("Number of words:", len(sample_lyrics[0].split()))
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    # 0.80 is usually the "sweet spot" for multilingual deduplication
    cleaned_text = [remove_redundant_lyrics(model, lyrics, threshold=0.80) for lyrics in sample_lyrics]
    print("--- Cleaned Lyrics ---")
    print(cleaned_text[0])
    print("Number of words after cleaning:", len(cleaned_text[0].split()))