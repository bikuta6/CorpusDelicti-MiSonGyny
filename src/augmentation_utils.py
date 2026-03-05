import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import random
import nltk
from tqdm.auto import tqdm

# Asegurar que los recursos necesarios de NLTK estén presentes
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class LyricsAugmentor:
    def __init__(self):
        # 1. Sustitución por Sinónimos (Español)
        self.aug_syn = naw.SynonymAug(aug_src='wordnet', lang='spa')
        
        # 2. Random Swap (Intercambia palabras)
        self.aug_swap = naw.RandomWordAug(action="swap", aug_p=0.1)
        
        # 3. Ruido de caracteres (Simula typos: "perra" -> "perrs" o "pe rra")
        # Usamos RandomCharAug que no depende del idioma del teclado
        self.aug_char = nac.RandomCharAug(action="substitute", aug_char_p=0.1, aug_word_p=0.1)

    def augment_dataframe(self, df, text_col="lyrics", label_col="label", minority_label=1, multiplier=1):
        """
        Aumenta SOLO la clase minoritaria del dataframe entregado.
        """
        print(f"--- Iniciando Aumentación (Factor x{multiplier}) ---")
        
        # Filtrar clase minoritaria
        minority_df = df[df[label_col] == minority_label].copy()
        new_rows = []

        for _, row in tqdm(minority_df.iterrows(), total=len(minority_df), desc="Augmenting M"):
            for _ in range(multiplier):
                text = str(row[text_col])
                if not text or text.lower() == "nan": continue
                
                choice = random.random()
                
                try:
                    if choice < 0.4:
                        # Sinónimos (40%)
                        augmented_text = self.aug_syn.augment(text)[0]
                        method = "synonym"
                    elif choice < 0.7:
                        # Swap de palabras (30%)
                        augmented_text = self.aug_swap.augment(text)[0]
                        method = "word_swap"
                    else:
                        # Typos/Ruido de caracteres (30%)
                        augmented_text = self.aug_char.augment(text)[0]
                        method = "char_noise"
                except Exception as e:
                    # Si falla (por texto muy corto, etc), mantenemos original
                    augmented_text = text
                    method = "original_fallback"

                new_row = row.copy()
                new_row[text_col] = augmented_text
                new_row["augmentation"] = method
                new_rows.append(new_row)

        df_augmented = pd.DataFrame(new_rows)
        
        if "augmentation" not in df.columns:
            df["augmentation"] = "original"
            
        combined_df = pd.concat([df, df_augmented], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Aumentación completada: {len(df)} -> {len(combined_df)} muestras.")
        return combined_df