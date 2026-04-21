import os
import pandas as pd
from LLMEngine import LLMEngine

if __name__ == "__main__":
    input_csv = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/train3.csv"
    output_csv = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/output_augmented_L2.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"Loading dataset from {input_csv} ...")
    df = pd.read_csv(input_csv)

    if "lyrics" not in df.columns or "song_id" not in df.columns:
        raise ValueError("CSV must contain 'lyrics' and 'song_id' columns")

    print(f"Rows loaded: {len(df)}")

    print("Initializing LLMEngine ...")
    llm = LLMEngine(
        modelo="google/flan-t5-small",
        creative=True
    )

    print("Applying LLM augmentation ...")
    df_aug = llm.augmentar_dataset(
    df,
    columna_texto="lyrics",
    columna_id="song_id",
    n=1,
    max_words=12,
    max_new_tokens=32
)

    print(f"Generated {len(df_aug)} synthetic samples")

    if len(df_aug) == 0:
        print("Warning: no synthetic samples were generated.")
    else:
        print(f"Saving augmented dataset to {output_csv} ...")
        df_aug.to_csv(output_csv, index=False)
        print("Done.")