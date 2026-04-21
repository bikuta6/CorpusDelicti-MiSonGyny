import os
import pandas as pd
from LLMEngine_Qw import LLMEngine

if __name__ == "__main__":
    input_csv = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/train3.csv"
    output_csv = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/output_augmented_qwen4bit.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    llm = LLMEngine(
        modelo="Qwen/Qwen2.5-3B-Instruct",
        creative=True,
        load_in_4bit=True
    )

    df_aug = llm.augmentar_dataset(
        df,
        columna_texto="lyrics",
        columna_id="song_id",
        n=1,
        max_words=18,
        max_new_tokens=40
    )

    df_aug.to_csv(output_csv, index=False)
    print(f"Guardado en: {output_csv}")