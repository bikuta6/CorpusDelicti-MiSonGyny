import pandas as pd
from BackTranslationEngine import BackTranslationEngine

if __name__ == "__main__":
    input_csv = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/train3.csv"
    output_csv = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/output_augmented.csv"

    print(f"Loading dataset from {input_csv} ...")
    df = pd.read_csv(input_csv)

    print("Initializing BackTranslationEngine ...")
    bte = BackTranslationEngine(
        output_file=output_csv,
        chunk_size_words=30,
        creative=True,
        chunk_min_words=12,
        chunk_max_words=40
    )


    print("Applying back translation augmentation ...")
    df_aug = bte.augmentar_dataset_backtranslation(
        df,
        columna_texto="lyrics",
        columna_id="song_id",
        save_every=50
    )

    print("Done.")