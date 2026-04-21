import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIGURACIÓN
# =========================
ORIGINAL_CSV_PATH = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/processed_train.csv"

GENERATED_CSV_PATH = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/output_augmented_generated.csv"

OUTPUT_CSV_PATH = "/home/emanuel/Documents/Master/NaturalLanguageP/Proyecto/CorpusDelicti-MiSonGyny/src/Augmentation_tools/similarity_result.csv"

LYRICS_COLUMN = "lyrics"
SIMILARITY_THRESHOLD = 90.0  # < 90 = aceptable, >= 90 = roto


def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bow_cosine_similarity(text1: str, text2: str) -> float:
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors.getrow(0), vectors.getrow(1))[0, 0]
    return float(similarity)


def compare_csvs() -> None:
    original_df = pd.read_csv(ORIGINAL_CSV_PATH)
    generated_df = pd.read_csv(GENERATED_CSV_PATH)

    if LYRICS_COLUMN not in original_df.columns:
        raise ValueError(f"El CSV original no contiene la columna '{LYRICS_COLUMN}'.")
    if LYRICS_COLUMN not in generated_df.columns:
        raise ValueError(f"El CSV generado no contiene la columna '{LYRICS_COLUMN}'.")

    if len(original_df) != len(generated_df):
        raise ValueError(
            f"Los CSV no tienen la misma cantidad de filas: "
            f"original={len(original_df)}, generado={len(generated_df)}"
        )

    similarities = []

    for original_text, generated_text in zip(original_df[LYRICS_COLUMN], generated_df[LYRICS_COLUMN]):
        original_clean = preprocess_text(original_text)
        generated_clean = preprocess_text(generated_text)

        sim = bow_cosine_similarity(original_clean, generated_clean) * 100
        similarities.append(round(sim, 2))

    result_df = generated_df.copy()
    result_df["similaridad_porcentaje"] = similarities
    result_df["estado"] = [
        "aceptable" if sim < SIMILARITY_THRESHOLD else "roto"
        for sim in similarities
    ]

    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"CSV resultado guardado en: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    compare_csvs()