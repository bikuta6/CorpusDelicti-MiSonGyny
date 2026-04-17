from __future__ import annotations

import sys
from typing import Dict, List

from transformers import AutoTokenizer

sys.path.append("src")
from random_crop_collator import RandomCropDataCollator

MODELS: Dict[str, str] = {
    "BETO": "dccuchile/bert-base-spanish-wwm-uncased",
    "XLM-R": "xlm-roberta-base",
    "MarIA": "IsGarrido/roberta-base-bne",
}

TEXTS: List[str] = [
    "hola mundo",  # Corta (debe rellenar con [PAD])
    "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince",  # Larga (crop aleatorio)
]

MAX_LENGTH = 10
N_RANDOM_CROPS = 5


def _decode(tokenizer, token_ids) -> str:
    # skip_special_tokens=False para inspeccionar tokens especiales y padding
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def run_for_model(model_name: str, model_id: str) -> None:
    print("=" * 90)
    print(f"MODELO: {model_name} ({model_id})")
    print("=" * 90)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print("No se pudo cargar el tokenizer para este modelo. Se omite la prueba.")
        print(f"Motivo: {type(e).__name__}: {e}")
        print()
        return

    collator = RandomCropDataCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

    # Tokenización sin truncado/padding (como en entrenamiento)
    features = []
    for t in TEXTS:
        enc = tokenizer(t, truncation=False, padding=False)
        feat = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if "token_type_ids" in enc:
            feat["token_type_ids"] = enc["token_type_ids"]
        features.append(feat)

    # Prueba 1: Secuencia corta
    print("\n1) Secuencia corta (esperado: padding al final):")
    batch_short = collator([features[0]])
    short_ids = batch_short["input_ids"][0].tolist()
    short_mask = batch_short["attention_mask"][0].tolist()
    print("decoded:", _decode(tokenizer, short_ids))
    print("len(input_ids):", len(short_ids), "| attention_sum:", sum(short_mask))
    print("has_token_type_ids:", "token_type_ids" in batch_short)

    # Prueba 2: Secuencia larga con crops aleatorios
    print(
        "\n2) Secuencia larga (esperado: crops distintos, longitud fija, máscaras válidas):"
    )
    print("texto original:", TEXTS[1])
    for i in range(N_RANDOM_CROPS):
        batch_long = collator([features[1]])
        long_ids = batch_long["input_ids"][0].tolist()
        long_mask = batch_long["attention_mask"][0].tolist()
        print(f"  -> Crop {i + 1}: {_decode(tokenizer, long_ids)}")
        print(
            f"     len={len(long_ids)} | attention_sum={sum(long_mask)} | "
            f"first_id={long_ids[0]} | last_id={long_ids[-1]}"
        )

    print()


def main() -> None:
    print("==== PRUEBA MULTIMODELO DEL DATA COLLATOR ====\n")
    print(
        f"Configuración: max_length={MAX_LENGTH}, crops_por_modelo={N_RANDOM_CROPS}, "
        f"textos={len(TEXTS)}\n"
    )

    for name, model_id in MODELS.items():
        run_for_model(name, model_id)

    print("==== FIN ====")


if __name__ == "__main__":
    main()
