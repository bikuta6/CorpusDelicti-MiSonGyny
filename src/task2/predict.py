import argparse
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_model_configs import MODEL_CONFIGS

from bert_pooling import load_bert_like_classifier, predict_with_chunks


def main(args):
    # 1. Cargar configuración del modelo original
    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"El modelo '{args.model_name}' no existe en las configuraciones."
        )
    cfg = MODEL_CONFIGS[args.model_name]

    # 2. Cargar datos de prueba
    print(f"Cargando datos de prueba desde {args.test_file}...")
    df = pd.read_csv(args.test_file)
    if "lyrics" not in df.columns or "song_id" not in df.columns:
        raise ValueError(
            "El archivo CSV debe contener las columnas 'song_id' y 'lyrics'."
        )

    # Renombramos lyrics a text para que sea compatible con la función de chunks
    ds = Dataset.from_pandas(
        df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

    # 3. Cargar Modelo y Tokenizador
    print(f"Cargando modelo y tokenizador desde {args.model_path}...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_bert_like_classifier(args.model_path, device)

    # 4. Realizar predicciones (con chunks para letras largas)
    print("Realizando predicciones...")
    pred_output = predict_with_chunks(
        dataset=ds,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=cfg.max_len,
        batch_size=args.batch_size,
        aggregation="max",
    )

    # Para Task 2 (Multietiqueta), usamos sigmoide en lugar de softmax
    logits = pred_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # Binarizar según el threshold (por defecto 0.5)
    preds = (probs >= args.threshold).astype(int)

    # 5. Mapear de vuelta a las columnas originales de la Task 2
    label_cols = ["type_sexualization", "type_violence", "type_hate"]
    for i, col in enumerate(label_cols):
        df[col] = preds[:, i]

    # 6. Guardar Resultados
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    out_cols = ["song_id"] + label_cols
    df_out = df[out_cols]
    df_out.to_csv(args.output_file, index=False)

    print(f"\nPredicciones guardadas exitosamente en: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de predicción para Task 2")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al directorio del modelo entrenado (ej. ../../models/task2/single/BETO)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Ruta al archivo CSV de test que contiene 'song_id' y 'lyrics'",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Ruta donde se guardará el CSV de salida",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BETO",
        help="Nombre del modelo en bert_model_configs.py (ej. BETO)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de decisión para las clases positivas (0.5 por defecto)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Tamaño de batch para la inferencia"
    )
    args = parser.parse_args()
    main(args)
