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
    # 1. Cargar datos de prueba
    print(f"Cargando datos de prueba desde {args.test_file}...")
    df = pd.read_csv(args.test_file)
    if "lyrics" not in df.columns:
        raise ValueError("El archivo CSV debe contener la columna 'lyrics'.")

    # Renombramos lyrics a text para que sea compatible con la función de chunks
    ds = Dataset.from_pandas(
        df.rename(columns={"lyrics": "text"}), preserve_index=False
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 2. Configurar el mapeo de etiquetas y modelos a utilizar
    labels_map = {
        "sexualization": ("S", args.model_s or args.model_name),
        "violence": ("V", args.model_v or args.model_name),
        "hate": ("H", args.model_h or args.model_name),
    }

    all_preds = {}

    # 3. Iterar sobre cada etiqueta y cargar su respectivo modelo OVR
    for label_name, (short_col, specific_model_name) in labels_map.items():
        if specific_model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"El modelo '{specific_model_name}' no existe en las configuraciones."
            )

        cfg = MODEL_CONFIGS[specific_model_name]

        # Construir ruta del modelo OVR
        model_path = os.path.join(
            args.models_dir, f"{specific_model_name}_{label_name.capitalize()}"
        )

        print(f"\n--- Procesando etiqueta: {label_name.upper()} ({short_col}) ---")
        print(f"Cargando modelo OVR desde {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. Asegúrate de haberlo entrenado con train_ovr.py."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = load_bert_like_classifier(model_path, device)

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

        logits = pred_output.predictions
        # En OVR, logits tiene forma (N, 1), aplanamos para obtener (N,)
        probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()

        # Binarizar según el threshold (por defecto 0.5)
        preds = (probs >= args.threshold).astype(int)
        all_preds[short_col] = preds

        # Liberar memoria del modelo actual antes de cargar el siguiente
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Mapear al DataFrame original
    for short_col in ["S", "V", "H"]:
        df[short_col] = all_preds[short_col]

    # 5. Guardar Resultados
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    if "song_id" in df.columns:
        df_out = df[["song_id", "S", "V", "H"]].rename(columns={"song_id": "id"})
    elif "id" in df.columns:
        df_out = df[["id", "S", "V", "H"]]
    else:
        df["id"] = [f"T2_TEST_{i + 1:04d}" for i in range(len(df))]
        df_out = df[["id", "S", "V", "H"]]

    df_out.to_csv(args.output_file, index=False)

    print(f"\nPredicciones combinadas guardadas exitosamente en: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de predicción para Task 2 usando modelos OVR"
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
        default="../../task_2_predictions.csv",
        help="Ruta donde se guardará el CSV de salida",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="../../models/task2/OVR",
        help="Directorio base donde se encuentran los modelos OVR",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BETO",
        help="Nombre del modelo base en bert_model_configs.py (ej. BETO). Se usará por defecto para todas las etiquetas.",
    )
    parser.add_argument(
        "--model_s",
        type=str,
        default=None,
        help="Sobreescribe el modelo a usar específicamente para la etiqueta Sexualization (ej. RoBERTa_baseline)",
    )
    parser.add_argument(
        "--model_v",
        type=str,
        default=None,
        help="Sobreescribe el modelo a usar específicamente para la etiqueta Violence",
    )
    parser.add_argument(
        "--model_h",
        type=str,
        default=None,
        help="Sobreescribe el modelo a usar específicamente para la etiqueta Hate",
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
