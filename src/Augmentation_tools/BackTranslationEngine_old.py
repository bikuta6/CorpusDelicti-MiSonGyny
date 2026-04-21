import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class BackTranslationEngine:
    def __init__(self, modelo_es_en="Helsinki-NLP/opus-mt-es-en", modelo_en_es="Helsinki-NLP/opus-mt-en-es", output_file=None,
                 creative=True):
        self.modelo_es_en = modelo_es_en
        self.modelo_en_es = modelo_en_es
        self.output_file = output_file
        self.creative = creative

        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_gpu_available = torch.cuda.is_available()
        print(f"Usando dispositivo: {self.device}")

        # Carga los tokenizers y modelos
        self.tokenizer_es_en = AutoTokenizer.from_pretrained(self.modelo_es_en)
        self.tokenizer_en_es = AutoTokenizer.from_pretrained(self.modelo_en_es)

        if self._is_gpu_available:
            print("GPU disponible. Cargando modelos en GPU...")
            self.model_es_en = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_es_en).to(self.device)
            self.model_en_es = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_en_es).to(self.device)
        else:            
            print("GPU no disponible. Cargando modelos en CPU...")
            self.model_es_en = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_es_en)
            self.model_en_es = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_en_es)

    def traducir(self, texto, tokenizer, model, max_length=512, creative=True):
        """
        Traduce un texto usando un modelo seq2seq.

        Parámetros:
        - texto: string de entrada
        - tokenizer: tokenizer del modelo
        - model: modelo de traducción
        - max_length: longitud máxima de salida

        Devuelve:
        - texto traducido
        """
        inputs = tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )

        if self._is_gpu_available:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if creative:
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                temperature=0.8
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=max_length
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def back_translate(self, texto, creative=True):
        """
        Aplica back translation:
        español -> inglés -> español

        Parámetros:
        - texto: string en español

        Devuelve:
        - texto transformado
        """
        texto_en = self.traducir(texto, self.tokenizer_es_en, self.model_es_en, creative=creative)
        texto_bt = self.traducir(texto_en, self.tokenizer_en_es, self.model_en_es, creative=creative)
        return texto_bt

    def augmentar_dataset_backtranslation(self, df, columna_texto="lyrics", columna_id="song_id"):
        """
        Aplica back translation a todos los textos de un dataset.

        Parámetros:
        - df: DataFrame original.
        - columna_texto: nombre de la columna con los textos.
        - columna_id: nombre de la columna con las etiquetas.

        Devuelve:
        - DataFrame con datos originales y sintéticos.
        """
        
        rows_nuevas = []
        for _, row in df.iterrows():
            # Captura el texto original
            texto_original = row[columna_texto]
            
            # Realiza la traducción inversa
            texto_bt = self.back_translate(texto_original)
            
            # Crea un nuevo diccionario con los datos originales y el nuevo texto traducido
            nueva_fila = row.to_dict()
            nueva_fila[columna_id] = f"{row[columna_id]}_bt"  # Modifica el ID
            nueva_fila[columna_texto] = texto_bt  # Agrega el texto traducido
            
            rows_nuevas.append(nueva_fila)  # Agrega la nueva fila a la lista

        # Crea un nuevo DataFrame con las filas generadas
        df_sintetico = pd.DataFrame(rows_nuevas)
        return df_sintetico