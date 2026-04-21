import pandas as pd
import re
import torch
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class BackTranslationEngine:
    def __init__(
        self,
        modelo_es_en="Helsinki-NLP/opus-mt-es-en",
        modelo_en_es="Helsinki-NLP/opus-mt-en-es",
        output_file=None,
        creative=True,
        chunk_size_words=30,
        chunk_min_words=12,
        chunk_max_words=40
    ):
        self.modelo_es_en = modelo_es_en
        self.modelo_en_es = modelo_en_es
        self.output_file = output_file
        self.creative = creative

        # chunk_size_words ahora funciona como target
        self.chunk_size_words = chunk_size_words
        self.chunk_min_words = chunk_min_words
        self.chunk_max_words = chunk_max_words

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_gpu_available = torch.cuda.is_available()
        print(f"Usando dispositivo: {self.device}")

        print("Cargando tokenizers...")
        self.tokenizer_es_en = AutoTokenizer.from_pretrained(self.modelo_es_en)
        self.tokenizer_en_es = AutoTokenizer.from_pretrained(self.modelo_en_es)

        print("Cargando modelos...")
        self.model_es_en = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_es_en).to(self.device)
        self.model_en_es = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_en_es).to(self.device)

        self.model_es_en.eval()
        self.model_en_es.eval()

        if self._is_gpu_available:
            print("GPU disponible. Modelos cargados en GPU.")
        else:
            print("GPU no disponible. Modelos cargados en CPU.")

    def limpiar_texto(self, texto):
        if not isinstance(texto, str):
            return ""

        # primero limpieza musical / repetitiva
        texto = self.limpiar_repeticiones_musicales(texto)
        texto = self.limpiar_repeticiones_excesivas(texto, max_reps=3)

        # normalización para chunking y traducción
        texto = self.normalizar_texto_para_chunks(texto)

        return texto

    def normalizar_texto_para_chunks(self, texto):
        """
        Normaliza texto sin destruir completamente la estructura útil
        para dividir en chunks.
        """
        if not isinstance(texto, str):
            return ""

        # Normalización unicode
        texto = unicodedata.normalize("NFKC", texto)

        # Quitar algunos caracteres raros frecuentes
        texto = texto.replace("ą", " ").replace("Ą", " ")
        texto = texto.replace("_", " ")

        # Unificar algunos separadores raros
        texto = texto.replace(";", ",")
        texto = texto.replace("|", ",")
        texto = texto.replace("/", " / ")

        # Mantener puntuación útil para dividir;
        # quitar símbolos muy raros pero conservar .,!?:
        texto = re.sub(r"[^\w\s\.,!?:\-áéíóúüñÁÉÍÓÚÜÑ]", " ", texto, flags=re.UNICODE)

        # Compactar signos repetidos
        texto = re.sub(r"[!]{2,}", "!", texto)
        texto = re.sub(r"[?]{2,}", "?", texto)
        texto = re.sub(r"[.]{2,}", ".", texto)
        texto = re.sub(r"[,]{2,}", ",", texto)

        # Espacios
        texto = re.sub(r"\s+", " ", texto).strip()

        return texto

    def limpiar_repeticiones_musicales(self, texto):
        if not isinstance(texto, str):
            return ""

        texto = re.sub(r'(?i)\b(oh)(\s*,?\s*\1\b){3,}', 'oh oh oh', texto)
        texto = re.sub(r'(?i)\b(yeah)(\s*,?\s*\1\b){3,}', 'yeah yeah yeah', texto)
        texto = re.sub(r'(?i)\b(uh)(\s*,?\s*\1\b){3,}', 'uh uh uh', texto)
        texto = re.sub(r'(?i)\b(ay)(\s*,?\s*\1\b){3,}', 'ay ay ay', texto)
        texto = re.sub(r'(?i)\b(ja)(\s*,?\s*\1\b){3,}', 'ja ja ja', texto)
        texto = re.sub(r'(?i)\b(je)(\s*,?\s*\1\b){3,}', 'je je je', texto)
        texto = re.sub(r'(?i)\b(eh)(\s*,?\s*\1\b){3,}', 'eh eh eh', texto)
        texto = re.sub(r'(?i)\b(ah)(\s*,?\s*\1\b){3,}', 'ah ah ah', texto)
        texto = re.sub(r'(?i)\b(hah)(\s*,?\s*\1\b){3,}', 'hah hah hah', texto)

        return texto

    def limpiar_repeticiones_excesivas(self, texto, max_reps=3):
        if not isinstance(texto, str):
            return ""

        palabras = texto.split()
        if not palabras:
            return texto

        resultado = []
        anterior = None
        contador = 0

        for palabra in palabras:
            palabra_norm = palabra.lower()

            if palabra_norm == anterior:
                contador += 1
            else:
                anterior = palabra_norm
                contador = 1

            if contador <= max_reps:
                resultado.append(palabra)

        return " ".join(resultado)

    def _contar_palabras(self, fragmentos):
        return len(" ".join(fragmentos).split())

    def dividir_en_chunks(
        self,
        texto,
        chunk_size_words=None,
        min_words=None,
        max_words=None
    ):
        """
        Divide el texto de forma híbrida:
        1) normaliza
        2) corta por puntuación
        3) reagrupe fragmentos pequeños
        4) si algo queda muy largo, fallback por palabras
        """
        if chunk_size_words is None:
            chunk_size_words = self.chunk_size_words
        if min_words is None:
            min_words = self.chunk_min_words
        if max_words is None:
            max_words = self.chunk_max_words

        if not isinstance(texto, str) or not texto.strip():
            return []

        texto = self.normalizar_texto_para_chunks(texto)

        if not texto:
            return []

        # Cortar por puntuación media/fuerte
        partes = re.split(r"\s*[\.,!?:]\s*", texto)
        partes = [p.strip() for p in partes if p.strip()]

        if not partes:
            return []

        chunks = []
        actual = []

        for parte in partes:
            palabras_parte = len(parte.split())
            palabras_actual = self._contar_palabras(actual)

            if palabras_actual == 0:
                actual.append(parte)
                continue

            # si cabe cómodamente en el target, agregar
            if palabras_actual + palabras_parte <= chunk_size_words:
                actual.append(parte)
            else:
                # si el actual quedó muy pequeño, permitir crecer hasta max_words
                if palabras_actual < min_words and palabras_actual + palabras_parte <= max_words:
                    actual.append(parte)
                else:
                    chunks.append(" ".join(actual).strip())
                    actual = [parte]

        if actual:
            chunks.append(" ".join(actual).strip())

        # Fallback: si algún chunk quedó demasiado largo, partir por palabras
        final_chunks = []
        for chunk in chunks:
            palabras = chunk.split()

            if len(palabras) <= max_words:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(palabras), chunk_size_words):
                    subchunk = " ".join(palabras[i:i + chunk_size_words]).strip()
                    if subchunk:
                        final_chunks.append(subchunk)

        return final_chunks

    def unir_chunks(self, chunks):
        """
        Une los chunks nuevamente en un solo texto.
        """
        if not chunks:
            return ""
        return " ".join(
            chunk.strip() for chunk in chunks
            if isinstance(chunk, str) and chunk.strip()
        )

    def traducir(self, texto, tokenizer, model, max_length=512, creative=None):
        """
        Traduce un texto usando un modelo seq2seq.
        """
        if creative is None:
            creative = self.creative

        if not isinstance(texto, str) or not texto.strip():
            return texto

        inputs = tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
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

    def back_translate(self, texto, creative=None):
        """
        Aplica back translation:
        español -> inglés -> español
        """
        if creative is None:
            creative = self.creative

        if not isinstance(texto, str) or not texto.strip():
            return texto

        texto_en = self.traducir(
            texto,
            self.tokenizer_es_en,
            self.model_es_en,
            creative=creative
        )

        texto_bt = self.traducir(
            texto_en,
            self.tokenizer_en_es,
            self.model_en_es,
            creative=creative
        )

        return texto_bt

    def back_translate_por_chunks(self, texto, creative=None, chunk_size_words=None):
        """
        Divide el texto en chunks, aplica backtranslation a cada chunk
        y luego los vuelve a unir.
        """
        if creative is None:
            creative = self.creative

        chunks = self.dividir_en_chunks(texto, chunk_size_words=chunk_size_words)

        if not chunks:
            return texto, 0

        chunks_traducidos = []
        total_chunks = len(chunks)

        for chunk_idx, chunk in enumerate(chunks, start=1):
            try:
                chunk_bt = self.back_translate(chunk, creative=creative)
                chunks_traducidos.append(chunk_bt)
            except Exception as e:
                print(f"    Error en chunk {chunk_idx}/{total_chunks}: {e}")
                chunks_traducidos.append(chunk)  # fallback: conserva chunk original

        texto_final = self.unir_chunks(chunks_traducidos)
        texto_final = self.limpiar_repeticiones_excesivas(texto_final, max_reps=3)
        texto_final = re.sub(r"\s+", " ", texto_final).strip()

        return texto_final, total_chunks

    def augmentar_dataset_backtranslation(
        self,
        df,
        columna_texto="lyrics",
        columna_id="song_id",
        save_every=None
    ):
        """
        Aplica back translation por chunks a todos los textos de un dataset.
        """
        total = len(df)
        rows_nuevas = []

        print(f"Total de canciones a traducir: {total}")
        print(f"Chunk target: {self.chunk_size_words} palabras")
        print(f"Chunk min/max: {self.chunk_min_words}/{self.chunk_max_words}")

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            texto_original = row[columna_texto]
            texto_limpio = self.limpiar_texto(texto_original)

            try:
                texto_bt, total_chunks = self.back_translate_por_chunks(texto_limpio)

                nueva_fila = row.to_dict()
                nueva_fila[columna_id] = f"{row[columna_id]}_bt"
                nueva_fila[columna_texto] = texto_bt
                rows_nuevas.append(nueva_fila)

                print(f"[{idx}/{total}] Canciones traducidas: {idx} | chunks: {total_chunks}")

            except Exception as e:
                print(f"[{idx}/{total}] Error procesando song_id={row[columna_id]}: {e}")

            if (
                save_every is not None
                and self.output_file is not None
                and idx % save_every == 0
            ):
                df_parcial = pd.DataFrame(rows_nuevas)
                df_parcial.to_csv(self.output_file, index=False)
                print(f"Guardado parcial en '{self.output_file}' con {len(rows_nuevas)} canciones.")

        df_sintetico = pd.DataFrame(rows_nuevas)

        if self.output_file is not None:
            df_sintetico.to_csv(self.output_file, index=False)
            print(f"Guardado final en '{self.output_file}' con {len(df_sintetico)} canciones.")

        return df_sintetico