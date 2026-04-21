import re
from typing import Any
from difflib import SequenceMatcher

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


class LLMEngine:
    def __init__(
        self,
        modelo="google/flan-t5-small",
        output_file=None,
        creative=True
    ):
        self.modelo = modelo
        self.output_file = output_file
        self.creative = creative

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_gpu_available = torch.cuda.is_available()
        print(f"Usando dispositivo: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modelo,
            trust_remote_code=True,
            use_fast=False,
            legacy=True
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(self.modelo, trust_remote_code=True)
        self.is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}

        if self._is_gpu_available:
            print("GPU disponible. Cargando modelo en GPU...")
            model_kwargs["torch_dtype"] = torch.float16
        else:
            print("GPU no disponible. Cargando modelo en CPU...")

        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.modelo,
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.modelo,
                **model_kwargs
            )

        if self._is_gpu_available:
            self.model = self.model.to(self.device)

        self.model.eval()

    def limpiar_texto(self, texto: str) -> str:
        texto = str(texto)
        texto = texto.replace("–", " ")
        texto = texto.replace("...", " ")
        texto = texto.replace("Ą", " ")
        texto = re.sub(r"\s+", " ", texto)
        return texto.strip()

    def generar_texto(self, prompt, max_new_tokens=32, creative=True):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        if self._is_gpu_available:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if creative:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.8,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

        if not self.is_encoder_decoder:
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
        else:
            generated_ids = outputs[0]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def dividir_en_chunks(self, texto, max_words=12):
        texto = self.limpiar_texto(texto)
        palabras = texto.split()

        return [
            " ".join(palabras[i:i + max_words])
            for i in range(0, len(palabras), max_words)
        ]

    def similitud_texto(self, a, b):
        return SequenceMatcher(None, str(a), str(b)).ratio()

    def es_demasiado_ingles(self, texto):
        palabras = texto.lower().split()
        if not palabras:
            return False

        en_words = {
            "the", "and", "you", "is", "are", "that", "this",
            "all", "time", "through", "official"
        }
        en_count = sum(1 for p in palabras if p in en_words)
        return en_count > len(palabras) * 0.3

    def salida_valida(self, original, salida, similitud_max=0.93):
        if not isinstance(salida, str):
            return False

        s = self.limpiar_texto(salida)
        o = self.limpiar_texto(original)

        if not s:
            return False

        if s.lower() == o.lower():
            return False

        if "<extra_id_" in s.lower():
            return False

        if "parafrasea" in s.lower() or "reescribe" in s.lower():
            return False

        if self.es_demasiado_ingles(s):
            return False

        invalid_patterns = [
            "the following lyrics",
            "the final word is",
            "title:",
            "artist:",
            "pese in spanish",
            "spell in spanish",
            "spelled in spanish",
            "all the time",
            "right through"
        ]
        if any(p in s.lower() for p in invalid_patterns):
            return False

        if re.search(r"\b(luv|ruv|sugo)\b", s.lower()):
            return False

        if len(s.split()) < 4:
            return False

        palabras = s.lower().split()
        unique_ratio = len(set(palabras)) / max(len(palabras), 1)
        if unique_ratio < 0.5:
            return False

        if self.similitud_texto(o, s) > similitud_max:
            return False

        return True

    def parafrasear(self, texto, creative=True, max_new_tokens=32):
        texto = self.limpiar_texto(texto)
        prompt = f"Reescribe completamente con otras palabras en español:\n{texto}"
        return self.generar_texto(
            prompt,
            max_new_tokens=max_new_tokens,
            creative=creative
        )

    def augmentar_texto_por_chunks(
        self,
        texto,
        max_words=12,
        max_new_tokens=32
    ):
        chunks = self.dividir_en_chunks(texto, max_words=max_words)
        if not chunks:
            return self.limpiar_texto(texto)

        resultado = []

        for chunk in chunks:
            nuevo = None

            for _ in range(2):
                intento = self.parafrasear(
                    chunk,
                    creative=self.creative,
                    max_new_tokens=max_new_tokens
                )
                if self.salida_valida(chunk, intento, similitud_max=0.93):
                    nuevo = self.limpiar_texto(intento)
                    break

            if nuevo is not None:
                resultado.append(nuevo)
            else:
                resultado.append(chunk)

        return " ".join(resultado)

    def augmentar_dataset(
        self,
        df,
        columna_texto="lyrics",
        columna_id="song_id",
        n=1,
        max_words=12,
        max_new_tokens=32
    ):
        rows_nuevas = []

        for _, row in df.iterrows():
            texto_original = row[columna_texto]

            if pd.isna(texto_original):
                continue

            generadas = set()

            for i in range(n):
                texto_aug = self.augmentar_texto_por_chunks(
                    texto_original,
                    max_words=max_words,
                    max_new_tokens=max_new_tokens
                )

                if not self.salida_valida(texto_original, texto_aug, similitud_max=0.97):
                    continue

                if texto_aug in generadas:
                    continue

                generadas.add(texto_aug)

                nueva = row.to_dict()
                nueva[columna_id] = f"{row[columna_id]}_llm_{i+1}"
                nueva[columna_texto] = texto_aug
                rows_nuevas.append(nueva)

        return pd.DataFrame(rows_nuevas)