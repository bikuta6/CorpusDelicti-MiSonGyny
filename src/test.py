import os
import sys
import threading

# Silenciar ANTES de importar transformers
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"

# Monkey-patch para ignorar errores en threads secundarios
_original_excepthook = threading.excepthook
def _silent_excepthook(args):
    if "safetensors" in str(args.exc_value) or "JSONDecodeError" in str(type(args.exc_value)):
        pass  # Ignorar errores de conversión safetensors
    else:
        _original_excepthook(args)
threading.excepthook = _silent_excepthook

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained(
    'IsGarrido/roberta-base-bne'
)
model = AutoModel.from_pretrained(
    'IsGarrido/roberta-base-bne',
    ignore_mismatched_sizes=True,
    add_pooling_layer=False,
    use_safetensors=False
)

text = "Gracias a los datos de la BNE se ha podido desarrollar este modelo del lenguaje."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

cls_embedding = output.last_hidden_state[:, 0, :]
print(f"CLS embedding shape: {cls_embedding.shape}")