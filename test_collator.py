from transformers import AutoTokenizer
import sys
sys.path.append('src')
from random_crop_collator import RandomCropDataCollator

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
collator = RandomCropDataCollator(tokenizer=tokenizer, max_length=10)

texts = [
    "hola mundo", # Corta (se debe rellenar con [PAD])
    "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince" # Larga (se debe recortar aleatoriamente)
]

# 1. Tokenizamos sin truncar ni hacer padding (igual que en el entrenamiento)
features = []
for t in texts:
    enc = tokenizer(t, truncation=False, padding=False)
    features.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

print("==== PRUEBA DEL DATA COLLATOR ====\n")

# Prueba 1: Secuencia corta
print("1. Secuencia corta (debe tener [PAD] al final):")
batch_short = collator([features[0]])
print(tokenizer.decode(batch_short["input_ids"][0]))

# Prueba 2: Secuencia larga (múltiples llamadas para ver la aleatoriedad)
print("\n2. Secuencia larga (max_length=10, preservando [CLS] y [SEP]):")
print(f"Texto original: {texts[1]}")
for i in range(5):
    batch_long = collator([features[1]])
    print(f"  -> Crop {i+1}:", tokenizer.decode(batch_long["input_ids"][0]))
