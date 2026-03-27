import pandas as pd
import re, unicodedata, hashlib
from difflib import SequenceMatcher
from tqdm import tqdm
cur = pd.read_csv("data/task1/processed_train.csv")      # columnas: ... lyrics, label
prev = pd.read_csv("prev_data/task1/processed_train.csv") # columnas: id, lyrics, label

cur = cur[["lyrics", "label"]].copy()
prev = prev[["lyrics", "label"]].copy()

cur["source"] = "current"
prev["source"] = "previous"

def normalize_lyrics(t):
    t = "" if pd.isna(t) else str(t)
    t = t.lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("utf-8")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\w\s]", "", t)
    return t

def make_key(t):
    n = normalize_lyrics(t)
    return hashlib.sha1(n.encode("utf-8")).hexdigest()

cur["dup_key"] = cur["lyrics"].apply(make_key)
prev["dup_key"] = prev["lyrics"].apply(make_key)

merged = pd.concat([cur, prev], ignore_index=True)

# detectar claves presentes en ambos datasets (duplicado exacto tras normalización)
sources_per_key = merged.groupby("dup_key")["source"].nunique()
both_keys = set(sources_per_key[sources_per_key == 2].index)

exact_dups = merged[merged["dup_key"].isin(both_keys)].copy()
exact_dups = exact_dups.sort_values(["dup_key", "source"]) 

# detectar conflictos de etiqueta para misma canción/letra
conflicts = (merged.groupby("dup_key")["label"].nunique() > 1)
conflict_keys = set(conflicts[conflicts].index)

# opción conservadora: quitar conflictos
merged_no_conflict = merged[~merged["dup_key"].isin(conflict_keys)].copy()

# dedup: prioriza current (se queda la primera al ordenar)
merged_no_conflict["source_prio"] = merged_no_conflict["source"].map({"current": 0, "previous": 1})
merged_no_conflict = merged_no_conflict.sort_values("source_prio")
final = merged_no_conflict.drop_duplicates(subset="dup_key", keep="first")

print("current:", len(cur))
print("previous:", len(prev))
print("merged raw:", len(merged))
print("exact duplicate keys in both datasets:", len(both_keys))
print("exact duplicate rows (current+previous):", len(exact_dups))
print("conflicting keys:", len(conflict_keys))
print("final dedup:", len(final))

# guardar salida de duplicados exactos para inspección
exact_out = exact_dups[["source", "lyrics", "label", "dup_key"]].copy()
#exact_out.to_csv("data/task1/duplicates_exact_between_datasets.csv", index=False)

# detectar letras muy parecidas (no exactas) entre current y previous
# IMPORTANTE: hashes SOLO sirven para igualdad exacta, no para similitud.
cur_norm = cur[["lyrics", "label", "dup_key"]].copy()
prev_norm = prev[["lyrics", "label", "dup_key"]].copy()
cur_norm["lyrics_norm"] = cur_norm["lyrics"].apply(normalize_lyrics)
prev_norm["lyrics_norm"] = prev_norm["lyrics"].apply(normalize_lyrics)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

threshold = 0.94
near_rows = []

for i, r_cur in tqdm(cur_norm.iterrows(), total=len(cur_norm)):
    text_cur = r_cur["lyrics_norm"]
    len_cur = len(text_cur)
    if len_cur == 0:
        continue

    # filtro barato por longitud para reducir comparaciones
    candidates = prev_norm[
        (prev_norm["lyrics_norm"].str.len() >= int(len_cur * 0.8))
        & (prev_norm["lyrics_norm"].str.len() <= int(len_cur * 1.2))
    ]

    best_score = 0.0
    best_match = None
    for _, r_prev in tqdm(candidates.iterrows(), total=len(candidates)):
        if r_cur["dup_key"] == r_prev["dup_key"]:
            continue  # ya contado como exacto
        score = similarity(text_cur, r_prev["lyrics_norm"])
        if score > best_score:
            best_score = score
            best_match = r_prev

    if best_match is not None and best_score >= threshold:
        near_rows.append(
            {
                "similarity": round(best_score, 4),
                "current_lyrics": r_cur["lyrics"],
                "current_label": r_cur["label"],
                "previous_lyrics": best_match["lyrics"],
                "previous_label": best_match["label"],
                "current_dup_key": r_cur["dup_key"],
                "previous_dup_key": best_match["dup_key"],
            }
        )

near_df = pd.DataFrame(near_rows).sort_values("similarity", ascending=False)
#near_df.to_csv("data/task1/duplicates_near_between_datasets.csv", index=False)

print("near-duplicate pairs (similarity >=", threshold, "):", len(near_df))