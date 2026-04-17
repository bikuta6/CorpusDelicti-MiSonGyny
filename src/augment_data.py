import argparse
import os
import random
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_match(val, targets):
    return val in targets or val.replace(".0", "") in targets or val + ".0" in targets


def get_label_columns(df):
    exclude = {
        "song_id",
        "song_title",
        "artist_id",
        "artist_name",
        "lyrics",
        "language",
        "augmentation",
        "augmentation_idx",
        "source_row_idx",
        "augmentation_slot",
        "row_idx",
    }
    return [col for col in df.columns if col not in exclude]


def get_ordered_methods(methods_dict):
    """
    Returns methods ordered by desired execution priority:
      1) fast local methods
      2) prompting
      3) backtranslation
    Only methods with weight > 0 are enabled.
    """
    priority_order = [
        "char_remove",
        "char_change",
        "aeda",
        "synonym",
        "llm_paraphrase",
        "backtranslation",
    ]

    positive_methods = []
    for m, w in methods_dict.items():
        try:
            if float(w) > 0:
                positive_methods.append(m)
        except (TypeError, ValueError):
            continue

    enabled = [m for m in priority_order if m in positive_methods]
    # Include unknown/custom methods at the end in their config insertion order
    enabled += [m for m in positive_methods if m not in enabled]
    return enabled


class DataAugmentor:
    def __init__(self, methods_config, llm_model="llama3"):
        self.methods_config = methods_config
        self.llm_model = llm_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augmenters = {}
        self.disabled_methods = set()
        self.init_augmenters()

    def _safe_init(self, name: str, init_fn: Callable[[], None]):
        try:
            init_fn()
        except Exception as e:
            self.disabled_methods.add(name)
            print(
                f"  Warning: disabling '{name}' augmenter due to initialization error: {e}"
            )

    def init_augmenters(self):
        if "synonym" in self.methods_config:

            def _init_synonym():
                import nlpaug.augmenter.word as naw
                from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

                # Compatibility patch for nlpaug with newer transformers
                def _convert_token_to_id_compat(self, token):
                    return self.convert_tokens_to_ids(token)

                if not hasattr(PreTrainedTokenizer, "_convert_token_to_id"):
                    PreTrainedTokenizer._convert_token_to_id = (
                        _convert_token_to_id_compat
                    )
                if not hasattr(PreTrainedTokenizerFast, "_convert_token_to_id"):
                    PreTrainedTokenizerFast._convert_token_to_id = (
                        _convert_token_to_id_compat
                    )

                self.augmenters["synonym"] = naw.ContextualWordEmbsAug(
                    model_path="dccuchile/distilbert-base-spanish-uncased",
                    action="substitute",
                    aug_max=None,
                    device=self.device,
                )

            self._safe_init("synonym", _init_synonym)

        if "char_remove" in self.methods_config:

            def _init_char_remove():
                import nlpaug.augmenter.char as nac

                self.augmenters["char_remove"] = nac.RandomCharAug(
                    action="delete", aug_word_max=None
                )

            self._safe_init("char_remove", _init_char_remove)

        if "char_change" in self.methods_config:

            def _init_char_change():
                import nlpaug.augmenter.char as nac

                self.augmenters["char_change"] = nac.RandomCharAug(
                    action="substitute",
                    aug_word_max=None,
                    aug_word_p=0.1,
                    include_numeric=False,
                )

            self._safe_init("char_change", _init_char_change)

        if "backtranslation" in self.methods_config:

            def _init_backtranslation():
                from transformers import MarianMTModel, MarianTokenizer

                self.es_en_tokenizer = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-es-en"
                )
                self.es_en_model = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-es-en"
                ).to(self.device)
                self.en_es_tokenizer = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-es"
                )
                self.en_es_model = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-es"
                ).to(self.device)

            self._safe_init("backtranslation", _init_backtranslation)

    def aeda_augment(self, text):
        if not isinstance(text, str):
            return text

        punctuations = [".", ";", "?", ":", "!", ","]
        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return text

        num_punctuations = max(1, int(0.3 * num_words))

        # Insert punctuation attached to the previous token so there is
        # no space before punctuation, only after (from token joining).
        for _ in range(num_punctuations):
            if not words:
                break

            if len(words) == 1:
                attach_idx = 0
            else:
                attach_idx = random.randint(0, len(words) - 2)

            punc = random.choice(punctuations)
            words[attach_idx] = f"{words[attach_idx]}{punc}"

        return " ".join(words)

    def apply(self, text, method):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text

        if method in self.disabled_methods:
            return text

        try:
            if method == "synonym":
                augmenter = self.augmenters.get("synonym")
                if not augmenter:
                    return text
                res = augmenter.augment(text)
                return res[0] if isinstance(res, list) else res

            elif method == "char_remove":
                augmenter = self.augmenters.get("char_remove")
                if not augmenter:
                    return text
                res = augmenter.augment(text)
                return res[0] if isinstance(res, list) else res

            elif method == "char_change":
                augmenter = self.augmenters.get("char_change")
                if not augmenter:
                    return text
                res = augmenter.augment(text)
                return res[0] if isinstance(res, list) else res

            elif method == "aeda":
                return self.aeda_augment(text)

            elif method == "backtranslation":
                if not all(
                    hasattr(self, attr)
                    for attr in [
                        "es_en_tokenizer",
                        "es_en_model",
                        "en_es_tokenizer",
                        "en_es_model",
                    ]
                ):
                    return text

                inputs = self.es_en_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                translated = self.es_en_model.generate(**inputs)
                en_text = self.es_en_tokenizer.decode(
                    translated[0], skip_special_tokens=True
                )

                inputs = self.en_es_tokenizer(
                    en_text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                back_translated = self.en_es_model.generate(**inputs)
                return self.en_es_tokenizer.decode(
                    back_translated[0], skip_special_tokens=True
                )

            elif method == "llm_paraphrase":
                import requests

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": (
                            "Prafrasea la siguiente letra de canción en español naturalmente, manteniendo "
                            "el mismo significado. Devuelve solo el texto parafraseado, sin explicaciones ni formato adicional, y nada más que el texto parafraseado"
                            f":\n{text}"
                        ),
                        "stream": False,
                    },
                    timeout=60,
                )
                if response.status_code == 200:
                    res = response.json().get("response", "").strip()
                    return res if res else text

            return text
        except Exception as e:
            print(
                f"  Warning: method '{method}' failed during apply; returning original text. Error: {e}"
            )
            return text


def build_target_mask(df, target_classes):
    """
    Returns a boolean mask over df rows indicating which original rows
    are eligible for augmentation according to target_classes.
    """
    if target_classes == "all":
        return np.ones(len(df), dtype=bool)

    label_cols = get_label_columns(df)
    mask = np.zeros(len(df), dtype=bool)

    if isinstance(target_classes, dict):
        for i, row in df.iterrows():
            match = False
            for col, allowed_vals in target_classes.items():
                if col not in df.columns:
                    continue
                val = str(row[col])
                allowed_strs = [
                    str(v).strip()
                    for v in (
                        allowed_vals
                        if isinstance(allowed_vals, list)
                        else [allowed_vals]
                    )
                ]
                if is_match(val, allowed_strs):
                    match = True
                    break
            mask[i] = match
    else:
        targets = [t.strip() for t in str(target_classes).split(",")]
        for i, row in df.iterrows():
            row_labels = [str(row[c]) for c in label_cols if pd.notna(row[c])]
            mask[i] = any(is_match(rl, targets) for rl in row_labels)

    return mask


def build_method_assignment_mask(
    df, eligible_mask, ordered_methods, multiplier, rng, method_probs=None
):
    """
    Build deterministic method assignment mask with shape [N, multiplier],
    where each value is method index in ordered_methods, and -1 means
    no augmentation for that slot (ineligible row or non-original row).
    If method_probs is provided, weighted assignment is used.
    """
    n = len(df)
    m = max(0, int(multiplier))

    mask = np.full((n, m), -1, dtype=int)
    if m == 0 or len(ordered_methods) == 0:
        return mask

    if method_probs is not None:
        probs = np.array(method_probs, dtype=float)
        probs = probs / probs.sum()
    else:
        probs = None

    for i in range(n):
        if not eligible_mask[i]:
            continue

        if probs is None:
            mask[i, :] = rng.integers(low=0, high=len(ordered_methods), size=m)
        else:
            mask[i, :] = rng.choice(len(ordered_methods), size=m, p=probs)

    return mask


def save_method_mask(mask, methods, out_base_path):
    """
    Saves mask in CSV and NPY formats:
      - {out_base_path}_method_mask.csv
      - {out_base_path}_method_mask.npy
      - {out_base_path}_method_index_map.csv
    """
    csv_path = f"{out_base_path}_method_mask.csv"
    npy_path = f"{out_base_path}_method_mask.npy"
    map_path = f"{out_base_path}_method_index_map.csv"

    # mask CSV with slot columns
    slot_cols = [f"slot_{i}" for i in range(mask.shape[1])]
    df_mask = pd.DataFrame(mask, columns=slot_cols)
    df_mask.insert(0, "row_idx", np.arange(mask.shape[0]))
    df_mask.to_csv(csv_path, index=False)

    np.save(npy_path, mask)

    # index map
    df_map = pd.DataFrame({"method_idx": list(range(len(methods))), "method": methods})
    df_map.to_csv(map_path, index=False)

    return csv_path, npy_path, map_path


def process_task(task, task_config, base_dir, use_processed, llm_model, seed=42):
    print(f"\nProcessing {task}...")

    file_name = "processed_train_df.csv" if use_processed else "train_df.csv"
    possible_paths = [
        os.path.join(base_dir, task, file_name),
        os.path.join(base_dir, "data", task, file_name),
        os.path.join(".", "data", task, file_name),
    ]

    train_path = None
    for p in possible_paths:
        if os.path.exists(p):
            train_path = p
            break

    if not train_path:
        print(f"  Could not find {file_name} for {task} in {base_dir}")
        return

    print(f"  Reading from {train_path}")
    df = pd.read_csv(train_path)

    if "augmentation" not in df.columns:
        df["augmentation"] = "original"

    methods_dict = task_config.get("methods", {})
    if not methods_dict:
        print("  No methods configured.")
        return

    ordered_methods = get_ordered_methods(methods_dict)
    if not ordered_methods:
        print("  No methods with positive probability configured.")
        return

    method_probs = np.array(
        [float(methods_dict[m]) for m in ordered_methods], dtype=float
    )
    if method_probs.sum() <= 0:
        print("  Sum of enabled method probabilities is not positive.")
        return
    method_probs = method_probs / method_probs.sum()

    multiplier = int(task_config.get("multiplier", 1))
    target_classes = task_config.get("augment_target", "all")

    # augment only original rows
    original_mask = (
        df["augmentation"].fillna("original").astype(str).eq("original").values
    )
    target_mask = build_target_mask(df, target_classes)
    eligible_mask = original_mask & target_mask

    print(f"  Enabled methods (priority order): {ordered_methods}")
    print(
        f"  Enabled method probabilities: {dict(zip(ordered_methods, method_probs.tolist()))}"
    )
    print(f"  Multiplier: {multiplier}")
    print(f"  Eligible original rows: {eligible_mask.sum()} / {len(df)}")

    # deterministic RNG
    rng = np.random.default_rng(seed)

    method_mask = build_method_assignment_mask(
        df=df,
        eligible_mask=eligible_mask,
        ordered_methods=ordered_methods,
        multiplier=multiplier,
        rng=rng,
        method_probs=method_probs,
    )

    out_base = train_path.replace(".csv", "")
    mask_csv, mask_npy, map_csv = save_method_mask(
        method_mask, ordered_methods, out_base_path=out_base
    )
    print(f"  Saved method mask CSV: {mask_csv}")
    print(f"  Saved method mask NPY: {mask_npy}")
    print(f"  Saved method index map: {map_csv}")

    # init augmentor after method list is finalized
    enabled_methods_dict = {m: methods_dict[m] for m in ordered_methods}
    augmentor = DataAugmentor(enabled_methods_dict, llm_model)

    # group methods by stage for ordered execution
    fast_methods = [
        m
        for m in ordered_methods
        if m in {"char_remove", "char_change", "aeda", "synonym"}
    ]
    prompt_methods = [m for m in ordered_methods if m in {"llm_paraphrase"}]
    slow_methods = [m for m in ordered_methods if m in {"backtranslation"}]
    other_methods = [
        m
        for m in ordered_methods
        if m not in set(fast_methods + prompt_methods + slow_methods)
    ]

    stage_defs = [
        ("fast", fast_methods),
        ("prompting", prompt_methods),
        ("backtranslation", slow_methods),
        ("other", other_methods),
    ]
    stage_defs = [(name, ms) for name, ms in stage_defs if len(ms) > 0]

    # Build reverse index: method -> list of (row_idx, slot)
    assigned = {m: [] for m in ordered_methods}
    for row_idx in range(method_mask.shape[0]):
        for slot in range(method_mask.shape[1]):
            midx = method_mask[row_idx, slot]
            if midx < 0:
                continue
            method = ordered_methods[midx]
            assigned[method].append((row_idx, slot))

    augmented_rows = []

    for stage_name, stage_methods in stage_defs:
        total_stage_jobs = sum(len(assigned[m]) for m in stage_methods)
        print(f"\n  Stage '{stage_name}': {stage_methods} ({total_stage_jobs} jobs)")

        if total_stage_jobs == 0:
            print("    No jobs in this stage.")
            continue

        for method in stage_methods:
            jobs = assigned[method]
            if not jobs:
                continue

            print(f"    Method '{method}' -> {len(jobs)} rows")
            for row_idx, slot in tqdm(
                jobs, desc=f"{task}:{stage_name}:{method}", leave=False
            ):
                row = df.iloc[row_idx]
                new_text = augmentor.apply(row["lyrics"], method)

                new_row = row.copy()
                new_row["lyrics"] = new_text
                new_row["augmentation"] = method
                new_row["augmentation_idx"] = ordered_methods.index(method)
                new_row["source_row_idx"] = row_idx
                new_row["augmentation_slot"] = slot
                augmented_rows.append(new_row)

            # Lightweight per-method sample preview
            sample_count = min(2, len(jobs))
            if sample_count > 0:
                preview_idxs = jobs[:sample_count]
                print(f"      Sample results for '{method}':")
                for pr_row_idx, pr_slot in preview_idxs:
                    src_text = str(df.iloc[pr_row_idx]["lyrics"])
                    # find the just-added row quickly from tail
                    preview_row = next(
                        (
                            r
                            for r in reversed(augmented_rows)
                            if int(r.get("source_row_idx", -1)) == pr_row_idx
                            and int(r.get("augmentation_slot", -1)) == pr_slot
                            and str(r.get("augmentation", "")) == method
                        ),
                        None,
                    )
                    out_text = (
                        str(preview_row["lyrics"])
                        if preview_row is not None
                        else src_text
                    )
                    print(f"        row={pr_row_idx}, slot={pr_slot}")
                    print(f"          in : {src_text[:120].replace(chr(10), ' ')}")
                    print(f"          out: {out_text[:120].replace(chr(10), ' ')}")

    if augmented_rows:
        df_new = pd.DataFrame(augmented_rows)
        df_augmented = pd.concat([df, df_new], ignore_index=True)
    else:
        df_augmented = df

    out_path = train_path.replace(".csv", "_augmented.csv")
    df_augmented.to_csv(out_path, index=False)

    print(f"\n  Saved augmented {task} to {out_path} ({len(df_augmented)} total rows)")
    print(f"  New augmented rows: {len(augmented_rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="augmentation_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--base-dir", type=str, default="data", help="Base data directory"
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Use processed_train_df.csv instead of train_df.csv",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma4:e2b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks to process (e.g., task1,task2). Default is all tasks in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic method mask assignment",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found!")
        return

    config = load_config(args.config)

    tasks_to_run = config.keys()
    if args.tasks:
        tasks_to_run = [t.strip() for t in args.tasks.split(",")]

    # Global seeds for reproducibility where relevant
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for task in tasks_to_run:
        if task in config:
            process_task(
                task=task,
                task_config=config[task],
                base_dir=args.base_dir,
                use_processed=args.processed,
                llm_model=args.llm_model,
                seed=args.seed,
            )
        else:
            print(f"Task {task} not found in config file. Skipping...")


if __name__ == "__main__":
    main()
