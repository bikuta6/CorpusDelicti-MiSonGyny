import os
import argparse
import yaml
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class DataAugmentor:
    def __init__(self, methods_config, llm_model='llama3'):
        self.methods_config = methods_config
        self.llm_model = llm_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.augmenters = {}
        self.init_augmenters()

    def init_augmenters(self):
        if 'synonym' in self.methods_config:
            import nlpaug.augmenter.word as naw
            # Using HuggingFace Spanish BERT for contextual word substitution
            self.augmenters['synonym'] = naw.ContextualWordEmbsAug(
                model_path='dccuchile/bert-base-spanish-wwm-cased', 
                action="substitute",
                device=self.device
            )
        if 'char_remove' in self.methods_config:
            import nlpaug.augmenter.char as nac
            self.augmenters['char_remove'] = nac.RandomCharAug(action="delete")
        if 'char_change' in self.methods_config:
            import nlpaug.augmenter.char as nac
            self.augmenters['char_change'] = nac.RandomCharAug(action="substitute")
        if 'backtranslation' in self.methods_config:
            from transformers import MarianMTModel, MarianTokenizer
            # HuggingFace Backtranslation
            self.es_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
            self.es_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en').to(self.device)
            self.en_es_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
            self.en_es_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es').to(self.device)

    def aeda_augment(self, text):
        if not isinstance(text, str):
            return text
        punctuations = ['.', ';', '?', ':', '!', ',']
        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return text
        num_punctuations = max(1, int(0.1 * num_words))
        
        for _ in range(num_punctuations):
            pos = random.randint(0, len(words))
            punc = random.choice(punctuations)
            words.insert(pos, punc)
        return ' '.join(words)

    def apply(self, text, method):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text
            
        try:
            if method == 'synonym':
                res = self.augmenters['synonym'].augment(text)
                return res[0] if isinstance(res, list) else res
            elif method == 'char_remove':
                res = self.augmenters['char_remove'].augment(text)
                return res[0] if isinstance(res, list) else res
            elif method == 'char_change':
                res = self.augmenters['char_change'].augment(text)
                return res[0] if isinstance(res, list) else res
            elif method == 'aeda':
                return self.aeda_augment(text)
            elif method == 'backtranslation':
                # es -> en -> es
                inputs = self.es_en_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                translated = self.es_en_model.generate(**inputs)
                en_text = self.es_en_tokenizer.decode(translated[0], skip_special_tokens=True)
                
                inputs = self.en_es_tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                back_translated = self.en_es_model.generate(**inputs)
                return self.en_es_tokenizer.decode(back_translated[0], skip_special_tokens=True)
            elif method == 'llm_paraphrase':
                # Strictly using Ollama
                import requests
                response = requests.post('http://localhost:11434/api/generate', json={
                    "model": self.llm_model,
                    "prompt": f"Paraphrase the following Spanish text naturally, keeping the same meaning. Output ONLY the paraphrased text and nothing else:\n{text}",
                    "stream": False
                }, timeout=60)
                if response.status_code == 200:
                    res = response.json().get('response', '').strip()
                    return res if res else text
            return text
        except Exception as e:
            return text

def get_label_columns(df):
    exclude = {'song_id', 'song_title', 'artist_id', 'artist_name', 'lyrics', 'language', 'augmentation'}
    return [col for col in df.columns if col not in exclude]

def is_match(val, targets):
    return val in targets or val.replace('.0', '') in targets or val + '.0' in targets

def process_task(task, task_config, base_dir, use_processed, llm_model):
    print(f"\nProcessing {task}...")
    
    file_name = "processed_train_df.csv" if use_processed else "train_df.csv"
    
    possible_paths = [
        os.path.join(base_dir, task, file_name),
        os.path.join(base_dir, 'data', task, file_name),
        os.path.join('.', 'data', task, file_name)
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
    
    if 'augmentation' not in df.columns:
        df['augmentation'] = 'original'
        
    methods_dict = task_config.get('methods', {})
    if not methods_dict:
        print("  No methods configured.")
        return
        
    methods = list(methods_dict.keys())
    probs = list(methods_dict.values())
    probs = np.array(probs) / sum(probs)
    
    multiplier = task_config.get('multiplier', 1)
    target_classes = task_config.get('augment_target', 'all')
    
    augmentor = DataAugmentor(methods_dict, llm_model)
    
    label_cols = get_label_columns(df)
    print(f"  Detected label columns: {label_cols}")
    
    augmented_rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting {task}"):
        if row.get('augmentation', 'original') != 'original':
            continue
            
        if target_classes != 'all':
            match = False
            if isinstance(target_classes, dict):
                for col, allowed_vals in target_classes.items():
                    if col in df.columns:
                        val = str(row[col])
                        allowed_strs = [str(v).strip() for v in (allowed_vals if isinstance(allowed_vals, list) else [allowed_vals])]
                        if is_match(val, allowed_strs):
                            match = True
                            break
            else:
                targets = [t.strip() for t in str(target_classes).split(',')]
                row_labels = [str(row[c]) for c in label_cols if pd.notna(row[c])]
                if any(is_match(rl, targets) for rl in row_labels):
                    match = True
                    
            if not match:
                continue
                
        for _ in range(multiplier):
            chosen_method = np.random.choice(methods, p=probs)
            new_text = augmentor.apply(row['lyrics'], chosen_method)
            
            new_row = row.copy()
            new_row['lyrics'] = new_text
            new_row['augmentation'] = chosen_method
            augmented_rows.append(new_row)
            
    if augmented_rows:
        df_new = pd.DataFrame(augmented_rows)
        df_augmented = pd.concat([df, df_new], ignore_index=True)
    else:
        df_augmented = df
        
    out_path = train_path.replace('.csv', '_augmented.csv')
    df_augmented.to_csv(out_path, index=False)
    print(f"  Saved augmented {task} to {out_path} ({len(df_augmented)} total rows)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='augmentation_config.yaml', help='Path to YAML config')
    parser.add_argument('--base-dir', type=str, default='data', help='Base data directory')
    parser.add_argument('--processed', action='store_true', help='Use processed_train_df.csv instead of train_df.csv')
    parser.add_argument('--llm-model', type=str, default='llama3', help='Ollama Model name (default: llama3)')
    parser.add_argument('--tasks', type=str, help='Comma-separated list of tasks to process (e.g., task1,task2). Default is all tasks in config.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found!")
        return
        
    config = load_config(args.config)
    
    tasks_to_run = config.keys()
    if args.tasks:
        tasks_to_run = [t.strip() for t in args.tasks.split(',')]
        
    for task in tasks_to_run:
        if task in config:
            process_task(task, config[task], args.base_dir, args.processed, args.llm_model)
        else:
            print(f"Task {task} not found in config file. Skipping...")

if __name__ == '__main__':
    main()
