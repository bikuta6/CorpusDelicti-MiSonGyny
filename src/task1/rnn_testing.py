"""
Comparativa de modelos RNN para Task 1: Clasificación Binaria de Misoginia.

Arquitecturas evaluadas:
  - LSTM
  - GRU
  - BiLSTM  (LSTM bidireccional)
  - BiGRU   (GRU bidireccional)

Embeddings: vectores preentrenados en formato texto (GloVe / fastText .vec)
            o binario de fastText (requiere `fasttext` instalado).

Uso:
    python rnn_testing.py --embeddings /ruta/cc.es.300.vec

    Opciones adicionales:
    --data_path        Ruta al CSV de entrenamiento (default: ../../data/task1/processed_train.csv)
    --test_path        Ruta al CSV de test         (default: ../../data/task1/test.csv)
    --test_labels_path Ruta a las etiquetas de test (default: ../../data/task1/test_labels.csv)
    --results_file     Fichero CSV de resultados    (default: ../../results/task1/tabla_rnn.csv)
    --embedding_dim    Dimensión de los embeddings  (default: 300)
    --hidden_size      Tamaño de las capas ocultas  (default: 256)
    --num_layers       Número de capas RNN          (default: 2)
    --dropout          Dropout entre capas          (default: 0.3)
    --batch_size       Tamaño de batch              (default: 32)
    --epochs           Épocas máximas               (default: 30)
    --patience         Paciencia early-stopping     (default: 5)
    --lr               Tasa de aprendizaje          (default: 1e-3)
    --max_len          Longitud máxima de secuencia (default: 512)
    --min_freq         Frecuencia mínima de palabra (default: 2)
    --val_size         Fracción de validación       (default: 0.2)
    --pooling          Estrategia de pooling        (default: max_mean)
                       Opciones: max | mean | max_mean | attention
    --label_smoothing  Label smoothing en la pérdida (default: 0.1)
    --seed             Semilla de reproducibilidad  (default: 42)
"""

import os
import sys
import argparse
import gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Importar utilidades del proyecto ──────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_seed, DEFAULT_SEED
from trainer import FocalLoss

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES POR DEFECTO
# ══════════════════════════════════════════════════════════════════════════════
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARIO Y TOKENIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def simple_tokenize(text: str) -> list[str]:
    """Tokenización básica: minúsculas + split por espacios/puntuación."""
    import re
    text = text.lower()
    # Separar puntuación y mantener palabras (incluye acentos y ñ)
    tokens = re.findall(r"[a-záéíóúüñ']+", text)
    return tokens


class Vocabulary:
    """Vocabulario construido a partir del corpus de entrenamiento."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._pad_idx = 0
        self._unk_idx = 1

    def build(self, texts: list[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"  Vocabulario: {len(self.word2idx):,} tokens "
              f"(min_freq={self.min_freq})")

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = simple_tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, self._unk_idx) for t in tokens]
        # Padding
        ids += [self._pad_idx] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self._pad_idx


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def load_embeddings_txt(path: str, vocab: Vocabulary, emb_dim: int) -> torch.Tensor:
    """
    Carga embeddings en formato texto (GloVe, fastText .vec, word2vec .txt).
    Primera línea puede ser la cabecera '<n_words> <dim>' (se ignora).

    Devuelve una tensor de forma (vocab_size, emb_dim).
    """
    print(f"  Cargando embeddings desde '{path}' …")
    embedding_matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)

    # Inicializar UNK con distribución uniforme pequeña
    embedding_matrix[vocab.word2idx[UNK_TOKEN]] = np.random.uniform(
        -0.05, 0.05, emb_dim
    )

    n_found = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            # Saltar cabecera (<n_words> <dim>) si la hay
            if len(parts) == 2:
                continue
            word = parts[0]
            if word not in vocab.word2idx:
                continue
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) != emb_dim:
                    continue
                embedding_matrix[vocab.word2idx[word]] = vec
                n_found += 1
            except ValueError:
                continue

    coverage = n_found / max(len(vocab) - 2, 1) * 100
    print(f"  Embeddings cargados: {n_found:,}/{len(vocab)-2:,} tokens "
          f"({coverage:.1f}% de cobertura)")
    return torch.tensor(embedding_matrix)


def load_embeddings_fasttext_bin(
    path: str, vocab: Vocabulary, emb_dim: int
) -> torch.Tensor:
    """
    Carga embeddings de un modelo fastText binario (.bin) usando la librería `fasttext`.
    """
    try:
        import fasttext  # type: ignore
    except ImportError:
        raise ImportError(
            "Instala 'fasttext' para usar modelos binarios: pip install fasttext"
        )

    print(f"  Cargando modelo fastText binario desde '{path}' …")
    ft_model = fasttext.load_model(path)
    embedding_matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    embedding_matrix[vocab.word2idx[UNK_TOKEN]] = np.random.uniform(
        -0.05, 0.05, emb_dim
    )
    n_found = 0
    for word, idx in vocab.word2idx.items():
        if word in (PAD_TOKEN, UNK_TOKEN):
            continue
        vec = ft_model.get_word_vector(word)
        if len(vec) == emb_dim:
            embedding_matrix[idx] = vec
            n_found += 1

    coverage = n_found / max(len(vocab) - 2, 1) * 100
    print(f"  Embeddings cargados: {n_found:,}/{len(vocab)-2:,} tokens "
          f"({coverage:.1f}% de cobertura)")
    return torch.tensor(embedding_matrix)


def load_embeddings(path: str, vocab: Vocabulary, emb_dim: int) -> torch.Tensor:
    """Detecta automáticamente el formato y carga los embeddings."""
    if path.endswith(".bin"):
        return load_embeddings_fasttext_bin(path, vocab, emb_dim)
    else:
        return load_embeddings_txt(path, vocab, emb_dim)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PYTORCH
# ══════════════════════════════════════════════════════════════════════════════

class LyricsDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_len: int,
    ):
        self.encoded = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.encoded[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ARQUITECTURAS RNN
# ══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """Atención aditiva simple sobre los pasos temporales de la RNN."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: (batch, seq_len, hidden)
        scores = self.attn(rnn_out).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (rnn_out * weights).sum(dim=1)             # (batch, hidden)


class RNNClassifier(nn.Module):
    """
    Clasificador binario genérico basado en LSTM / GRU (opcionalmente bidireccional).

    Arquitectura:
        Embedding (preentrenado, congelado opcionalmente)
        → RNN (LSTM o GRU, N capas, bidireccional o no)
        → Pooling temporal  (max | mean | max_mean | attention)
        → LayerNorm + Dropout + Linear oculta + Linear → logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str,          # "LSTM" | "GRU"
        bidirectional: bool,
        pad_idx: int,
        pretrained_embeddings: torch.Tensor,
        freeze_embeddings: bool = False,
        pooling: str = "max_mean",   # max | mean | max_mean | attention
        num_classes: int = 2,
    ):
        super().__init__()
        self.pooling = pooling.lower()

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings,
            padding_idx=pad_idx,
            freeze=freeze_embeddings,
        )

        rnn_cls = nn.LSTM if cell_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_size = hidden_size * (2 if bidirectional else 1)

        if self.pooling == "max_mean":
            pooled_size = rnn_out_size * 2
        else:
            pooled_size = rnn_out_size

        if self.pooling == "attention":
            self.attn_pool = AttentionPooling(rnn_out_size)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(pooled_size)
        self.classifier = nn.Linear(pooled_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        emb = self.dropout(self.embedding(input_ids))  # (batch, seq_len, emb_dim)
        out, _ = self.rnn(emb)                         # (batch, seq_len, hidden*dirs)

        if self.pooling == "max":
            pooled, _ = out.max(dim=1)
        elif self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "max_mean":
            max_pool, _ = out.max(dim=1)
            mean_pool   = out.mean(dim=1)
            pooled = torch.cat([max_pool, mean_pool], dim=1)
        elif self.pooling == "attention":
            pooled = self.attn_pool(out)
        else:
            raise ValueError(f"Pooling desconocido: '{self.pooling}'")

        x = self.norm(pooled)
        logits = self.classifier(self.dropout(x))
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO Y EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0.0
    )
    acc = accuracy_score(all_labels, all_preds)
    return {
        "loss": total_loss / len(loader),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: argparse.Namespace,
    device: torch.device,
    model_name: str,
    checkpoint_path: Path,
) -> dict:
    """
    Entrena el modelo con early-stopping, guarda el mejor checkpoint en disco
    y devuelve las métricas del mejor epoch.
    """
    criterion = nn.CrossEntropyLoss(
        weight=cfg.class_weights.to(device),
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    # Warmup lineal las primeras 2 épocas, luego ReduceLROnPlateau
    warmup_epochs = 2
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_f1 = -1.0
    best_metrics = {}
    patience_counter = 0

    print(f"\n{'─'*60}")
    print(f"  Entrenando: {model_name}")
    print(f"{'─'*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_metrics["f1_macro"])

        print(
            f"  Epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_metrics = {**val_metrics}
            patience_counter = 0
            # Guardar checkpoint en disco
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint guardado → {checkpoint_path.name} "
                  f"(val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  ⏹  Early stopping en epoch {epoch} "
                      f"(sin mejora en {cfg.patience} epochs)")
                break

    # Restaurar mejor modelo desde disco
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"  ✓  Mejor val F1-macro: {best_val_f1:.4f} "
          f"(cargado desde {checkpoint_path.name})")
    return best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test de modelos LSTM/GRU con embeddings preentrenados"
    )
    parser.add_argument(
        "--embeddings", type=str, required=False,
        default="../../models/word_embs/cc.es.300.vec",
        help="Ruta al fichero de embeddings preentrenados (.vec/.txt/.bin)"
    )
    parser.add_argument("--data_path", type=str,
                        default="../../data/task1/processed_train.csv")
    parser.add_argument("--test_path", type=str,
                        default="../../data/task1/test.csv")
    parser.add_argument("--test_labels_path", type=str,
                        default="../../data/task1/test_labels.csv")
    parser.add_argument("--results_file", type=str,
                        default="../../results/task1/tabla_rnn.csv")
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_size",   type=int, default=256)
    parser.add_argument("--num_layers",    type=int, default=2)
    parser.add_argument("--dropout",       type=float, default=0.5)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--epochs",        type=int, default=30)
    parser.add_argument("--patience",      type=int, default=5)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--max_len",       type=int, default=512)
    parser.add_argument("--min_freq",      type=int, default=2)
    parser.add_argument("--val_size",      type=float, default=0.2)
    parser.add_argument("--freeze_emb",    action="store_true",
                        help="Congelar embeddings durante el entrenamiento")
    parser.add_argument("--pooling",       type=str, default="max_mean",
                        choices=["max", "mean", "max_mean", "attention"],
                        help="Estrategia de pooling temporal")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing en CrossEntropyLoss (0 = sin smoothing)")
    parser.add_argument("--seed",          type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nDispositivo: {device}")

    # ── 1. Cargar datos ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  CARGA DE DATOS")
    print(f"{'═'*60}")

    train_df = pd.read_csv(cfg.data_path)
    train_df["label"] = train_df["label"].map({"NM": 0, "M": 1})
    print(f"  Dataset train: {len(train_df):,} canciones")
    print(f"  Distribución:\n{train_df['label'].value_counts().to_string()}")

    test_df = pd.read_csv(cfg.test_path)
    test_labels_df = pd.read_csv(cfg.test_labels_path)
    test_df = test_df.merge(test_labels_df, on="id")
    test_df["label"] = test_df["label"].map({"NM": 0, "M": 1})
    print(f"  Dataset test:  {len(test_df):,} canciones")

    # Split train / val
    train_split, val_split = train_test_split(
        train_df,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=train_df["label"],
    )

    # ── 2. Vocabulario ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  VOCABULARIO")
    print(f"{'═'*60}")
    vocab = Vocabulary(min_freq=cfg.min_freq)
    vocab.build(train_split["lyrics"].tolist())

    # ── 3. Embeddings ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  EMBEDDINGS PREENTRENADOS")
    print(f"{'═'*60}")
    embedding_matrix = load_embeddings(cfg.embeddings, vocab, cfg.embedding_dim)

    # ── 4. DataLoaders ─────────────────────────────────────────────────────────
    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = LyricsDataset(
            texts=df["lyrics"].tolist(),
            labels=df["label"].tolist(),
            vocab=vocab,
            max_len=cfg.max_len,
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          pin_memory=(device.type == "cuda"))

    train_loader = make_loader(train_split, shuffle=True)
    val_loader   = make_loader(val_split,   shuffle=False)
    test_loader  = make_loader(test_df,     shuffle=False)

    # Pesos de clase para pérdida ponderada
    n_pos  = int(train_split["label"].sum())
    n_neg  = len(train_split) - n_pos
    total  = n_pos + n_neg
    w0 = total / (2 * n_neg)
    w1 = total / (2 * n_pos)
    cfg.class_weights = torch.tensor([w0, w1], dtype=torch.float)
    print(f"\n  Pesos de clase → NM: {w0:.3f}  M: {w1:.3f}")

    # ── 5. Configuraciones de modelos ──────────────────────────────────────────
    model_configs = [
        {"name": "LSTM",    "cell_type": "LSTM", "bidirectional": False},
        {"name": "BiLSTM",  "cell_type": "LSTM", "bidirectional": True},
        {"name": "GRU",     "cell_type": "GRU",  "bidirectional": False},
        {"name": "BiGRU",   "cell_type": "GRU",  "bidirectional": True},
    ]

    # ── 6. Entrenamiento y evaluación ──────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ENTRENAMIENTO")
    print(f"{'═'*60}")

    # Directorio para checkpoints
    ckpt_dir = Path(cfg.results_file).parent / "rnn_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    criterion_eval = nn.CrossEntropyLoss(
        weight=cfg.class_weights.to(device),
        label_smoothing=cfg.label_smoothing,
    )

    for mcfg in model_configs:
        set_seed(cfg.seed)  # Reproducibilidad por modelo

        model = RNNClassifier(
            vocab_size=len(vocab),
            embedding_dim=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            cell_type=mcfg["cell_type"],
            bidirectional=mcfg["bidirectional"],
            pad_idx=vocab.pad_idx,
            pretrained_embeddings=embedding_matrix,
            freeze_embeddings=cfg.freeze_emb,
            pooling=cfg.pooling,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Modelo: {mcfg['name']}  |  Parámetros entrenables: {n_params:,}")

        # Ruta del checkpoint para este modelo
        ckpt_path = ckpt_dir / f"{mcfg['name']}_best.pt"

        # Entrenamiento
        best_val = train_model(
            model, train_loader, val_loader, cfg, device, mcfg["name"], ckpt_path
        )

        # Evaluación en test
        test_metrics = evaluate(model, test_loader, criterion_eval, device)
        print(
            f"  TEST → F1-macro={test_metrics['f1_macro']:.4f} | "
            f"Acc={test_metrics['accuracy']:.4f} | "
            f"Prec={test_metrics['precision']:.4f} | "
            f"Rec={test_metrics['recall']:.4f}"
        )

        results.append({
            "Modelo": mcfg["name"],
            # Validación
            "Val F1-macro":  best_val["f1_macro"],
            "Val Accuracy":  best_val["accuracy"],
            "Val Precision": best_val["precision"],
            "Val Recall":    best_val["recall"],
            # Test
            "Test F1-macro":  test_metrics["f1_macro"],
            "Test Accuracy":  test_metrics["accuracy"],
            "Test Precision": test_metrics["precision"],
            "Test Recall":    test_metrics["recall"],
        })

        # Liberar memoria
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 7. Tabla de resultados ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  RESULTADOS FINALES")
    print(f"{'═'*60}")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Guardar CSV
    results_path = Path(cfg.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n  Resultados guardados en '{results_path}'")

    # Destacar el mejor modelo por F1-macro en test
    best_row = results_df.loc[results_df["Test F1-macro"].idxmax()]
    print(
        f"\n  Mejor modelo: {best_row['Modelo']}  "
        f"(Test F1-macro = {best_row['Test F1-macro']:.4f})"
    )


if __name__ == "__main__":
    main()
