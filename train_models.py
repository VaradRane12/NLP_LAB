import gc
import json
import os
import pickle
import re
import sys
from collections import Counter

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


nltk.download("stopwords", quiet=True)

DATA_PATH = "data/train-balanced-sarcasm.csv"
ARTIFACT_DIR = "artifacts"
MAX_SAMPLES = 5000
MAX_LEN = 60
BATCH_SIZE = 64
EPOCHS = 100
TRAIN_DEVICE = os.environ.get("TRAIN_DEVICE", "cpu").lower()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

if TRAIN_DEVICE == "mps" and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif TRAIN_DEVICE == "cuda" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Training device: {DEVICE}")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return text.split()


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(word, 1) for word in tokenize(text)]


def pad_to_fixed_length(sequences: list[list[int]], max_len: int = MAX_LEN) -> torch.Tensor:
    tensor_sequences = [torch.tensor(sequence[:max_len], dtype=torch.long) for sequence in sequences]
    if not tensor_sequences:
        return torch.empty(0, max_len, dtype=torch.long)
    padded = pad_sequence(tensor_sequences, batch_first=True, padding_value=0)
    if padded.size(1) < max_len:
        padding = torch.zeros((padded.size(0), max_len - padded.size(1)), dtype=torch.long)
        padded = torch.cat([padded, padding], dim=1)
    return padded[:, :max_len]


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lengths = inputs.ne(0).sum(dim=1).cpu().clamp(min=1)
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.fc(hidden[-1])


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lengths = inputs.ne(0).sum(dim=1).cpu().clamp(min=1)
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lengths = inputs.ne(0).sum(dim=1).cpu().clamp(min=1)
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.fc(hidden[-1])


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lengths = inputs.ne(0).sum(dim=1).cpu().clamp(min=1)
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=inputs.size(1))
        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(inputs.eq(0), float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(outputs * weights, dim=1)
        return self.fc(context)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = MAX_LEN):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pe[:, : inputs.size(1)]


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        embedded = self.positional_encoding(embedded)
        padding_mask = inputs.eq(0)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        valid_tokens = (~padding_mask).unsqueeze(-1).float()
        if encoded.size(1) != valid_tokens.size(1):
            valid_tokens = valid_tokens[:, : encoded.size(1), :]
        pooled = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


MODEL_DESCRIPTIONS = {
    "RNN": "Vanilla recurrent network that learns sequence order but can struggle with long dependencies.",
    "LSTM": "Classic memory-based recurrent model, matching the notebook's core architecture.",
    "GRU": "A lighter gated recurrent model that often trains faster than LSTM.",
    "Attention LSTM": "LSTM encoder plus attention over hidden states to focus on important tokens.",
    "Transformer Encoder": "A small transformer built with self-attention and positional encoding.",
    "BERT": "Transformer encoder features from bert-base-uncased with a logistic classification head.",
    "RoBERTa": "Transformer encoder features from roberta-base with a logistic classification head.",
    "DistilBERT": "Transformer encoder features from distilbert-base-uncased with a logistic classification head.",
}

HF_MODEL_IDS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
}

PYTORCH_MODEL_SPECS = {
    "RNN": {"builder": lambda vocab_size: RNNClassifier(vocab_size), "lr": 0.001},
    "LSTM": {"builder": lambda vocab_size: LSTMClassifier(vocab_size), "lr": 0.001},
    "GRU": {"builder": lambda vocab_size: GRUClassifier(vocab_size), "lr": 0.001},
    "Attention LSTM": {"builder": lambda vocab_size: AttentionLSTMClassifier(vocab_size), "lr": 0.001},
    "Transformer Encoder": {"builder": lambda vocab_size: TransformerEncoderClassifier(vocab_size), "lr": 0.0008},
}


def artifact_paths(model_name: str) -> dict[str, str]:
    safe_name = model_name.lower().replace(" ", "_")
    return {
        "state": os.path.join(ARTIFACT_DIR, f"{safe_name}.pt"),
        "hf": os.path.join(ARTIFACT_DIR, f"{safe_name}.joblib"),
    }


def build_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"comment": "text", "label": "label"})
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tokens = []
    for text in train_texts:
        tokens.extend(tokenize(text))

    vocab_counter = Counter(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word in vocab_counter:
        if word not in vocab:
            vocab[word] = len(vocab)

    train_sequences = [encode(text, vocab) for text in train_texts]
    val_sequences = [encode(text, vocab) for text in val_texts]

    train_padded = pad_to_fixed_length(train_sequences)
    val_padded = pad_to_fixed_length(val_sequences)
    train_targets = torch.tensor(train_labels.values, dtype=torch.long)
    val_targets = torch.tensor(val_labels.values, dtype=torch.long)
    return train_padded, train_targets, val_padded, val_targets, vocab, train_texts.tolist(), val_texts.tolist(), train_labels.values, val_labels.values


def save_vocab(vocab: dict[str, int]) -> None:
    with open(os.path.join(ARTIFACT_DIR, "vocab.pkl"), "wb") as file_handle:
        pickle.dump(vocab, file_handle)


def save_results(results: pd.DataFrame) -> None:
    results.to_csv(os.path.join(ARTIFACT_DIR, "model_results.csv"), index=False)


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    logits_list = []
    targets_list = []
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            total_loss += loss.item() * batch_inputs.size(0)
            logits_list.append(logits.cpu())
            targets_list.append(batch_targets.cpu())
    logits = torch.cat(logits_list, dim=0)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    targets = torch.cat(targets_list, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
    predictions = torch.argmax(logits, dim=1).numpy()
    try:
        roc_auc = roc_auc_score(targets, probabilities)
    except ValueError:
        roc_auc = 0.5
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc,
    }


def train_pytorch_model(model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=PYTORCH_MODEL_SPECS[model_name]["lr"])
    criterion = nn.CrossEntropyLoss()
    print(f"Training {model_name} on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1}/{EPOCHS} loss={total_loss:.4f}")

    metrics = evaluate_model(model, val_loader)
    return metrics


def get_hf_backbone(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id).to(DEVICE)
    encoder.eval()
    return tokenizer, encoder


def embed_texts_with_hf(texts: list[str], model_id: str, batch_size: int = 24, max_len: int = 96) -> np.ndarray:
    tokenizer, encoder = get_hf_backbone(model_id)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            tokens = {key: value.to(DEVICE) for key, value in tokens.items()}
            outputs = encoder(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
    del encoder
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(embeddings)


def main() -> None:
    train_inputs, train_targets, val_inputs, val_targets, vocab, train_texts, val_texts, train_labels_np, val_labels_np = build_dataset()

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=BATCH_SIZE)

    results_rows = []

    for model_name in PYTORCH_MODEL_SPECS:
        model = PYTORCH_MODEL_SPECS[model_name]["builder"](len(vocab)).to(DEVICE)
        metrics = train_pytorch_model(model_name, model, train_loader, val_loader)
        torch.save(model.state_dict(), artifact_paths(model_name)["state"])
        results_rows.append(
            {
                "Model": model_name,
                "Family": "PyTorch",
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC AUC": metrics["roc_auc"],
                "Val Loss": metrics["loss"],
                "Notes": MODEL_DESCRIPTIONS[model_name],
            }
        )

    if TRANSFORMERS_AVAILABLE:
        hf_train_subset = min(800, len(train_texts))
        hf_val_subset = min(250, len(val_texts))
        hf_train_texts = train_texts[:hf_train_subset]
        hf_val_texts = val_texts[:hf_val_subset]
        hf_train_labels = train_labels_np[:hf_train_subset]
        hf_val_labels = val_labels_np[:hf_val_subset]

        for display_name, model_id in HF_MODEL_IDS.items():
            print(f"Training {display_name} feature classifier...")
            train_embeddings = embed_texts_with_hf(hf_train_texts, model_id)
            val_embeddings = embed_texts_with_hf(hf_val_texts, model_id)
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            classifier.fit(train_embeddings, hf_train_labels)
            joblib.dump(classifier, artifact_paths(display_name)["hf"])
            probabilities = classifier.predict_proba(val_embeddings)[:, 1]
            predictions = classifier.predict(val_embeddings)
            val_loss = float(np.mean(-np.log(np.clip(np.where(hf_val_labels == 1, probabilities, 1 - probabilities), 1e-8, 1.0))))
            try:
                roc_auc = roc_auc_score(hf_val_labels, probabilities)
            except ValueError:
                roc_auc = 0.5
            results_rows.append(
                {
                    "Model": display_name,
                    "Family": "Transformer",
                    "Accuracy": accuracy_score(hf_val_labels, predictions),
                    "Precision": precision_score(hf_val_labels, predictions, zero_division=0),
                    "Recall": recall_score(hf_val_labels, predictions, zero_division=0),
                    "F1": f1_score(hf_val_labels, predictions, zero_division=0),
                    "ROC AUC": roc_auc,
                    "Val Loss": val_loss,
                    "Notes": MODEL_DESCRIPTIONS[display_name],
                }
            )
    else:
        print("Transformers not available; skipping BERT, RoBERTa, and DistilBERT.")

    results = pd.DataFrame(results_rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
    save_vocab(vocab)
    save_results(results)
    print("Saved all artifacts to", ARTIFACT_DIR)
    print(results)


if __name__ == "__main__":
    main()