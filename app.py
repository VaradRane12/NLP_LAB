import os
import re
import gc
import sys
import json
import pickle
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import joblib
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

AutoModel = None
AutoTokenizer = None

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR = ""
except Exception:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = repr(sys.exc_info()[1])


nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="Sarcasm Detector", layout="wide")
st.title("🎭 Sarcasm Detector")
st.write("Compare deep learning models trained on the sarcasm dataset and choose one for live inference.")
st.caption(f"Python runtime: {sys.executable}")


DATA_PATH = "data/train-balanced-sarcasm.csv"
ARTIFACT_DIR = "artifacts"
MAX_SAMPLES = 2500
MAX_LEN = 60
BATCH_SIZE = 64
EPOCHS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# Use MPS on Apple Silicon when available, otherwise fall back to CUDA/CPU.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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

def get_model_artifact_paths(model_name: str) -> dict[str, str]:
    safe_name = model_name.lower().replace(" ", "_")
    return {
        "state": os.path.join(ARTIFACT_DIR, f"{safe_name}.pt"),
        "meta": os.path.join(ARTIFACT_DIR, f"{safe_name}.json"),
        "hf": os.path.join(ARTIFACT_DIR, f"{safe_name}.joblib"),
    }


def save_vocab(vocab_map: dict[str, int]) -> None:
    with open(os.path.join(ARTIFACT_DIR, "vocab.pkl"), "wb") as file_handle:
        pickle.dump(vocab_map, file_handle)


def load_vocab() -> dict[str, int] | None:
    vocab_path = os.path.join(ARTIFACT_DIR, "vocab.pkl")
    if not os.path.exists(vocab_path):
        return None
    with open(vocab_path, "rb") as file_handle:
        return pickle.load(file_handle)


def save_results(results: pd.DataFrame) -> None:
    results.to_csv(os.path.join(ARTIFACT_DIR, "model_results.csv"), index=False)


def load_results() -> pd.DataFrame | None:
    results_path = os.path.join(ARTIFACT_DIR, "model_results.csv")
    if not os.path.exists(results_path):
        return None
    return pd.read_csv(results_path)


def save_pytorch_model(model_name: str, model: nn.Module, vocab_map: dict[str, int]) -> None:
    paths = get_model_artifact_paths(model_name)
    torch.save(model.state_dict(), paths["state"])
    metadata = {
        "model_name": model_name,
        "vocab_size": len(vocab_map),
    }
    with open(paths["meta"], "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle)


def load_pytorch_model(model_name: str, vocab_size: int) -> nn.Module:
    paths = get_model_artifact_paths(model_name)
    model = PYTORCH_MODEL_SPECS[model_name]["builder"](vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(paths["state"], map_location=DEVICE))
    model.eval()
    return model


def save_hf_model(model_name: str, classifier: LogisticRegression) -> None:
    paths = get_model_artifact_paths(model_name)
    joblib.dump(classifier, paths["hf"])


def load_hf_model(model_name: str) -> LogisticRegression:
    paths = get_model_artifact_paths(model_name)
    return joblib.load(paths["hf"])


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
        # Some PyTorch transformer paths can return compressed sequence lengths.
        if encoded.size(1) != valid_tokens.size(1):
            valid_tokens = valid_tokens[:, : encoded.size(1), :]
        pooled = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


MODEL_BUILDERS = {
    "RNN": lambda vocab_size: RNNClassifier(vocab_size),
    "LSTM": lambda vocab_size: LSTMClassifier(vocab_size),
    "GRU": lambda vocab_size: GRUClassifier(vocab_size),
    "Attention LSTM": lambda vocab_size: AttentionLSTMClassifier(vocab_size),
    "Transformer Encoder": lambda vocab_size: TransformerEncoderClassifier(vocab_size),
}


def build_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], int, list[str], list[str], np.ndarray, np.ndarray]:
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

    all_tokens: list[str] = []
    for text in train_texts:
        all_tokens.extend(tokenize(text))

    vocab_counter = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in vocab_counter.items():
        if word not in vocab:
            vocab[word] = len(vocab)

    train_sequences = [encode(text, vocab) for text in train_texts]
    val_sequences = [encode(text, vocab) for text in val_texts]

    train_padded = pad_to_fixed_length(train_sequences)
    val_padded = pad_to_fixed_length(val_sequences)

    train_targets = torch.tensor(train_labels.values, dtype=torch.long)
    val_targets = torch.tensor(val_labels.values, dtype=torch.long)
    return (
        train_padded,
        train_targets,
        val_padded,
        val_targets,
        vocab,
        len(df),
        train_texts.tolist(),
        val_texts.tolist(),
        train_labels.values,
        val_labels.values,
    )


def get_hf_backbone(model_id: str):
    assert AutoModel is not None and AutoTokenizer is not None
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
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            tokens = {key: value.to(DEVICE) for key, value in tokens.items()}
            outputs = encoder(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())

    stacked = np.vstack(embeddings)

    # Release backbone memory before moving to the next model to avoid crashes on memory-constrained systems.
    del encoder
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stacked


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            total_loss += loss.item() * batch_inputs.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(batch_targets.cpu())

    logits = torch.cat(all_logits, dim=0)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    targets = torch.cat(all_targets, dim=0)
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
    predictions = torch.argmax(logits, dim=1).numpy()

    try:
        roc_auc = roc_auc_score(targets.numpy(), probabilities)
    except ValueError:
        roc_auc = 0.5

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets.numpy(), predictions),
        "precision": precision_score(targets.numpy(), predictions, zero_division=0),
        "recall": recall_score(targets.numpy(), predictions, zero_division=0),
        "f1": f1_score(targets.numpy(), predictions, zero_division=0),
        "roc_auc": roc_auc,
    }
    return metrics


def train_single_model(model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> tuple[nn.Module, dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=PYTORCH_MODEL_SPECS[model_name]["lr"])
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    metrics = evaluate_model(model, val_loader)
    return model, metrics


@st.cache_resource
def train_models(force_retrain: bool = False):
    if not force_retrain:
        loaded = load_saved_artifacts()
        if loaded is not None:
            return loaded

    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        vocab,
        sample_size,
        train_texts,
        val_texts,
        train_labels_np,
        val_labels_np,
    ) = build_dataset()

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=BATCH_SIZE)

    trained_models = {}
    result_rows = []

    progress = st.progress(0)
    status = st.empty()

    for index, (model_name, builder) in enumerate(MODEL_BUILDERS.items(), start=1):
        status.text(f"Training {model_name}...")
        model = builder(len(vocab)).to(DEVICE)
        model, metrics = train_single_model(model_name, model, train_loader, val_loader)
        trained_models[model_name] = {"family": "pytorch", "model": model}
        save_pytorch_model(model_name, model, vocab)
        result_rows.append(
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
        progress.progress(index / len(MODEL_BUILDERS))

    if TRANSFORMERS_AVAILABLE:
        hf_train_subset = min(800, len(train_texts))
        hf_val_subset = min(250, len(val_texts))
        hf_train_texts = train_texts[:hf_train_subset]
        hf_val_texts = val_texts[:hf_val_subset]
        hf_train_labels = train_labels_np[:hf_train_subset]
        hf_val_labels = val_labels_np[:hf_val_subset]

        total_steps = len(MODEL_BUILDERS) + len(HF_MODEL_IDS)

        for hf_index, (display_name, model_id) in enumerate(HF_MODEL_IDS.items(), start=1):
            status.text(f"Training {display_name} feature model...")
            train_embeddings = embed_texts_with_hf(hf_train_texts, model_id)
            val_embeddings = embed_texts_with_hf(hf_val_texts, model_id)

            classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            classifier.fit(train_embeddings, hf_train_labels)
            save_hf_model(display_name, classifier)

            predictions = classifier.predict(val_embeddings)
            probabilities = classifier.predict_proba(val_embeddings)[:, 1]
            val_loss = float(np.mean(-np.log(np.clip(np.where(hf_val_labels == 1, probabilities, 1 - probabilities), 1e-8, 1.0))))

            metrics = {
                "accuracy": accuracy_score(hf_val_labels, predictions),
                "precision": precision_score(hf_val_labels, predictions, zero_division=0),
                "recall": recall_score(hf_val_labels, predictions, zero_division=0),
                "f1": f1_score(hf_val_labels, predictions, zero_division=0),
                "roc_auc": roc_auc_score(hf_val_labels, probabilities),
            }

            trained_models[display_name] = {
                "family": "hf",
                "model_id": model_id,
                "classifier": classifier,
            }

            result_rows.append(
                {
                    "Model": display_name,
                    "Family": "Transformer",
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "ROC AUC": metrics["roc_auc"],
                    "Val Loss": val_loss,
                    "Notes": MODEL_DESCRIPTIONS[display_name],
                }
            )

            progress.progress((len(MODEL_BUILDERS) + hf_index) / total_steps)
    else:
        st.warning(
            "Transformers is not available in this runtime. "
            "Use the same Python interpreter for both install and launch. "
            f"Import error: {TRANSFORMERS_IMPORT_ERROR}"
        )

    progress.empty()
    status.empty()

    results = pd.DataFrame(result_rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
    save_vocab(vocab)
    save_results(results)
    return trained_models, results, vocab


def load_saved_artifacts():
    vocab_map = load_vocab()
    results = load_results()
    if vocab_map is None or results is None:
        return None

    trained_models = {}
    for model_name in MODEL_BUILDERS:
        paths = get_model_artifact_paths(model_name)
        if not os.path.exists(paths["state"]):
            return None
        trained_models[model_name] = {"family": "pytorch", "model": load_pytorch_model(model_name, len(vocab_map))}

    if TRANSFORMERS_AVAILABLE:
        for display_name in HF_MODEL_IDS:
            paths = get_model_artifact_paths(display_name)
            if not os.path.exists(paths["hf"]):
                return None
            trained_models[display_name] = {
                "family": "hf",
                "model_id": HF_MODEL_IDS[display_name],
                "classifier": load_hf_model(display_name),
            }

    return trained_models, results, vocab_map


trained_models: dict[str, dict] = {}
model_results = pd.DataFrame()
vocab: dict[str, int] = {}
artifact_bundle = load_saved_artifacts()

if artifact_bundle is None:
    st.error(
        "No saved model bundle found yet. Run train_models.py once from the terminal to train and save the models, then reopen the app."
    )
    st.stop()
else:
    trained_models, model_results, vocab = artifact_bundle


def vectorize_text(text: str, vocab_map: dict[str, int]) -> torch.Tensor:
    cleaned = clean_text(text)
    token_ids = encode(cleaned, vocab_map)
    padded = pad_to_fixed_length([token_ids])
    return padded


tab_live, tab_compare = st.tabs(["Live Inference", "Model Comparison"])

with tab_live:
    selected_model_name = st.selectbox("Select a model", list(trained_models.keys()))
    st.caption(MODEL_DESCRIPTIONS[selected_model_name])

    user_input = st.text_area("Enter text to analyze", placeholder="Type or paste a comment here...", height=150)

    if user_input.strip():
        selected_model = trained_models[selected_model_name]

        if selected_model["family"] == "pytorch":
            model = selected_model["model"]
            input_tensor = vectorize_text(user_input, vocab).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                prediction = int(np.argmax(probabilities))
                confidence = float(probabilities[prediction])
        else:
            model_id = selected_model["model_id"]
            classifier = selected_model["classifier"]
            embeddings = embed_texts_with_hf([clean_text(user_input)], model_id)
            probabilities = classifier.predict_proba(embeddings)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            if prediction == 1:
                st.metric("Result", "🎭 Sarcasm detected", f"{confidence*100:.1f}% confidence")
            else:
                st.metric("Result", "Not sarcasm", f"{confidence*100:.1f}% confidence")
        with right:
            st.metric("Selected Model", selected_model_name, None)

        st.markdown("### Probability breakdown")
        st.bar_chart({"Not Sarcasm": probabilities[0] * 100, "Sarcasm": probabilities[1] * 100})

        with st.expander("Cleaned input"):
            st.write(clean_text(user_input))

with tab_compare:
    st.subheader("Deep Learning Model Comparison")
    st.write(f"All models were trained on sampled rows from the sarcasm dataset. Includes PyTorch sequence models and BERT-family feature models.")

    st.dataframe(model_results, use_container_width=True, hide_index=True)

    metric_choice = st.selectbox("Choose a metric", ["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "Val Loss"])
    chart_frame = model_results[["Model", metric_choice]].set_index("Model")
    st.bar_chart(chart_frame)

    best_model = model_results.iloc[0]
    st.success(f"Best model by F1: {best_model['Model']} with F1 = {best_model['F1']:.3f}")

    with st.expander("Model details"):
        for _, row in model_results.iterrows():
            st.markdown(f"**{row['Model']}**")
            st.write(row["Notes"])