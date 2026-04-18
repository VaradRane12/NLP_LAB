import re
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent / "data" / "datasets"


def tokenize(text: str):
    return re.findall(r"[a-z]+", text.lower())


def load_corpus(max_chars=500000):
    wt = DATA_DIR / "wikitext2_train.txt"
    if wt.exists():
        return wt.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    return "machine learning natural language processing text classification model data"


def build_language_stats():
    words = tokenize(load_corpus())
    if not words:
        words = ["the", "model", "data"]
    unigram = Counter(words)
    bigram = Counter(zip(words[:-1], words[1:]))
    trigram = Counter(zip(words[:-2], words[1:-1], words[2:]))
    vocab = set(unigram.keys())
    return unigram, bigram, trigram, vocab


UNIGRAM, BIGRAM, TRIGRAM, VOCAB = build_language_stats()

TYPO_MAP = {
    "helo": "hello",
    "lernning": "learning",
    "natrual": "natural",
    "langauge": "language",
    "recieve": "receive",
    "teh": "the",
    "wrold": "world",
}

NEXT_HINTS = {
    "machine": "learning",
    "natural": "language",
    "text": "classification",
    "named": "entity",
    "support": "vector",
}

AUTOCORRECT_MODELS = [
    "none",
    "typo_map",
    "edit_distance_1",
    "hybrid",
]

FILL_MODELS = [
    "hints",
    "bigram",
    "trigram",
    "trigram_anti_repeat",
]


def edits1(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def suggest_by_edit1(word):
    w = word.lower()
    if w in VOCAB:
        return w
    cands = [c for c in edits1(w) if c in VOCAB]
    if not cands:
        return w
    return max(cands, key=lambda c: UNIGRAM[c])


def autocorrect_word(word, model_name):
    w = word.lower()
    if model_name == "none":
        return w
    if model_name == "typo_map":
        return TYPO_MAP.get(w, w)
    if model_name == "edit_distance_1":
        return suggest_by_edit1(w)
    if model_name == "hybrid":
        if w in TYPO_MAP:
            return TYPO_MAP[w]
        return suggest_by_edit1(w)
    return w


def correct_text(text, model_name):
    def repl(match):
        tok = match.group(0)
        fixed = autocorrect_word(tok, model_name)
        if tok.isupper():
            return fixed.upper()
        if tok.istitle():
            return fixed.title()
        return fixed

    return re.sub(r"[A-Za-z]+", repl, text)


def suggest_next_word(prefix_text, model_name):
    toks = tokenize(prefix_text)
    if not toks:
        return "the"

    last = toks[-1]

    if model_name == "hints":
        return NEXT_HINTS.get(last, "the")

    if last in NEXT_HINTS:
        return NEXT_HINTS[last]

    if model_name == "bigram":
        cands = [(w2, c) for (w1, w2), c in BIGRAM.items() if w1 == last]
        if not cands:
            return "the"
        cands.sort(key=lambda x: (x[1], UNIGRAM[x[0]]), reverse=True)
        return cands[0][0]

    if model_name in ("trigram", "trigram_anti_repeat"):
        recent = toks[-6:]
        scores = {}

        if len(toks) >= 2:
            w1, w2 = toks[-2], toks[-1]
            for (a, b, w3), c in TRIGRAM.items():
                if a == w1 and b == w2:
                    scores[w3] = scores.get(w3, 0.0) + (4.0 * c)

        for (w1, w2), c in BIGRAM.items():
            if w1 == last:
                scores[w2] = scores.get(w2, 0.0) + (2.0 * c)

        if not scores:
            return "the"

        if model_name == "trigram_anti_repeat":
            for w in list(scores.keys()):
                scores[w] = scores[w] - (4.0 * recent.count(w))

        ranked = sorted(scores.items(), key=lambda x: (x[1], UNIGRAM[x[0]]), reverse=True)
        if model_name == "trigram_anti_repeat":
            for w, _ in ranked:
                if recent.count(w) <= 1:
                    return w
        return ranked[0][0]

    return "the"


def benchmark_autocorrect():
    test_data = [
        ("helo", "hello"),
        ("lernning", "learning"),
        ("natrual", "natural"),
        ("langauge", "language"),
        ("teh", "the"),
        ("recieve", "receive"),
    ]

    rows = []
    for model in AUTOCORRECT_MODELS:
        correct = 0
        for wrong, target in test_data:
            pred = autocorrect_word(wrong, model)
            if pred == target:
                correct += 1
        acc = correct / len(test_data)
        rows.append({"Model": model, "Accuracy": round(acc, 4), "Correct": f"{correct}/{len(test_data)}"})
    return pd.DataFrame(rows)


def benchmark_fill():
    test_data = [
        ("machine", "learning"),
        ("natural", "language"),
        ("text", "classification"),
        ("named", "entity"),
        ("support", "vector"),
    ]

    rows = []
    for model in FILL_MODELS:
        correct = 0
        for prefix, target in test_data:
            pred = suggest_next_word(prefix, model)
            if pred == target:
                correct += 1
        acc = correct / len(test_data)
        rows.append({"Model": model, "Accuracy": round(acc, 4), "Correct": f"{correct}/{len(test_data)}"})
    return pd.DataFrame(rows)


st.set_page_config(page_title="AutoCorrect/Fill", layout="wide")
st.title("AutoCorrect / Fill")

(tab0,) = st.tabs(["AutoCorrect / Fill"])

with tab0:
    st.subheader("AutoCorrect (Multiple Models)")
    raw = st.text_area("Input text", value="helo i am lernning natrual langauge")
    selected_auto = st.multiselect(
        "Choose autocorrect models",
        AUTOCORRECT_MODELS,
        default=["hybrid", "typo_map", "edit_distance_1"],
    )

    if st.button("Run AutoCorrect"):
        if not selected_auto:
            st.warning("Select at least one autocorrect model.")
        else:
            out = []
            for m in selected_auto:
                out.append({"Model": m, "Output": correct_text(raw, m)})
            st.dataframe(pd.DataFrame(out), use_container_width=True)

    st.markdown("---")
    st.subheader("Next-Word Fill (Multiple Models)")
    prefix = st.text_input("Type prefix for next-word suggestion", value="machine")
    selected_fill = st.multiselect(
        "Choose fill models",
        FILL_MODELS,
        default=["trigram_anti_repeat", "trigram", "bigram", "hints"],
    )

    if st.button("Run Fill Models"):
        if not selected_fill:
            st.warning("Select at least one fill model.")
        else:
            out = []
            for m in selected_fill:
                out.append({"Model": m, "Suggestion": suggest_next_word(prefix, m)})
            st.dataframe(pd.DataFrame(out), use_container_width=True)

    st.markdown("---")
    st.subheader("Model Comparison")
    if st.button("Compare All Models"):
        ac_df = benchmark_autocorrect()
        fill_df = benchmark_fill()
        st.markdown("Autocorrect Model Comparison")
        st.dataframe(ac_df, use_container_width=True)
        st.markdown("Fill Model Comparison")
        st.dataframe(fill_df, use_container_width=True)
