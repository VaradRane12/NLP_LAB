import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC


NEWSAPI_URL = "https://newsapi.org/v2/everything"
DEFAULT_CATEGORIES = ["business", "entertainment", "health", "science", "sports", "technology"]


def fetch_newsapi_articles(
    api_key: str,
    categories: List[str],
    brand_query: str = "Amazon India",
    page_size: int = 100,
    language: str = "en",
    sort_by: str = "publishedAt",
) -> pd.DataFrame:
    rows = []
    for cat in categories:
        params = {
            "q": f"{brand_query} {cat}",
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "apiKey": api_key,
        }
        resp = requests.get(NEWSAPI_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "ok":
            raise ValueError(f"NewsAPI error for category '{cat}': {payload}")

        for a in payload.get("articles", []):
            title = (a.get("title") or "").strip()
            desc = (a.get("description") or "").strip()
            content = (a.get("content") or "").strip()
            text = " ".join(x for x in [title, desc, content] if x)
            if not text:
                continue
            rows.append({"text": text, "label": cat, "source": (a.get("source") or {}).get("name", "")})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    return df


def evaluate_model(pipe: Pipeline, x_test: List[str], y_test: List[str], labels: List[str]) -> Dict:
    pred = pipe.predict(x_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    roc_auc = None
    y_bin = label_binarize(y_test, classes=labels)

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        score = pipe.predict_proba(x_test)
        if y_bin.shape[1] == np.asarray(score).shape[1]:
            roc_auc = float(roc_auc_score(y_bin, score, average="weighted", multi_class="ovr"))
    elif hasattr(clf, "decision_function"):
        score = pipe.decision_function(x_test)
        score = np.asarray(score)
        if score.ndim == 1:
            score = np.vstack([-score, score]).T
        if y_bin.shape[1] == score.shape[1]:
            roc_auc = float(roc_auc_score(y_bin, score, average="weighted", multi_class="ovr"))

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "cm": cm_df,
    }


def train_and_compare(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    x = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    labels = sorted(df["label"].unique().tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "SVM (LinearSVC)": LinearSVC(),
        "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=1500, random_state=42),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1500, random_state=42),
        "Ridge Classifier": RidgeClassifier(),
        "Voting Ensemble (Soft)": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000)),
                ("nb", MultinomialNB()),
                ("sgd", SGDClassifier(loss="log_loss", max_iter=1500, random_state=42)),
            ],
            voting="soft",
        ),
    }

    rows = []
    cms = {}

    for model_name, clf in models.items():
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, stop_words="english")),
                ("clf", clf),
            ]
        )
        pipe.fit(x_train, y_train)
        m = evaluate_model(pipe, x_test, y_test, labels)

        rows.append(
            {
                "Model": model_name,
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "ROC-AUC": None if m["roc_auc"] is None else round(m["roc_auc"], 4),
            }
        )
        cms[model_name] = m["cm"]

    result_df = pd.DataFrame(rows).sort_values(by="F1", ascending=False)
    return result_df, cms


st.set_page_config(page_title="News Topic Classification", layout="wide")
st.title("News Topic Classification")
st.write("Fetch live news from NewsAPI and compare multiple ML + ensemble classifiers.")

with st.sidebar:
    st.subheader("API Settings")
    default_key = os.getenv("NEWSAPI_KEY", "")
    api_key = st.text_input("NewsAPI Key", value=default_key, type="password")
    brand_query = st.text_input("Brand / Query", value="Amazon India")
    language = st.selectbox("Language", ["en", "hi", "fr", "de"], index=0)
    sort_by = st.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"], index=0)

    categories = st.multiselect("Categories", DEFAULT_CATEGORIES, default=["business", "sports", "technology"])
    page_size = st.slider("Articles per category", min_value=10, max_value=100, value=30, step=10)
    st.caption("NewsAPI Developer plan may reject non-localhost/browser-origin requests with corsNotAllowed.")

if "news_df" not in st.session_state:
    st.session_state.news_df = None

if st.button("Fetch News Data"):
    if not api_key:
        st.error("Enter your NewsAPI key first.")
    elif not categories:
        st.error("Select at least one category.")
    else:
        try:
            df = fetch_newsapi_articles(
                api_key=api_key,
                categories=categories,
                brand_query=brand_query,
                page_size=page_size,
                language=language,
                sort_by=sort_by,
            )
            if df.empty:
                st.error("No articles fetched. Try different query or categories.")
            else:
                st.session_state.news_df = df
                st.success(f"Fetched {len(df)} unique articles.")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                st.error("API rate limit reached (429). Lower page size or wait for quota reset.")
                st.stop()
            if "corsNotAllowed" in msg:
                st.error("NewsAPI Developer plan blocked this request (corsNotAllowed). Use localhost.")
            else:
                st.error(f"Fetch failed: {e}")

news_df = st.session_state.news_df

if news_df is not None and not news_df.empty:
    st.subheader("Fetched Dataset")
    st.dataframe(news_df.head(20), use_container_width=True)
    st.write("Class distribution")
    st.dataframe(news_df["label"].value_counts().rename_axis("label").reset_index(name="count"), use_container_width=True)

    st.subheader("Model Comparison (Text Classification + Ensemble)")
    if st.button("Train All Classification Models"):
        try:
            scores_df, cm_map = train_and_compare(news_df)
            st.dataframe(scores_df, use_container_width=True)

            model_name = st.selectbox("Confusion Matrix Model", list(cm_map.keys()))
            st.dataframe(cm_map[model_name], use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")
else:
    st.info("Fetch News Data from sidebar settings to start classification.")
