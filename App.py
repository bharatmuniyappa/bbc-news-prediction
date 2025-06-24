"""
app.py — Streamlit front‑end for the BBC News classifier
========================================================
Run with:
    streamlit run app.py

Requirements (install via pip):
    streamlit pandas scikit-learn joblib matplotlib

This app will look for a pre‑trained pipeline in `bbc_pipeline.joblib` (the file
created by *bbc_news_classifier.py*). If it does not find one it will train a
fresh model automatically using the default `bbc-text.csv` dataset in the same
folder. You can also supply a different labelled CSV via Settings ▸ Retrain.
"""
from __future__ import annotations

import pathlib
import tempfile
from typing import Tuple

import joblib
import pandas as pd
import streamlit as st

# The model helper defined earlier (import here or copy‑paste if packaged)
from bbc_news_classifier import BBCNewsClassifier  # type: ignore

MODEL_PATH = pathlib.Path("bbc_pipeline.joblib")
DATA_PATH = pathlib.Path("bbc-text.csv")

st.set_page_config(page_title="BBC News Classifier", page_icon="📰", layout="centered")
st.title("📰 BBC News Desk Classifier")
st.markdown(
    "Classify BBC‑style news stories into **business**, **entertainment**, **politics**, **sport**, or **tech**."
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _train_model(data_path: pathlib.Path) -> BBCNewsClassifier:
    """Train a model (or load if cached) and return it."""
    with st.spinner("Training model – this happens only once …"):
        df = pd.read_csv(data_path)
        model = BBCNewsClassifier()
        model.fit(df["text"], df["category"])
        model.save(MODEL_PATH)
        return model


@st.cache_resource(show_spinner=False)
def get_model() -> Tuple[BBCNewsClassifier, pd.DataFrame]:
    """Load existing model or train a new one; also return training metrics."""
    if MODEL_PATH.exists():
        model = BBCNewsClassifier.load(MODEL_PATH)
        metrics = None  # skip evaluation to save time; user can retrain for metrics
    else:
        model = _train_model(DATA_PATH)
        metrics = model.evaluate()
    return model, metrics


model, initial_metrics = get_model()

# ---------------------------------------------------------------------------
# Sidebar – retraining & settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    if st.checkbox("Retrain with new CSV"):
        st.markdown("Upload a **labelled** CSV with columns `text` and `category`.")
        uploaded = st.file_uploader("Choose CSV", type="csv")
        if uploaded is not None:
            tmp_path = pathlib.Path(tempfile.mkstemp(suffix=".csv")[1])
            with tmp_path.open("wb") as fh:
                fh.write(uploaded.read())
            model = _train_model(tmp_path)
            st.success("Model retrained and saved – you can start classifying!")
            st.button("Refresh page to use new model", on_click=lambda: st.experimental_rerun())

# ---------------------------------------------------------------------------
# Main interface – two tabs: single + batch
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔍 Single text", "📄 Batch CSV"])

# --- Single text input ------------------------------------------------------
with tab_single:
    sample = st.text_area(
        "Paste a news headline or article snippet:",
        height=150,
        placeholder="e.g. Government announces new budget to tackle inflation …",
    )
    if st.button("Predict category", key="single_predict") and sample.strip():
        pred = model.pipeline.predict([sample])[0]  # type: ignore[attr-defined]
        st.markdown(f"### ➡️ Predicted category: **{pred.capitalize()}**")

# --- Batch classification ---------------------------------------------------
with tab_batch:
    batch_file = st.file_uploader(
        "Upload a CSV with a `text` column to classify …", type="csv", key="batch_upl"
    )
    if batch_file is not None:
        df_batch = pd.read_csv(batch_file)
        if "text" not in df_batch.columns:
            st.error("CSV must contain a column named `text`.")
        else:
            preds = model.pipeline.predict(df_batch["text"])  # type: ignore[attr-defined]
            df_out = df_batch.copy()
            df_out["predicted_category"] = preds
            st.success(f"Classified {len(df_out)} rows.")
            st.dataframe(df_out)

            # Offer download link
            tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df_out.to_csv(tmp_csv.name, index=False)
            st.download_button(
                label="Download results as CSV",
                data=open(tmp_csv.name, "rb").read(),
                file_name="bbc_predictions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Optional metrics display (if we trained in this session)
# ---------------------------------------------------------------------------
if initial_metrics is not None:
    with st.expander("Show evaluation metrics (hold‑out + CV)"):
        st.write(f"**Hold‑out accuracy:** {initial_metrics['holdout_accuracy']:.3f}")
        st.write(f"**Cross‑val accuracy:** {initial_metrics['cross_val_accuracy']:.3f}")
        st.write("**F1 by class:**")
        st.json(initial_metrics["per_class_f1"], expanded=False)
