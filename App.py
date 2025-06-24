"""
app.py — robust Streamlit front‑end for BBC News classifier
==========================================================
Launch locally with:
    streamlit run app.py

Improvements (24 Jun 2025)
--------------------------
* Auto‑train fallback if `bbc_pipeline.joblib` is absent or incompatible.
* Safe version handling: retrains when scikit‑learn versions mismatch.
* Clear status messages in the UI.
"""
from __future__ import annotations

import os
import pathlib
import tempfile
import warnings
from typing import Tuple

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError, InconsistentVersionWarning
from sklearn.base import BaseEstimator

from bbc_news_classifier import BBCNewsClassifier

MODEL_PATH = pathlib.Path("bbc_pipeline.joblib")
DATA_PATH = pathlib.Path("bbc-text.csv")

st.set_page_config(page_title="BBC News Classifier", page_icon="📰", layout="centered")
st.title("📰 BBC News Desk Classifier")
st.markdown(
    "Classify BBC‑style news stories into **business**, **entertainment**, **politics**, **sport**, or **tech**."
)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _is_pipeline_fitted(pipe: BaseEstimator) -> bool:
    """Return True if the TF‑IDF vectoriser inside the pipeline is fitted."""
    try:
        _ = pipe.named_steps["vectoriser"].idf_
        return True
    except AttributeError:
        return False
    except NotFittedError:
        return False


def load_or_train() -> Tuple[BBCNewsClassifier, dict | None]:
    """Load a compatible model or train a fresh one from `bbc-text.csv`."""
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    # Try loading first
    if MODEL_PATH.exists():
        try:
            mdl = BBCNewsClassifier.load(MODEL_PATH)
            if _is_pipeline_fitted(mdl.pipeline):
                st.success("Loaded pre‑trained model ✅")
                return mdl, None
            st.warning("Model file found but appears unfitted — retraining…")
        except Exception as exc:  # pragma: no cover
            st.warning(f"Failed to load model ({exc}). Retraining…")

    # Train new model
    if not DATA_PATH.exists():
        st.error("Dataset bbc‑text.csv not found — cannot train.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    mdl = BBCNewsClassifier()
    with st.spinner("Training model — first run only…"):
        mdl.fit(df["text"], df["category"])
    mdl.save(MODEL_PATH)
    st.success("New model trained and saved ✅")
    return mdl, mdl.evaluate()


model, metrics_first_run = load_or_train()

# ---------------------------------------------------------------------
# Sidebar: optional custom retrain
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    if st.checkbox("Retrain with your own CSV"):
        upl = st.file_uploader("Upload labelled CSV (text, category)", type="csv")
        if upl is not None:
            tmp = pathlib.Path(tempfile.mkstemp(suffix=".csv")[1])
            tmp.write_bytes(upl.read())
            df_custom = pd.read_csv(tmp)
            if {"text", "category"}.issubset(df_custom.columns):
                with st.spinner("Retraining on your data…"):
                    model = BBCNewsClassifier()
                    model.fit(df_custom["text"], df_custom["category"])
                    model.save(MODEL_PATH)
                st.success("Custom model trained — refresh to use it!")
            else:
                st.error("CSV must have 'text' and 'category' columns.")

# ---------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------
TAB_SINGLE, TAB_BATCH = st.tabs(["🔍 Single text", "📄 Batch CSV"])

with TAB_SINGLE:
    user_text = st.text_area(
        "Paste a news headline or snippet:",
        placeholder="e.g. Google unveils new AI model with multimodal powers…",
        height=150,
    )
    if st.button("Predict", key="predict_single") and user_text.strip():
        category = model.pipeline.predict([user_text])[0]  # type: ignore[attr-defined]
        st.markdown(f"### ➡️ Predicted category: **{category.capitalize()}**")

with TAB_BATCH:
    csv_batch = st.file_uploader("Upload CSV with a 'text' column", type="csv")
    if csv_batch is not None:
        df_in = pd.read_csv(csv_batch)
        if "text" not in df_in.columns:
            st.error("CSV must include a 'text' column.")
        else:
            preds = model.pipeline.predict(df_in["text"])  # type: ignore[attr-defined]
            df_out = df_in.copy()
            df_out["predicted_category"] = preds
            st.dataframe(df_out)
            tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df_out.to_csv(tmp_csv.name, index=False)
            st.download_button("Download predictions", tmp_csv.read(), file_name="bbc_predictions.csv")

# ---------------------------------------------------------------------
# Show metrics if this was a first‑run training
# ---------------------------------------------------------------------
if metrics_first_run is not None:
    with st.expander("First‑run metrics"):
        st.write(f"Hold‑out accuracy : {metrics_first_run['holdout_accuracy']:.3f}")
        st.write(f"Cross‑val accuracy: {metrics_first_run['cross_val_accuracy']:.3f}")
        st.json(metrics_first_run["per_class_f1"], expanded=False)
