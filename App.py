"""
App.py — bullet-proof Streamlit front-end for the BBC News classifier
=====================================================================
First run:  trains a fresh model from `bbc-text.csv`
Later runs: loads the saved `bbc_pipeline.joblib`
If the saved file is broken or version-mismatched, it retrains automatically.
"""
from __future__ import annotations

import pathlib, warnings, tempfile, os
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError, InconsistentVersionWarning

from bbc_news_classifier import BBCNewsClassifier   # ← your helper class

MODEL_PATH = pathlib.Path("bbc_pipeline.joblib")
DATA_PATH  = pathlib.Path("bbc-text.csv")

st.set_page_config(page_title="BBC News Classifier", page_icon="📰")
st.title("📰 BBC News Desk Classifier")

# ------------------------------------------------------------------ #
# safe loader: train if missing / incompatible                        #
# ------------------------------------------------------------------ #
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def load_or_train() -> BBCNewsClassifier:
    if MODEL_PATH.exists():
        try:
            model = BBCNewsClassifier.load(MODEL_PATH)
            # verify TF-IDF is fitted
            _ = model.pipeline.named_steps["vectoriser"].idf_
            st.success("Loaded pre-trained model ✅")
            return model
        except (NotFittedError, Exception):
            st.warning("Saved model incompatible – retraining…")

    if not DATA_PATH.exists():
        st.error("bbc-text.csv not found – cannot train.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    model = BBCNewsClassifier()
    with st.spinner("Training model (first run only)…"):
        model.fit(df["text"], df["category"])
    model.save(MODEL_PATH)
    st.success("New model trained and saved ✅")
    return model

model = load_or_train()

# ------------------------------------------------------------------ #
# UI – single text & batch CSV                                        #
# ------------------------------------------------------------------ #
tab_single, tab_batch = st.tabs(["🔍 Single text", "📄 Batch CSV"])

with tab_single:
    text = st.text_area(
        "Paste a news headline or snippet:",
        placeholder="e.g. Google unveils new AI model with multimodal powers…",
        height=150,
    )
    if st.button("Predict") and text.strip():
        pred = model.pipeline.predict([text])[0]
        st.markdown(f"### ➡️ **{pred.capitalize()}**")

with tab_batch:
    csv = st.file_uploader("Upload CSV with a 'text' column", type="csv")
    if csv:
        df = pd.read_csv(csv)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            preds = model.pipeline.predict(df["text"])
            df["predicted_category"] = preds
            st.dataframe(df)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            st.download_button("Download predictions", tmp.read(),
                               file_name="bbc_predictions.csv")
