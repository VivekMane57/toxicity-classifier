# dashboard/app.py
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Toxicity Dashboard", layout="wide")
st.title("🧹 YouTube Toxic Comments – Dashboard")

# -------------------------------------------------------------------
# Paths & artifacts
# -------------------------------------------------------------------
MODELS_DIR = Path("models")
VEC_PATH = MODELS_DIR / "tfidf.joblib"
MDL_PATH = MODELS_DIR / "logreg_toxic.joblib"
DEFAULT_DATA = Path("data/raw/train.csv")

FIG_CM = Path("reports/figures/confusion_matrix.png")
FIG_WC = Path("reports/figures/wordcloud.png")

# -------------------------------------------------------------------
# Load model (stop the app early if not trained)
# -------------------------------------------------------------------
if not VEC_PATH.exists() or not MDL_PATH.exists():
    st.warning("Model files not found. Train baseline first:\n\n`python -m src.models_baseline`")
    st.stop()

vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MDL_PATH)

# -------------------------------------------------------------------
# Sidebar: single comment prediction
# -------------------------------------------------------------------
st.sidebar.header("🔎 Predict a Comment")
threshold = st.sidebar.slider(
    "Decision threshold", 0.10, 0.90, 0.50, 0.01, help="Score ≥ threshold ⇒ Toxic"
)
user_text = st.sidebar.text_area("Enter a comment", height=120)

if st.sidebar.button("Classify"):
    if user_text.strip():
        X = vectorizer.transform([user_text])
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= threshold)
        st.sidebar.write(f"**Prediction:** {'🛑 Toxic' if pred == 1 else '✅ Non-toxic'}")
        st.sidebar.write(f"**Toxicity Probability:** {prob:.3f}")
    else:
        st.sidebar.info("Enter some text to classify.")

# -------------------------------------------------------------------
# Dataset Explorer (upload or auto-load)
# -------------------------------------------------------------------
st.header("📊 Dataset Explorer")

left, right = st.columns([3, 2])

with left:
    uploaded = st.file_uploader(
        "Upload a CSV with columns: `comment_text`, `toxic` (optional for ground-truth rate)",
        type=["csv"],
        help="If you don't upload, the app will try to preview data/raw/train.csv",
        key="preview_uploader",
    )

    df = None
    src_label = ""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        src_label = "📥 Uploaded file"
    elif DEFAULT_DATA.exists():
        try:
            df = pd.read_csv(DEFAULT_DATA)
            src_label = f"📂 {DEFAULT_DATA.as_posix()}"
        except Exception as e:
            st.info(f"Could not read default dataset: {e}")

    if df is not None:
        st.caption(src_label)
        st.dataframe(df.head(50), use_container_width=True)
        st.write("Rows:", len(df))

        # Safe toxic-rate (handles datasets where -1 means 'unlabeled')
        if "toxic" in df.columns:
            s = pd.to_numeric(df["toxic"], errors="coerce").fillna(0)
            s = s.clip(lower=0)  # convert -1 to 0
            toxic_rate = float(s.mean())
            st.metric("Toxic Rate (ground truth)", f"{toxic_rate*100:.2f}%")
    else:
        st.info("No data to preview. Upload a CSV or place one at `data/raw/train.csv`.")

with right:
    st.subheader("📈 Visualizations")
    if FIG_CM.exists():
        st.image(str(FIG_CM), caption="Confusion Matrix", use_container_width=True)
    else:
        st.caption("Confusion Matrix not found (expected at reports/figures/confusion_matrix.png)")

    if FIG_WC.exists():
        st.image(str(FIG_WC), caption="Word Cloud – Toxic Comments", use_container_width=True)
    else:
        st.caption("Word Cloud not found (expected at reports/figures/wordcloud.png)")

# -------------------------------------------------------------------
# Batch classification + downloads
# -------------------------------------------------------------------
st.header("⚙️ Batch Classification")

batch_file = st.file_uploader(
    "Upload a CSV to classify (must contain `comment_text` column)",
    type=["csv"],
    key="batch_uploader",
)

if batch_file is not None:
    try:
        bdf = pd.read_csv(batch_file)
        if "comment_text" not in bdf.columns:
            st.error("Missing column `comment_text` in the uploaded file.")
        else:
            Xb = vectorizer.transform(bdf["comment_text"].astype(str))
            probs = model.predict_proba(Xb)[:, 1]
            preds = (probs >= threshold).astype(int)

            out = bdf.copy()
            out["toxicity_prob"] = probs
            out["toxic_pred"] = preds

            st.success(f"Classified {len(out)} rows with threshold = {threshold:.2f}")
            st.dataframe(out.head(30), use_container_width=True)

            # Download: full results
            csv_all = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download classified CSV",
                data=csv_all,
                file_name="classified_comments.csv",
                mime="text/csv",
            )

            # Download: toxic-only rows
            toxic_only = out[out["toxic_pred"] == 1]
            csv_toxic = toxic_only.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download only Toxic rows",
                data=csv_toxic,
                file_name="toxic_comments_only.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Failed to classify file: {e}")

# -------------------------------------------------------------------
# Footer / tip
# -------------------------------------------------------------------
st.caption("Tip: Tune the threshold to trade off recall (catch more toxic) vs precision (fewer false alarms).")
