import os
import numpy as np  # ✅ added
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import joblib

from .config import DATA_RAW, TRAIN_FILENAME, TEXT_COL, LABEL_COL, TEST_SIZE, RANDOM_STATE
from .preprocess import clean_text
from .features_tfidf import build_tfidf
from .evaluate import evaluate_and_print

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def main():
    data_path = DATA_RAW / TRAIN_FILENAME
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Place your CSV there.")

    df = pd.read_csv(data_path)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Expected columns: {TEXT_COL}, {LABEL_COL}")

    df["cleaned"] = df[TEXT_COL].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df[LABEL_COL].astype(int),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]
    )

    vectorizer = build_tfidf()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ✅ class weights to mitigate imbalance (NumPy array required)
    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.to_numpy())
    class_weight = {cls: w for cls, w in zip(classes, cw)}

    model = LogisticRegression(max_iter=200, class_weight=class_weight, n_jobs=None)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    evaluate_and_print(y_test, y_pred, target_names=["non-toxic", "toxic"])

    # PR AUC (average precision) for positive class
    probs = model.predict_proba(X_test_vec)[:, 1]
    ap = average_precision_score(y_test, probs)
    print(f"Average Precision (PR AUC) for toxic class: {ap:.4f}")

    # save artifacts
    joblib.dump(vectorizer, MODELS_DIR / "tfidf.joblib")
    joblib.dump(model, MODELS_DIR / "logreg_toxic.joblib")
    print("Saved: models/tfidf.joblib, models/logreg_toxic.joblib")

if __name__ == "__main__":
    main()
