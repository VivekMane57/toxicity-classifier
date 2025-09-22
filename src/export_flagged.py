import pandas as pd, joblib
from .config import DATA_RAW, TRAIN_FILENAME, TEXT_COL, LABEL_COL
from .preprocess import clean_text

THRESH = 0.5  # raise to 0.6 for stricter flags
df = pd.read_csv(DATA_RAW / TRAIN_FILENAME)
df["cleaned"] = df[TEXT_COL].astype(str).apply(clean_text)

vec = joblib.load("models/tfidf.joblib")
mdl = joblib.load("models/logreg_toxic.joblib")

probs = mdl.predict_proba(vec.transform(df["cleaned"]))[:,1]
df_out = df.assign(toxicity_prob=probs, toxic_pred=(probs>=THRESH).astype(int))
df_out[df_out["toxic_pred"]==1].to_csv("data/processed/flagged_comments.csv", index=False)
print("Saved: data/processed/flagged_comments.csv")
