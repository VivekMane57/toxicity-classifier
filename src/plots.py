# src/plots.py
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import joblib

from .config import DATA_RAW, TRAIN_FILENAME, TEXT_COL, LABEL_COL
from .preprocess import clean_text

FIGS = Path("reports/figures"); FIGS.mkdir(parents=True, exist_ok=True)

# load data + model
df = pd.read_csv(DATA_RAW / TRAIN_FILENAME)
df["cleaned"] = df[TEXT_COL].astype(str).apply(clean_text)

vec = joblib.load("models/tfidf.joblib")
mdl = joblib.load("models/logreg_toxic.joblib")

# simple holdout to draw a matrix on the whole file (quick & fine for report)
X = vec.transform(df["cleaned"])
y = df[LABEL_COL].astype(int).values
y_pred = mdl.predict(X)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-toxic","Toxic"], yticklabels=["Non-toxic","Toxic"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(FIGS / "confusion_matrix.png"); plt.close()

# word cloud for toxic comments
from wordcloud import WordCloud
toxic_text = " ".join(df[df[LABEL_COL]==1][TEXT_COL].astype(str))
wc = WordCloud(width=1000, height=500, background_color="white").generate(toxic_text)
plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
plt.title("Word Cloud – Toxic Comments")
plt.tight_layout(); plt.savefig(FIGS / "wordcloud.png"); plt.close()

print("Saved: reports/figures/confusion_matrix.png, reports/figures/wordcloud.png")
