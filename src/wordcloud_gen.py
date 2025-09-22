# src/wordcloud_gen.py
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

from .config import DATA_RAW, TRAIN_FILENAME, TEXT_COL, LABEL_COL  # relative import

os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv(DATA_RAW / TRAIN_FILENAME)
toxic_text = " ".join(df[df[LABEL_COL] == 1][TEXT_COL].astype(str))

wc = WordCloud(width=1000, height=500, background_color="white").generate(toxic_text)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
plt.title("Most Common Words in Toxic Comments")
plt.tight_layout()
plt.savefig("reports/figures/wordcloud.png")
plt.close()
print("Saved: reports/figures/wordcloud.png")
