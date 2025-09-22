
from pathlib import Path

# Paths
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Filenames (adjust to your dataset)
TRAIN_FILENAME = "train.csv"  # place in data/raw/
TEXT_COL = "comment_text"
LABEL_COL = "toxic"

# TF-IDF
TFIDF_MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)  # words; change to chars if you like
TEST_SIZE = 0.2
RANDOM_STATE = 42
