
# YouTube Comments Toxicity Classifier (Starter)

This is a ready-to-run VS Code project scaffold for building an NLP model that detects toxic comments.
It includes preprocessing, a baseline TF-IDF + Logistic Regression model, a Streamlit dashboard, and VS Code configs.

## Quickstart

```bash
# 1) Open folder in VS Code, then in Terminal:
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put your dataset (CSV) in: data/raw/
#    Expect columns: comment_text, toxic (0/1)  -- or adapt src/config.py

# 4) Train baseline
python src/models_baseline.py

# 5) Run dashboard
streamlit run dashboard/app.py
```

## Structure
```
toxicity-classifier/
├─ data/{raw,processed}
├─ src/
│  ├─ config.py
│  ├─ preprocess.py
│  ├─ features_tfidf.py
│  ├─ models_baseline.py
│  ├─ evaluate.py
│  ├─ inference.py
├─ dashboard/app.py
├─ notebooks/01_eda.ipynb
├─ .vscode/{settings.json,launch.json,tasks.json}
├─ requirements.txt
├─ .gitignore
├─ README.md
```

## Notes
- The baseline uses TF-IDF + Logistic Regression.
- Focus metric: Recall on the toxic (positive) class; we report full classification report too.
