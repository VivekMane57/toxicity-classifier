
# YouTube Comments Toxicity Classifier (Starter)

This is a ready-to-run VS Code project scaffold for building an NLP model that detects toxic comments.
It includes preprocessing, a baseline TF-IDF + Logistic Regression model, a Streamlit dashboard, and VS Code configs.

## 🌐 Live Demo
🚀 Try the app here: [Toxicity Classifier Streamlit App](https://youtub-toxicity-classifier.streamlit.app/)

*(Runs on Streamlit Cloud — may take a minute to load for the first time)*

## 🚀 Features  
- **Text Preprocessing**: cleaning, tokenization, lowercasing, stopword removal  
- **TF–IDF Feature Extraction** for NLP  
- **Logistic Regression Classifier**  
  - Accuracy: ~94%  
  - Precision (toxic): ~0.64  
  - Recall (toxic): ~0.86  
  - F1-score (toxic): ~0.74  
  - PR AUC: ~0.86  
- **Interactive Streamlit Dashboard**:  
  - Classify individual comments  
  - Upload CSV files for batch classification  
  - Adjust decision threshold  
  - Export toxic-only comments  
- **Visual Insights**: Confusion Matrix 📊 & Word Cloud ☁️  

---

## 📂 Dataset  
- Source: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  
- 160k+ YouTube comments labeled for toxicity (binary classification in this project).  

---

## 🛠️ Tech Stack  
- **Python**: Pandas, Scikit-learn, Seaborn, Matplotlib, WordCloud  
- **Streamlit**: dashboard & deployment  
- **Joblib**: model persistence

  ---

## 🖼️ Screenshots

### 🔎 Single Comment Prediction
## Classify individual YouTube comments with probability score.
<img width="1908" height="1076" alt="image" src="https://github.com/user-attachments/assets/61c79019-f4ec-40ff-a47c-4333bb06d422" />


### 📊 Dataset Explorer & Confusion Matrix
## Explore uploaded CSVs, check toxicity distribution, and view model performance.
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/34f4c36c-e8ae-4852-a55c-052d1868d35e" />

### 📦 Batch Classification
## Upload large CSV files and classify all comments at once with adjustable threshold.
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/c35836ff-622d-4dbb-9eed-eee80441309d" />

## ⚙️ Installation & Setup  

Clone this repo:
```bash
git clone https://github.com/VivekMane57/toxicity-classifier.git
cd toxicity-classifier


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
