
import joblib
# (keep your class as-is)


class ToxicityPredictor:
    def __init__(self, vectorizer_path: str, model_path: str):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
