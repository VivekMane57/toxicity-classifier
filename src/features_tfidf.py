from sklearn.feature_extraction.text import TfidfVectorizer
def build_tfidf():
    # char n-grams catch obfuscations like "i d i o t" or "1nsult"
    return TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=200000)
