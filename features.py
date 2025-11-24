from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def build_vectorizer(corpus, max_features=20000, ngram_range=(1,2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(corpus)
    return vec, X

def save_vectorizer(vec, path):
    joblib.dump(vec, path)

def load_vectorizer(path):
    return joblib.load(path)
