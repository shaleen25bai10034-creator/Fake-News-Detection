import joblib

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.vec = joblib.load(model_path.replace('.pkl','_vec.pkl'))

    def predict(self, texts):
        X = self.vec.transform(texts)
        probs = self.model.predict_proba(X)[:,1]
        labels = (probs >= 0.5).astype(int)
        return labels, probs

if __name__ == '__main__':
    import sys
    p = Predictor(sys.argv[1])
    texts = [" ".join(sys.argv[2:])]
    print(p.predict(texts))
