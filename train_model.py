import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from src.features import build_vectorizer, save_vectorizer

def train(data_csv, model_out):
    df = pd.read_csv(data_csv)
    X_text = df['text'].astype(str).values
    y = df['label'].map({'real':0,'fake':1}).values

    vec, X = build_vectorizer(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    params = {'C':[0.1,1,10]}
    gs = GridSearchCV(clf, params, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    preds = best.predict(X_test)
    print(classification_report(y_test, preds))
    try:
        print('ROC-AUC:', roc_auc_score(y_test, best.predict_proba(X_test)[:,1]))
    except Exception:
        pass

    joblib.dump(best, model_out)
    save_vectorizer(vec, model_out.replace('.pkl','_vec.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    train(args.data, args.model)
