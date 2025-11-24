from flask import Flask, request, jsonify
import argparse
from src.predict import Predictor

app = Flask(__name__)
predictor = None

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    text = data.get('text','')
    labels, probs = predictor.predict([text])
    return jsonify({'label': int(labels[0]), 'probability': float(probs[0])})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    predictor = Predictor(args.model)
    app.run(host='0.0.0.0', port=5000)
