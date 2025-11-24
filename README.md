

**Project:** Fake News Detection using NLP & Machine Learning

## Overview
A pipeline to detect fake news from text articles. Includes data ingestion, preprocessing, feature extraction (TF-IDF + optional embeddings), model training (Logistic Regression / XGBoost), evaluation, and a small Flask demo for inference.

## Features
- Data cleaning and preprocessing
- Feature extraction (TF-IDF)
- Train/test pipeline with hyperparameter search
- Model evaluation (accuracy, precision, recall, F1, ROC-AUC)
- Inference API (Flask)
- Unit tests

## Tech stack
- Python 3.10+
- scikit-learn, XGBoost
- pandas, numpy
- Flask (for demo)

## Quick start
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
2. Prepare data: put raw CSVs in `data/raw/` (see `data/raw/README.md`).
3. Preprocess and extract features:
```bash
python src/data_processing.py --input data/raw/news.csv --output data/processed/news_processed.csv
```
4. Train model:
```bash
python src/train_model.py --data data/processed/news_processed.csv --model models/best_model.pkl
```
5. Run demo API:
```bash
python src/app.py --model models/best_model.pkl
```
