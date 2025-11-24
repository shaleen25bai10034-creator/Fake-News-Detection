# Problem Statement

The spread of fake news on social media and news websites influences public opinion and can cause harm. This project aims to build an automated system to classify news articles as real or fake using natural language processing and machine learning techniques.

## Scope
- Binary classification: `real` vs `fake`
- Uses textual article content; optional use of metadata
- End-to-end pipeline: preprocessing -> feature extraction -> model training -> API

## Target Users
- Educators and students for research
- Journalists and fact-checkers for triage and prioritization
- Developers who want a baseline ML system to detect fake news

## High-level features
- Clean & preprocess raw news text
- Extract TF-IDF and n-gram features
- Train and evaluate models (with cross-validation)
- Provide a lightweight Flask API for inference

