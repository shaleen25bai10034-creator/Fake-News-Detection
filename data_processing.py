"""Data processing script: reads a CSV with columns 'title', 'text', 'label' and outputs cleaned CSV."""
import argparse
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in STOP]
    return " ".join(tokens)

def process(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df['text_filled'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['clean_text'] = df['text_filled'].apply(clean_text)
    df = df[['clean_text', 'label']].rename(columns={'clean_text':'text'})
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    process(args.input, args.output)
