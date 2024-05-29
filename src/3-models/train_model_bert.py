import pandas as pd
from transformers import pipeline

df = pd.read_parquet('./data/3-processed/ba-reviews-features.parquet', columns=['clean-reviews', 'split-reviews'])

reviews = df['clean-reviews']
sentiment_pipeline = pipeline("sentiment-analysis")
sentiment_pipeline(reviews)
