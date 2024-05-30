import pandas as pd
from transformers import pipeline

df = pd.read_parquet('./data/3-processed/ba-reviews-features.parquet', columns=['clean-reviews', 'split-reviews'])

reviews = df['clean-reviews'].to_list()
sentiment_pipeline = pipeline("sentiment-analysis")
sentiments = sentiment_pipeline(reviews)

def sent2label(sentiment):
  map = {'NEGATIVE': 0, 'POSITIVE': 1}
  sentiment = sentiment['label']
  label = map[sentiment]
  return label

labels = list(map(sent2label, sentiments))
df['bert_preds'] = labels

df.to_parquet('./data/3-processed/ba-reviews-preds-bert.parquet')
