import pandas as pd
from src.modules.utils import preprocessing

df = pd.read_parquet('./data/1-raw/ba-reviews.parquet')

df['clean-reviews'] = df['reviews'].apply(preprocessing)

df.to_parquet('./data/2-interim/ba-reviews-cleaned.parquet')
