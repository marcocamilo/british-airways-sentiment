import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import pickle
import gzip


df = pd.read_parquet('./data/2-interim/ba-reviews-cleaned.parquet', columns=['clean-reviews'])

reviews = df['clean-reviews'].values
reviews_split = [review.split() for review in reviews]
df['split-reviews'] = reviews_split

vector_size = 50
max_length = max(map(len, reviews_split))
n_sequences = df.shape[0]

model = Word2Vec(reviews_split, vector_size=50, window=4)
vectors = model.wv

embedding_matrix = np.zeros((n_sequences, max_length, vector_size))

for i, seq in enumerate(reviews_split):
    words = np.array(seq)
    word_indices = np.array([i for i, word in enumerate(seq) if word in vectors])
    valid_vectors = np.array([vectors[word] for word in words[word_indices]])

    embedding_matrix[i, word_indices, :] = valid_vectors

with gzip.open('./models/embedding_matrix.pkl.gz', 'wb') as f:
    pickle.dump(embedding_matrix, f)

df.to_parquet('./data/3-processed/ba-reviews-features.parquet')
