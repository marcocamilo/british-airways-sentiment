import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from itertools import chain
from matplotlib_venn import venn2

df = pd.read_parquet('./data/3-processed/ba-reviews-preds-bert.parquet') 

# Value Count
sent_count = df['bert_preds'].value_counts().reset_index()

sns.barplot(sent_count, x='bert_preds', y='count')
plt.show()

# Most frequent words
words = df['split-reviews']
words = list(chain(*words))
word_count = Counter(words).most_common()

pos_words = df.query('bert_preds == 1')['split-reviews']
pos_words = list(chain(*pos_words))
pos_word_count = Counter(pos_words).most_common()
top_pos_words = [word for word, _ in pos_word_count[:100]]

neg_words = df.query('bert_preds == 0')['split-reviews']
neg_words = list(chain(*neg_words))
neg_word_count = Counter(neg_words).most_common()
top_neg_words = [word for word, _ in neg_word_count[:100]]

# Wordcloud
only_pos = set(top_pos_words) - set(top_neg_words)
only_neg = set(top_neg_words) - set(top_pos_words)
common = set(top_pos_words) & set(top_neg_words)

data = [only_pos, only_neg, common]
titles = ['Positive', 'Negative', 'Common']

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
axes = axes.flatten()

for count, ax, title in zip(data, axes, titles):
    wordcloud = WordCloud().generate(' '.join(count))
    ax.imshow(wordcloud)
    ax.axis('off')
    ax.set_title(title)

plt.show()
