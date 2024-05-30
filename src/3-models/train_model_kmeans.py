import pandas as pd
import gzip
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet('./data/3-processed/ba-reviews-features.parquet')
with gzip.open('./models/embedding_matrix.pkl.gz') as f:
    X = pickle.load(f)
    
#  ────────────────────────────────────────────────────────────────────
#   DATA PREPROCESSING                                                 
#  ────────────────────────────────────────────────────────────────────
X_sent = np.mean(X, axis=1)
scaler = StandardScaler()
averaged_matrix = scaler.fit_transform(X_sent)

#  ────────────────────────────────────────────────────────────────────
#   PCA                                                                
#  ────────────────────────────────────────────────────────────────────
# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(averaged_matrix)
explained_variance = pca.explained_variance_ratio_

#  ────────────────────────────────────────────────────────────────────
#   KMeans                                                             
#  ────────────────────────────────────────────────────────────────────
n_clusters = 2
kmeans = KMeans(n_clusters, max_iter=50, verbose=1)
kmeans_labels = kmeans.fit_predict(averaged_matrix)

plt.figure(figsize=(8, 6))
for cluster in range(n_clusters):
    cluster_points = pca_result[kmeans_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
plt.title('KMeans Clusters of Averaged Embedding Matrix')
plt.legend()
plt.show()

#  ────────────────────────────────────────────────────────────────────
#   PREDICTIONS                                                        
#  ────────────────────────────────────────────────────────────────────
df['kmean-preds'] = kmeans_labels
