#Load & Preprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Load data ---
rfm = pd.read_csv("customer_data.csv")

# Standardize RFM features
scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm)

# Choose Optimal Number of Clusters
sil_scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled_rfm)
    sil_scores[k] = silhouette_score(scaled_rfm, labels)

best_k = max(sil_scores, key=sil_scores.get)
print(f"Best k by silhouette: {best_k} (score={sil_scores[best_k]:.3f})")

# Plot silhouette scores
plt.figure(figsize=(8,4))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Scores for KMeans")
plt.show()

#Final Model & Cluster Profiling
kmeans = KMeans(n_clusters=best_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(scaled_rfm)

cluster_profile = rfm.groupby('Cluster').agg(['mean','median','count'])
print("\nCluster Profile:\n", cluster_profile)

# Heatmap of mean RFM per cluster
plt.figure(figsize=(8,5))
sns.heatmap(rfm.groupby('Cluster').mean(), annot=True, fmt='.1f', cmap='Blues')
plt.title('Mean RFM Values by Cluster')
plt.show()

# Optional: distribution of each cluster size
rfm['Cluster'].value_counts().plot(kind='bar')
plt.title("Cluster Counts")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()

# Save results for dashboard
rfm.to_csv("customer_data_with_clusters.csv", index=False)
