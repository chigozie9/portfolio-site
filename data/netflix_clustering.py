
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('netflix_ratings.csv', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Aggregate user behavior
user_profiles = df.groupby('user_id').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count'),
    rating_std=('rating', 'std')
).fillna(0)

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_profiles)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
user_profiles['cluster'] = kmeans.fit_predict(scaled_features)

# Reduce to 2D with PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
user_profiles['pca1'] = pca_result[:, 0]
user_profiles['pca2'] = pca_result[:, 1]

# Save PCA cluster plot
plt.figure(figsize=(10, 6))
for label in sorted(user_profiles['cluster'].unique()):
    subset = user_profiles[user_profiles['cluster'] == label]
    plt.scatter(subset['pca1'], subset['pca2'], label=f'Cluster {label}', alpha=0.6)
plt.title("Netflix User Clusters (via K-Means + PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.savefig("netflix_user_clusters.png")
plt.close()

# Save cluster size chart
cluster_counts = user_profiles['cluster'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
cluster_counts.plot(kind='bar', color='skyblue')
plt.title("Number of Users per Cluster")
plt.xlabel("Cluster")
plt.ylabel("User Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("netflix_cluster_sizes.png")
plt.close()

# Save avg rating per cluster chart
avg_rating_per_cluster = user_profiles.groupby('cluster')['avg_rating'].mean()
plt.figure(figsize=(6, 4))
avg_rating_per_cluster.plot(kind='bar', color='mediumseagreen')
plt.title("Average Rating per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig("netflix_cluster_avg_rating.png")
plt.close()
