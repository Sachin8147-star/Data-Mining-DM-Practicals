
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

columns = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]

df = pd.read_csv(url, header=None, names=columns)

print("Dataset Shape:", df.shape)

# Select features
X = df.drop(["ID", "Diagnosis"], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Different DBSCAN parameters
parameters = [
    (2.5, 5),
    (2.5, 10),
    (3.0, 3),
    (3.0, 5),
    (3.5, 3)
]

results = []
best_score = -2
best_labels = None
best_params = None

# Apply DBSCAN
for eps, min_samples in parameters:

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_scaled)

    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = np.sum(labels == -1)

    # Calculate Silhouette Score
    mask = labels != -1

    if clusters >= 2:
        score = silhouette_score(
            X_scaled[mask],
            labels[mask]
        )
    else:
        score = None

    results.append([
        eps, min_samples, clusters, noise, score
    ])

    if score is not None and score > best_score:
        best_score = score
        best_labels = labels
        best_params = (eps, min_samples)

# Display results
result_df = pd.DataFrame(
    results,
    columns=[
        "Eps", "Min Samples", "Clusters",
        "Noise Points", "Silhouette Score"
    ]
)

print("\nDBSCAN Results:")
print(result_df.to_string(index=False))

result_df.to_csv("DBSCAN_results.csv", index=False)

# Visualize clusters using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

if best_labels is not None:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=best_labels,
        cmap="viridis",
        s=25
    )

    plt.title("DBSCAN Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()

    plt.savefig("DBSCAN_clusters.png")
    plt.show()

    print("\nBest Parameters:", best_params)
    print("Best Silhouette Score:", round(best_score, 4))

print("\nClustering Completed Successfully!")
