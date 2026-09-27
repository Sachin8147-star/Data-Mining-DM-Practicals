import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([
    [1, 2], [2, 1], [2, 3], [3, 2],
    [8, 8], [9, 7], [8, 9], [7, 8],
    [15, 2], [16, 3], [15, 4], [17, 2]
])

# K-Means function
def kmeans(X, k, max_iterations=10):

    # Select first k points as initial centroids
    centroids = X[:k].astype(float)

    mse_history = []

    for iteration in range(max_iterations):

        # Calculate distance of every point from every centroid
        distances = np.zeros((len(X), k))

        for i in range(len(X)):
            for j in range(k):
                distances[i][j] = np.linalg.norm(X[i] - centroids[j])

        # Assign points to nearest centroid
        labels = np.argmin(distances, axis=1)

        # Calculate MSE
        mse = 0

        for i in range(len(X)):
            mse += np.sum((X[i] - centroids[labels[i]]) ** 2)

        mse = mse / len(X)
        mse_history.append(mse)

        # Calculate new centroids
        new_centroids = []

        for j in range(k):
            cluster = X[labels == j]

            if len(cluster) > 0:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(centroids[j])

        new_centroids = np.array(new_centroids)

        # Stop if centroids do not change
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids, mse_history


# Take parameters from user
k = int(input("Enter number of clusters (k): "))
iterations = int(input("Enter maximum iterations: "))

# Apply K-Means
labels, centroids, mse_history = kmeans(X, k, iterations)

print("\nFinal Centroids:")
print(centroids)

print("\nMSE after each iteration:")
for i, mse in enumerate(mse_history):
    print("Iteration", i + 1, ":", mse)


# Plot clusters
plt.figure()

for i in range(k):
    cluster = X[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i+1}")

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")
plt.legend()
plt.show()


# Plot MSE vs iterations
plt.figure()

plt.plot(
    range(1, len(mse_history) + 1),
    mse_history,
    marker="o"
)

plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("MSE after Each Iteration")
plt.grid()

plt.show()