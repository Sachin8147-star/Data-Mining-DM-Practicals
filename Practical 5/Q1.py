import random
import matplotlib.pyplot as plt

# Generate 20 random 1-D points
points = [random.randint(1, 100) for _ in range(20)]

print("Points:", points)

# Take k from user
k = int(input("Enter value of k: "))

# Randomly select initial centroids
centroids = random.sample(points, k)

for iteration in range(10):

    # Create clusters
    clusters = [[] for _ in range(k)]

    for point in points:
        distances = [abs(point - c) for c in centroids]
        nearest = distances.index(min(distances))
        clusters[nearest].append(point)

    # Calculate new centroids
    new_centroids = []

    for cluster in clusters:
        if cluster:
            new_centroids.append(sum(cluster) / len(cluster))
        else:
            new_centroids.append(random.choice(points))

    # Stop if centroids do not change
    if new_centroids == centroids:
        break

    centroids = new_centroids

# Display result
print("\nFinal Centroids:", centroids)

for i in range(k):
    print("Cluster", i + 1, ":", clusters[i])

# Plot clusters
for i in range(k):
    plt.scatter(clusters[i], [i] * len(clusters[i]), label=f"Cluster {i+1}")

plt.scatter(centroids, [0] * k, marker="x", s=100, label="Centroids")

plt.xlabel("Point Value")
plt.ylabel("Cluster")
plt.title("K-Means Clustering")
plt.legend()
plt.show()