import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin

from src.data.data_processor import filter_data


class CustomKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, n_init=5, max_iter=100,
                 tol=1e-4, random_state=42, distance_metric="cosine"):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        if distance_metric == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            self.distance_metric = cosine_distances
        else:
            self.distance_metric = distance_metric

    def euclidean_distance(self, X, Y):
        return np.linalg.norm(X - Y, axis=1)

    def fit(self, X, y=None):
        X = np.asarray(X)
        best_inertia = None

        for _ in range(self.n_init):
            centroids = self._initialize_centroids(len(X[0]))
            for i in range(self.max_iter):
                # Assign labels
                labels = self._assign_labels(X, centroids)

                # Recalculate centroids
                new_centroids = np.zeros_like(centroids)
                for j in range(self.n_clusters):
                    mask = labels == j
                    if np.sum(mask) == 0:  # No signals assigned to cluster
                        continue
                    new_centroids[j] = X[labels == j].mean(axis=0)

                # Calculate shift
                shift = np.sum(self.distance_metric(centroids, new_centroids)[np.eye(self.n_clusters, dtype=bool)])
                centroids = new_centroids
                if shift <= self.tol:
                    break

            inertia = np.sum([np.sum(self.distance_metric(X[labels == j], centroids[j].reshape(1, -1))[0] ** 2) if sum(labels == j) > 0 else np.inf for j in range(self.n_clusters)])
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centroids
                self.labels_ = labels
                self.inertia_ = inertia

        return self

    def _initialize_centroids(self, length):
        centroids = np.array([np.random.normal(size=length) for _ in range(self.n_clusters)])
        return centroids

    def _assign_labels(self, X, centroids):
        # distances = np.array([self.distance_metric(X, c.reshape(1, -1)) for c in centroids]).T
        distances = self.distance_metric(X, centroids)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        X = np.asarray(X)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

