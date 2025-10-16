import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from tqdm import tqdm


from sklearn.neighbors import NearestNeighbors


class CustomAgglomerativeClustering:
    def __init__(self, 
                 k_neighbors: int = 10, 
                 k0_neighbors: int = 1, 
                 distance_metric: str = 'cosine'):
        """
        Graph-based agglomerative clustering based on Zhang et al. (2012)
        :param k_neighbors: number of neighbors to consider when clustering
        :param k0_neighbors: number of neighbors to consider during the initial clustering
        :param distance: distance metric
        """

        self.k_neighbors = k_neighbors
        self.k0_neighbors = k0_neighbors
        self.distance_metric = distance_metric

        self.sigma2 = None
        self.weights = None
        self.indegree = None
        self.outdegree = None
        self.point_affinity_matrix = None
        self.cluster_affinity_matrix = None

        self.linkage = None
        self.labels_ = None
        self.label_history = []
        self.full_history = []

        # Hyperparameters - TODO: optimize
        self.K = 1
        self.a = 1

    def _initialize_clustering(self, data: np.ndarray):
        """
        Build the initial k-NN graph for the initial clustering
        :param data: (array) shape (n_samples, n_features)
        :return:
        """

        # Import packages
        import networkx as nx

        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k0_neighbors + 1, metric=self.distance_metric).fit(data)
        dist, idx = nbrs.kneighbors(data, return_distance=True)

        # Exclude the point itself from the neighbor list
        n_samples = data.shape[0]
        indices = np.zeros((n_samples, self.k0_neighbors))
        distances = np.zeros((n_samples, self.k0_neighbors))
        for i in range(n_samples):
            if i in idx[i]:
                indices[i] = [x for x in idx[i] if x != i]
                distances[i] = [x for j, x in enumerate(dist[i]) if idx[i][j] != i]
            else:
                indices[i] = idx[i][:-1]
                distances[i] = dist[i][:-1]

        # Create a directed graph using adjacency list
        G = nx.DiGraph()
        G.add_nodes_from(range(n_samples))
        for i in range(n_samples):  # First index is the point itself
            for neighbor in indices[i]:
                G.add_edge(i, neighbor)

        # Store cluster labels_
        self.labels_ = [0] * n_samples
        weakly_connected_components = list(nx.weakly_connected_components(G))
        for i in range(len(weakly_connected_components)):
            for j in weakly_connected_components[i]:
                self.labels_[int(j)] = i

        # Store initial labeling
        self.full_history.append(self.labels_)
        self.label_history.append(self.labels_)

    def _build_weighted_knn_graph(self, data: np.ndarray):
        """
        Build the weighted k-NN graph for the final clustering
        :param data: (array) shape (n_samples, n_features)
        :return:
        """

        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric=self.distance_metric).fit(data)
        dist, idx = nbrs.kneighbors(data, return_distance=True)

        # Exclude the point itself from the neighbors
        n_samples = data.shape[0]
        indices = np.zeros((n_samples, self.k_neighbors))
        distances = np.zeros((n_samples, self.k_neighbors))
        for i in range(n_samples):
            if i in idx[i]:
                indices[i] = [x for x in idx[i] if x != i]
                distances[i] = [x for j, x in enumerate(dist[i]) if idx[i][j] != i]
            else:
                indices[i] = idx[i][:-1]
                distances[i] = dist[i][:-1]

        indices = indices.astype(int)

        # # Calculate Sigma^2
        self.sigma2 = 0
        for i in range(n_samples):
            for j in range(self.k_neighbors):
                if indices[i][j] != i:  # Exclude the own point from the neighbors
                    self.sigma2 += distances[i][j]**2

        self.sigma2 *= self.a / (n_samples * self.K)

        # Build the K-NN graph
        self.weights = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(self.k_neighbors):
                self.weights[i][indices[i][j]] = np.exp(-distances[i][j]**2 / self.sigma2)

    def _calculate_affinity(self):
        """
        Calculate the affinity between clusters using indegree and outdegree
        :return:
        """

        labels = np.array(self.labels_)
        n_clusters = len(np.unique(labels))
        self.cluster_affinity_matrix = np.zeros((n_clusters, n_clusters))

        cluster_indices = [np.where(labels == i)[0] for i in range(n_clusters)]
        for i in range(n_clusters):
            idx_i = cluster_indices[i]
            for j in range(n_clusters):
                if i == j:
                    continue

                idx_j = cluster_indices[j]
                len_j = len(idx_j)

                # Sparse submatrices
                W_ab = self.weights[idx_i][:, idx_j]
                W_ba = self.weights[idx_j][:, idx_i]

                # Compute row/column sums efficiently
                sum_W_ba = np.array(W_ba.sum(axis=0)).ravel()

                # Perform dot products
                self.cluster_affinity_matrix[i, j] = sum_W_ba @ W_ab @ np.ones(len_j) / (len_j ** 2)

    def _update_affinity(self, a, b):
        """
        Update the affinity matrix - Call only after updating self.labels_
        :param a: Label of the first merged cluster
        :param b: Label of the second merged cluster
        :return:
        """
        # TODO: test with smaller dataset
        # Generate new affinity matrix
        new_affinity = np.zeros((len(self.cluster_affinity_matrix) - 1, len(self.cluster_affinity_matrix) - 1))

        # Input the affinity of the unmerged clusters
        new_affinity[:-1, :-1] = np.delete(np.delete(self.cluster_affinity_matrix, [a, b], axis=0), [a, b], axis=1)

        # A_(ab->c) = A_(a->c) + A_(b->c)
        new_affinity[-1, :-1] = np.delete(self.cluster_affinity_matrix[a] + self.cluster_affinity_matrix[b], [a, b])

        # Calculate A_(c->ab)
        cluster_indices = [np.where(np.array(self.labels_) == i)[0] for i in range(len(new_affinity))]
        idx_j = cluster_indices[-1]
        len_j = len(idx_j)
        for i in range(len(new_affinity) - 1):
            idx_i = cluster_indices[i]

            # Sparse submatrices
            W_c_ab = self.weights[idx_i][:, idx_j]
            W_ab_c = self.weights[idx_j][:, idx_i]

            # Compute row/column sums efficiently
            sum_W_ab_c = np.array(W_ab_c.sum(axis=0)).ravel()

            # Perform dot products
            new_affinity[i, -1] = sum_W_ab_c @ W_c_ab @ np.ones(len_j) / (len_j ** 2)

        self.cluster_affinity_matrix = new_affinity

    def fit(self, data: np.ndarray):
        """
        Perform GDL clustering
        :param data: (array) shape (n_samples, n_features)
        :return:
        """
        # Initial clustering
        self._initialize_clustering(data)

        clusters = np.unique(self.labels_).tolist()
        sizes = np.ones(len(clusters))

        n_clusters = len(clusters)
        max_cluster = max(clusters)

        # Build k-NN graph
        self._build_weighted_knn_graph(data)
        self._calculate_affinity()
        for _ in tqdm(range(n_clusters - 1), desc='Merging clusters'):
            if np.all(self.cluster_affinity_matrix <= 0):
                break

            affinity = self.cluster_affinity_matrix + self.cluster_affinity_matrix.T
            cluster_a, cluster_b = np.unravel_index(affinity.argmax(), affinity.shape)

            if cluster_a == cluster_b:
                break

            # Update linkage matrix
            # distance = 10 - affinity for dendrogram to plot correctly
            link = np.array([[clusters[cluster_a], clusters[cluster_b],
                              6 -  self.cluster_affinity_matrix[cluster_a][cluster_b],
                              sizes[cluster_a] + sizes[cluster_b]]])

            if self.linkage is None:
                self.linkage = link
            else:
                self.linkage = np.vstack((self.linkage, link))

            # Merge clusters
            max_cluster += 1
            self.full_history.append([max_cluster if x == clusters[cluster_a] or x == clusters[cluster_b] else x
                                      for x in self.full_history[-1]])

            clusters.remove(clusters[max(cluster_a, cluster_b)])
            clusters.remove(clusters[min(cluster_a, cluster_b)])
            clusters.append(max_cluster)

            sizes = np.append(sizes, sizes[cluster_a] + sizes[cluster_b])
            sizes = np.delete(sizes, [cluster_a, cluster_b])

            self.labels_ = [max_cluster if x == cluster_a or x == cluster_b else x for x in self.labels_]
            for i, label in enumerate(np.unique(self.labels_)):
                self.labels_ = [i if x == label else x for x in self.labels_]

            self.label_history.append(self.labels_)

            # Check if any cluster was merged
            if len(self.label_history) > 2:
                if len(np.unique(self.label_history[-1])) == len(np.unique(self.label_history[-2])):
                    break

            # Update affinity matrix
            self._update_affinity(cluster_a, cluster_b)

        return self

    def get_cluster_linkage(self):
        final_n = len(np.unique(self.labels_))  # Final number of clusters

        cluster_idx = {i: np.where(np.array(self.labels_) == i)[0] for i in range(final_n)}
        final_linkage = {i: [] for i in range(final_n)}
        for k, (i, j, sim, count) in enumerate(self.linkage):
            i, j = int(i), int(j)

            # Check index corresponding to these clusters
            labels = np.array(self.full_history[k])
            sample_idx = np.where(labels == i)[0][0]

            final_cluster = [n for n in range(final_n) if sample_idx in cluster_idx[n]][0]
            final_linkage[final_cluster].append([i, j, sim, count])

        final_linkage = {i: np.array(final_linkage[i]) for i in range(final_n)}
        cluster_mapping = []
        for i in range(final_n):
            cluster_labels = np.sort(np.append(final_linkage[i][:, 0], final_linkage[i][:, 1]))
            mapping = {int(old): int(new) for new, old in enumerate(cluster_labels)}
            cluster_mapping.append(mapping)
            for j in range(len(final_linkage[i])):
                final_linkage[i][j][0] = mapping[final_linkage[i][j][0]]
                final_linkage[i][j][1] = mapping[final_linkage[i][j][1]]

        return final_linkage, cluster_mapping

    def plot_step_n(self, n: int, data: np.ndarray, frequency: np.ndarray = None):
        """
        Plot the cluster that were merged on step -n
        :param n:
        :param data:
        :param frequency:
        :return:
        """

        if n > len(self.label_history):
            raise ValueError(f"Step {n} out of range")
        if len(data) != len(self.label_history[0]):
            raise ValueError(f"The amount of data does not match the number of labels")

        if frequency is None:
            frequency = np.arange(len(data[0]))

        cluster_a, cluster_b = self.linkage[n][:2]
        labels = np.array(self.full_history[n])

        if cluster_a not in labels or cluster_b not in labels:
            raise ValueError(f"Cluster {cluster_a} or {cluster_b} not found at step {n}")

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        ax[0].plot(frequency, data[labels==cluster_a].mean(axis=0), label=f'{sum(labels == cluster_a)} signals')
        ax[0].legend(loc='upper right')

        ax[1].plot(frequency, data[labels == cluster_b].mean(axis=0), label=f'{sum(labels == cluster_b)} signals')
        ax[1].legend(loc='upper right')

        fig.supxlabel("Frequency [MHz]")
        fig.supylabel("Intensity [Jy]")
        fig.tight_layout()

        return fig, ax

    def make_movie(self, data, frequency, steps: np.ndarray =None, fps=2, reverse=True, title=None):
        """

        :param data:
        :param frequency:
        :param steps:
        :param fps:
        :param reverse:
        :return:
        """
        import imageio
        import os


        if steps is None:
            order = -1 if reverse else 1
            steps = np.arange(1, len(self.label_history)) * order

        filenames = []
        folder_dir = 'Data/Movies/ac_movies'
        os.makedirs(folder_dir, exist_ok=True)
        for step in tqdm(steps, desc='Making movie:'):
            fig, ax = self.plot_step_n(step, data, frequency)
            fig.suptitle(f'Step {step}', fontsize=12)
            plt.savefig(f'{folder_dir}/{abs(step)}.png')
            plt.close(fig)

            filenames.append(f'{folder_dir}/{abs(step)}.png')

        if title is None:
            import datetime
            import time
            title = f"ac_video_{datetime.date.today()}_{time.time_ns()}"

        with imageio.get_writer(f'Data/Movies/{title}.mp4', fps=fps) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)

        os.rmdir(folder_dir)

    def locate_major_mergers(self, threshold=20):
        steps = []
        for i in range(1, len(self.linkage)):
            labels = np.array(self.full_history[i])
            cluster_a, cluster_b = self.linkage[i][:2]

            n_signals_a = sum(labels == cluster_a)
            n_signals_b = sum(labels == cluster_b)

            if n_signals_a > threshold and n_signals_b > threshold:
                steps.append(i)

        return steps


def compare_clusterings(list_of_labels: list, plot: bool = True, legend: list = None):
    """
    Compare the output of several GDL clusterings
    :param list_of_labels: list with several label_history lists from several GDL clusterings
    :param plot: flag for plotting the results
    :return:
    """

    n_clusters = []
    for i in range(len(list_of_labels)):
        n_clusters.append([len(np.unique(list_of_labels[i][j])) for j in range(len(list_of_labels[i]))])

    max_cluster = np.inf
    min_cluster = 0
    for i in range(len(n_clusters)):
        max_cluster = min(max_cluster, max(n_clusters[i]))
        min_cluster = max(min_cluster, min(n_clusters[i]))

    coincidence = []
    for n in range(max_cluster, min_cluster - 1, -1):
        sub_coincidence = []
        for i in range(len(n_clusters)):
            for j in range(i + 1, len(n_clusters)):
                idx_i = n_clusters[i].index(n)
                idx_j = n_clusters[j].index(n)

                sub_coincidence.append(adjusted_rand_score(list_of_labels[i][idx_i], list_of_labels[j][idx_j]))

        coincidence.append(sub_coincidence)

    if plot:
        plt.plot(range(max_cluster, min_cluster - 1, -1), coincidence)
        plt.legend(legend)
        plt.xlabel('Number of clusters')
        plt.ylabel('Adjusted Rand Score')
        plt.show()

    return coincidence


def check_reclustering(signals, centroids, true_labels, pred_labels, cluster):
    idx = (true_labels == cluster) * (pred_labels != cluster)
    new_labels = pred_labels[idx]

    for i in np.unique(new_labels):
        for s in signals[idx][new_labels == i]:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
            fig.suptitle(f'Cluster {cluster} -> {i}')

            ax[0].plot(freq, centroids[cluster])
            ax[0].tick_params(bottom=False)
            ax[0].set_title(f'Centroid {cluster}')

            ax[1].plot(freq, s)
            ax[1].tick_params(bottom=False)
            ax[1].set_title(f'Misclasified signals')

            ax[2].plot(freq, centroids[i])
            ax[2].set_title(f'Centroid {i}')

            fig.supxlabel('Frequency [MHz]', fontsize=12)
            fig.supylabel('Intensity [Jy]', fontsize=12)
            plt.show()



    
    
    # from src.data.data_processor import filter_data
    # from src.features.transformations import shift_signal

    # from sklearn.metrics import adjusted_rand_score
    # import matplotlib.pyplot as plt
    # import pandas as pd

    # # Load raw data
    # freq:np.ndarray = np.load('Data/frequencies.npy')
    # intensity_array = np.load('Data/intensity_array.npy')
    # residual_array = np.load('Data/residual_array.npy')

    # data_info: pd.DataFrame = pd.read_csv('Data/data_info.csv')
    # best_v = data_info['Best velocity 2'].values

    # # Shift data
    # # pyrefly: ignore
    # shifted_signals: np.ndarray = np.array([shift_signal(freq, intensity_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    # # pyrefly: ignore
    # shifted_residual: np.ndarray = np.array([shift_signal(freq, residual_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])

    # # Filter data
    # subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    # # sigma_signal = filter_data(np.asarray(shifted_signals), 'sigma', shifte   d_residual)
    # # savgol_signal = filter_data(np.asarray(shifted_signals), 'savgol')

    # from src.features.evaluation import piecewise_cosine_distance
    # # dist_matrix = cosine_distances(subtraction_signal, subtraction_signal)
    
    # dist_matrix = piecewise_cosine_distance(subtraction_signal, window=2500, stride=200)
    # print(f"Distance matrix shape: {dist_matrix.shape}")

    # # Train complete agglomerative clustering
    # ac_complete: AgglomerativeClustering = AgglomerativeClustering(
    #     n_clusters=None, 
    #     metric='precomputed', 
    #     linkage='complete', 
    #     distance_threshold=0.3)
    # ac_complete.fit(dist_matrix)
    
    # # Train average agglomerative clustering
    # ac_average: AgglomerativeClustering = AgglomerativeClustering(
    #     n_clusters=None, 
    #     metric='precomputed', 
    #     linkage='average', 
    #     distance_threshold=0.3)
    # ac_average.fit(dist_matrix)
    
    # # Save label assignments
    # labels_complete = ac_complete.labels_
    # labels_average = ac_average.labels_
    
    # np.save('models/agglomerative/piecewise_complete_labels.npy', labels_complete)
    # np.save('models/agglomerative/piecewise_average_labels.npy', labels_average)
