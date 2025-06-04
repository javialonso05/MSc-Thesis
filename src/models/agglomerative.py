import numpy as np
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors


class AgglomerativeClustering:
    def __init__(self, k_neighbors: int = 10, k0_neighbors: int = 1, distance_metric: str = 'cosine'):
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

        # Calculate Sigma^2
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

    def _calculate_linkage(self, linkage_type: str):
        """
        Calculate indegree or outdegree of all vertices
        :param linkage_type: (str) either "indegree" or "outdegree"
        :return:
        """

        # Check inputs
        if self.weights is None or len(self.weights) == 0:
            raise ValueError('self.weights is not defined')

        if linkage_type != "indegree" and linkage_type != "outdegree":
            raise ValueError('linkage_type must be "indegree" or "outdegree"')

        # Define variables
        n_samples = self.weights.shape[0]
        n_clusters = len(np.unique(self.labels_))

        linkage = np.zeros([n_samples, n_clusters])
        for i in range(n_samples):  # Vertex with respect to which indegree is calculated
            for j in range(n_clusters):  # Cluster to which vertex i is calculating its affinity
                if j == self.labels_[i]:
                    # Vertex i belongs to j
                    continue

                linkage_list = []
                cluster_idx = np.where(np.array(self.labels_) == j)[0]
                for k in cluster_idx:  # Vertex in cluster j
                    if linkage_type == "indegree":
                        linkage_list.append(self.weights[k][i])
                    else:
                        linkage_list.append(self.weights[i][k])

                linkage[i][j] = np.mean(linkage_list)

        if linkage_type == "indegree":
            self.indegree = linkage
        else:
            self.outdegree = linkage

    def _calculate_affinity(self):
        """
        Calculate the affinity between clusters using indegree and outdegree
        :return:
        """

        self._calculate_linkage("indegree")
        self._calculate_linkage("outdegree")
        self.point_affinity_matrix = self.indegree * self.outdegree

        n_clusters = len(np.unique(self.labels_))
        self.cluster_affinity_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            cluster_i_idx = np.where(np.array(self.labels_) == i)[0]
            for j in range(n_clusters):
                if i == j:
                    continue

                cluster_j_idx = np.where(np.array(self.labels_) == j)[0]

                # Calculate cluster affinity
                W_ca = self.weights[cluster_i_idx][:, cluster_j_idx]
                W_cb = self.weights[cluster_j_idx][:, cluster_i_idx]
                self.cluster_affinity_matrix[i][j] = (np.sum(np.matmul(np.sum(W_ca, axis=0), W_cb)) / len(W_ca)**2 +
                                                      np.sum(np.matmul(np.sum(W_cb, axis=0), W_ca)) / len(W_cb)**2)

    def fit(self, data: np.ndarray):
        """
        Perform GDL clustering
        :param data: (array) shape (n_samples, n_features)
        :return:
        """

        # Initial clustering
        self._initialize_clustering(data)
        clusters = np.unique(self.labels_).tolist()

        n_clusters = len(clusters)
        max_cluster = max(clusters)

        # Build k-NN graph
        self._build_weighted_knn_graph(data)
        for i in tqdm(range(n_clusters - 1), desc='Merging clusters'):
            # Compute affinity
            self._calculate_affinity()

            if np.all(self.cluster_affinity_matrix <= 0):
                break

            cluster_a, cluster_b = np.unravel_index(self.cluster_affinity_matrix.argmax(),
                                                self.cluster_affinity_matrix.shape)

            if cluster_a == cluster_b:
                break

            # Update linkage matrix
            # distance = 10 - affinity for dendrogram to plot correctly
            link = np.array([[clusters[cluster_a], clusters[cluster_b],
                              20 - self.cluster_affinity_matrix[cluster_a][cluster_b],
                              self.labels_.count(cluster_a) + self.labels_.count(cluster_b)]])

            if self.linkage is None:
                self.linkage = link
            else:
                self.linkage = np.vstack((self.linkage, link))

            # Merge clusters
            max_cluster += 1
            clusters.remove(clusters[max(cluster_a, cluster_b)])
            clusters.remove(clusters[min(cluster_a, cluster_b)])
            clusters.append(max_cluster)

            self.labels_ = [max_cluster if x == cluster_a or x == cluster_b else x for x in self.labels_]
            for i, label in enumerate(np.unique(self.labels_)):
                self.labels_ = [i if x == label else x for x in self.labels_]

            self.label_history.append(self.labels_)
            # Check if any cluster was merged
            if len(self.label_history) > 2:
                if len(np.unique(self.label_history[-1])) == len(np.unique(self.label_history[-2])):
                    break



if __name__ == '__main__':
    from src.data.data_processor import build_array, filter_data
    from src.features.transformations import shift_signal

    import matplotlib.pyplot as plt
    import pandas as pd
    import pickle

    # Load raw data
    interpolated_data = pickle.load(open('Data/Raw/interpolated_data_dict.pkl', 'rb'))

    # Build arrays
    f0 = interpolated_data['100132']['core1'][0]['Frequency']
    f1 = interpolated_data['100132']['core1'][1]['Frequency']
    freq = np.hstack((f0, f1))

    spw1_array, mapping = build_array(interpolated_data, category='Intensity')
    spw0_array = build_array(interpolated_data, category='Intensity', spw=0, return_log=False)
    intensity_array = np.hstack((spw0_array, spw1_array))

    residual_array_spw1 = build_array(interpolated_data, category='Residual', return_log=False)
    residual_array = np.hstack((build_array(interpolated_data, category='Residual', return_log=False, spw=0), residual_array_spw1))

    # Shift signals
    data_info = pd.read_csv('Data/data_info.csv')
    best_v = data_info['Best velocity 2'].values

    shifted_signals = np.array([shift_signal(freq, intensity_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_residual = np.array([shift_signal(freq, residual_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_mapping = [mapping[i] for i in range(len(mapping)) if not np.isnan(best_v[i])]

    ## Filter data
    sigma_signal = filter_data(np.asarray(shifted_signals), 'sigma', shifted_residual)
    subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    savgol_signal = filter_data(np.asarray(shifted_signals), 'savgol')

    # Find old data index
    shifted_mapping = [shifted_mapping[i][:-1] for i in range(len(shifted_mapping))]

    # Train AC algorithm
    ac = AgglomerativeClustering(k_neighbors=100, k0_neighbors=1, distance_metric='cosine')
    ac.fit(subtraction_signal)

    """
    for i in range(len(ac.label_history)):
    labels = ac.label_history[i]
    n_clusters = len(np.unique(labels))
    
    fig, ax = plt.subplots(n_clusters, 1, sharex=True, figsize=(10, 6))
    for j in np.unique(labels):
        ax[j].plot(freq, subtraction_signal[labels == j].mean(axis=0), label=f'{sum(labels==j)} signals')
        ax[j].legend()
    
    fig.supxlabel('Frequency [MHz]', fontsize=14)
    fig.supylabel('Intensity [Jy]', fontsize=14)
    plt.tight_layout()
    plt.show()
    """