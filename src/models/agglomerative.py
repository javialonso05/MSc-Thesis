import numpy as np
from sklearn.metrics import silhouette_score
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
            len_i = len(idx_i)
            for j in range(n_clusters):
                if i == j:
                    continue

                idx_j = cluster_indices[j]
                len_j = len(idx_j)

                # Sparse submatrices
                W_ca = self.weights[idx_i][:, idx_j]
                W_cb = self.weights[idx_j][:, idx_i]

                # Compute row/column sums efficiently
                sum_W_ca = np.array(W_ca.sum(axis=0)).ravel()
                sum_W_cb = np.array(W_cb.sum(axis=0)).ravel()

                # Perform dot products
                term1 = sum_W_ca @ W_cb @ np.ones(len_i) / (len_i ** 2)
                term2 = sum_W_cb @ W_ca @ np.ones(len_j) / (len_j ** 2)

                self.cluster_affinity_matrix[i, j] = term1 + term2

    def _update_affinity(self, a, b):
        """
        Update the affinity matrix - Call only after updating self.labels_
        :param a: Label of the first merged cluster
        :param b: Label of the second merged cluster
        :return:
        """

        # Generate new affinity matrix
        new_affinity = np.zeros((len(self.cluster_affinity_matrix - 1), len(self.cluster_affinity_matrix - 1)))

        # Input the affinity of the unmerged clusters
        new_affinity[:-1, :-1] = np.delete(np.delete(self.cluster_affinity_matrix, [a, b], axis=0), [a, b], axis=1)

        # A_(ab->c) = A_(a->c) + A_(b->c)
        for i in range(len(self.cluster_affinity_matrix)):
            if a != i and b != i:
                new_affinity[-1][i] = self.cluster_affinity_matrix[a][i] + self.cluster_affinity_matrix[b][i]

        # Calculate A_(c->ab)
        cluster_indices = [np.where(np.array(self.labels_) == i)[0] for i in range(len(new_affinity))]
        idx_j = cluster_indices[-1]
        len_j = len(idx_j)
        for i in range(len(new_affinity)):
            idx_i = cluster_indices[i]
            len_i = len(idx_i)

            # Sparse submatrices
            W_ca = self.weights[idx_i][:, idx_j]
            W_cb = self.weights[idx_j][:, idx_i]

            # Compute row/column sums efficiently
            sum_W_ca = np.array(W_ca.sum(axis=0)).ravel()
            sum_W_cb = np.array(W_cb.sum(axis=0)).ravel()

            # Perform dot products
            term1 = sum_W_ca @ W_cb @ np.ones(len_i) / (len_i ** 2)
            term2 = sum_W_cb @ W_ca @ np.ones(len_j) / (len_j ** 2)

            new_affinity[i, -1] = term1 + term2

        self.cluster_affinity_matrix = new_affinity

    def _compute_affinity(self, i, j, cluster_indices):
        """
        Calculate the affinity between clusters using indegree and outdegree
        :return:
        """

        idx_i = cluster_indices[i]
        len_i = len(idx_i)

        idx_j = cluster_indices[j]
        len_j = len(idx_j)

        # Sparse submatrices
        W_ca = self.weights[idx_i][:, idx_j]
        W_cb = self.weights[idx_j][:, idx_i]

        # Compute row/column sums efficiently
        sum_W_ca = np.array(W_ca.sum(axis=0)).ravel()
        sum_W_cb = np.array(W_cb.sum(axis=0)).ravel()

        # Perform dot products
        term1 = sum_W_ca @ W_cb @ np.ones(len_i) / (len_i ** 2)
        term2 = sum_W_cb @ W_ca @ np.ones(len_j) / (len_j ** 2)

        return i, j, term1 + term2


    def fit(self, data: np.ndarray):
        """
        Perform GDL clustering
        :param data: (array) shape (n_samples, n_features)
        :return:
        """
        from joblib import Parallel, delayed

        #TODO: improve time performance

        # Initial clustering
        self._initialize_clustering(data)

        clusters = np.unique(self.labels_).tolist()

        n_clusters = len(clusters)
        max_cluster = max(clusters)

        # Build k-NN graph
        self._build_weighted_knn_graph(data)
        for _ in tqdm(range(n_clusters - 1), desc='Merging clusters'):
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


if __name__ == '__main__':
    from src.data.data_processor import build_array, filter_data
    from src.features.transformations import shift_signal

    from sklearn.metrics import adjusted_rand_score
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
    # subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    # sigma_signal = filter_data(np.asarray(shifted_signals), 'sigma', shifted_residual)
    # savgol_signal = filter_data(np.asarray(shifted_signals), 'savgol')

    # Find old data index
    shifted_mapping = [shifted_mapping[i][:-1] for i in range(len(shifted_mapping))]

    # Train AC algorithm
    # TODO: batch training

    # for filter in [None, 'sigma', 'savgol']:
    #     if filter is not None:
    #         signals = filter_data(np.asarray(shifted_signals), filter, shifted_residual)
    #     else:
    #         signals = shifted_signals
    #
    #     ac = AgglomerativeClustering(k_neighbors=25, distance_metric='cosine')
    #     ac.fit(signals)
    #     with open(f'models/agglomerative/full_{filter}_25_neighbors.pkl', 'wb') as f:
    #         pickle.dump(ac, f)
    #
    # subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    # for spw in [0, 1]:
    #     ac = AgglomerativeClustering(k_neighbors=25, distance_metric='cosine')
    #     if spw == 0:
    #         ac.fit(subtraction_signal[:, :10000])
    #     else:
    #         ac.fit(subtraction_signal[:, 10000:])
    #
    #     with open(f'models/agglomerative/full_subtraction_25_neighbors_spw{spw}.pkl', 'wb') as f:
    #         pickle.dump(ac, f)



    # TODO: train on reduced dataset
    # for p in [10, 25, 50]:
    #     signals = subtraction_signal[np.random.randint(len(subtraction_signal), size=len(subtraction_signal)//p)]
    #     ac = AgglomerativeClustering(k_neighbors=25, k0_neighbors=1, distance_metric='cosine')
    #     ac.fit(signals)
    #
    #     with open(f'models/agglomerative/{p}pc_subtraction_model_25_neighbors.pkl', 'wb') as f:
    #         pickle.dump(ac, f)

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

    subtraction_model = pickle.load(open('models/agglomerative/full_ac_model_25_neighbors.pkl', 'rb'))
    raw_model = pickle.load(open('models/agglomerative/full_None_25_neighbors.pkl', 'rb'))
    sigma_model = pickle.load(open('models/agglomerative/full_sigma_25_neighbors.pkl', 'rb'))
    savgol_model = pickle.load(open('models/agglomerative/full_savgol_25_neighbors.pkl', 'rb'))

    label_list = [subtraction_model.label_history,
                  sigma_model.label_history, savgol_model.label_history]
    coincidence = compare_clusterings(label_list, legend=['SU-SI', 'SU-SG', 'SI-SG'])

