import numpy as np
from pandas.core.config_init import styler_sparse_columns_doc
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import permutations

import umap
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin

from src.features.evaluation import within_cluster_similarity
from src.data.data_processor import filter_data
from src.features.transformations import transform_signal


class CustomKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, n_init=5, max_iter=100,
                 tol=1e-4, random_state=42, distance_metric=None):
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



def compound_training(classifier: CustomKMeans, signals, classifier_2: CustomKMeans, mask = None):
    """
    Train a k-means classifier in 2 steps: first with the entire signal, then sub-cluster after applying a mask
    :param classifier:
    :param signals:
    :param classifier_2:
    :param n_clusters:
    :param mask:
    :return:
    """

    km = classifier.fit(signals)
    masked_signals = np.asarray(signals)
    if mask is not None:
        for i in range(len(masked_signals)):
            masked_signals[i][mask] = 0

    labels = km.labels_
    for cluster in np.unique(labels):
        cluster_signals = masked_signals[labels == cluster]

        sub_km = classifier_2.fit(cluster_signals)
        labels[labels == cluster] = sub_km.labels_ + cluster

    return labels


if __name__ == '__main__':
    import pickle
    import pandas as pd
    from src.data.data_processor import build_array
    from src.features.transformations import shift_signal
    from sklearn.metrics import silhouette_score

    # Load signals
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
    best_v = data_info['Best velocity'].values

    shifted_signals = np.array([shift_signal(freq, intensity_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_residual = np.array([shift_signal(freq, residual_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_mapping = [mapping[i] for i in range(len(mapping)) if not np.isnan(best_v[i])]

    ## Filter data
    sigma_signal = filter_data(np.asarray(shifted_signals), 'sigma', shifted_residual)
    subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    savgol_signal = filter_data(np.asarray(shifted_signals), 'savgol')

    ## Save data
    # np.save('Data/Processed/filtered_signals.npy', np.dstack((shifted_signals, sigma_signal, subtraction_signal, savgol_signal)))
    # np.save('Data/Processed/shifted_residual.npy', shifted_residual)

    ### Load data
    # shifted_signals, sigma_signal, subtraction_signal, savgol_signal = np.load('Data/Processed/filtered_signals.npy')


    s_score = []
    for signals in tqdm([shifted_signals[:, 10000:], sigma_signal[:, 10000:], subtraction_signal[:, 10000:], savgol_signal[:, 10000:]],
                           desc=f'Clustering signals'):
        score = []
        for n in range(2, 21):
            norm_signals = signals / np.max(signals, axis=1).reshape(-1, 1)
            km = KMeans(n_clusters=n).fit(norm_signals)
            score.append(silhouette_score(shifted_signals[:, 10000:], km.labels_, metric='cosine'))

        s_score.append(score)

    s_score = np.array(s_score)
    plt.plot(range(2, 21), s_score.T, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.legend(['None', 'Sigma', 'Subtraction', 'SavGol'], title='Filter type')
    plt.xticks(ticks=range(2, 21, 2))
    plt.show()

    # Masking major peaks
    mask = np.zeros_like(f1, dtype=bool)
    mask[(219535 < f1) * (f1 < 219585)] = True  # C18O
    mask[(219925 < f1) * (f1 < 219975)] = True  # SO
    mask[(220375 < f1) * (f1 < 220425)] = True  # 13CO

    # models = []
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    # signal_names = ['Raw', 'Sigma-filtered', 'Subtraction-filtered', 'Savgol-filtered']
    # for k, signals in tqdm(enumerate([shifted_signals[:, 10000:], sigma_signal[:, 10000:],
    #                                   subtraction_signal[:, 10000:], savgol_signal[:, 10000:]]),
    #                        desc=f'Clustering signals'):
    #
    #     # Train classifier normally
    #     km = CustomKMeans(n_clusters=5)
    #     km.fit(signals)
    #
    #     # Plot centroids
    #     fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    #     for i in range(4):
    #         ax[i].plot(f1, km.cluster_centers_[i], color=colors[k],
    #                    label=f'{sum(km.labels_ == i)} signals')
    #         ax[i].legend()
    #     fig.supxlabel('Frequency [MHz]')
    #     fig.supylabel('Intensity [Jy]')
    #     fig.suptitle(f'{signal_names[k]} centroids')
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # Subdivide the signals and re-train the classifier
    #     s_score = []
    #     model_list = []
    #     for i in range(4):
    #         cluster_signals = signals[km.labels_ == i]
    #
    #         # Mask the main peaks
    #         for j in range(len(cluster_signals)):
    #             cluster_signals[j][mask] = 0
    #
    #         # Iterate over the number of clusters
    #         score = []
    #         sub_model_list = []
    #         for j in range(2, 15):
    #             km_2 = CustomKMeans(n_clusters=j)
    #             km_2.fit(cluster_signals)
    #             sub_model_list.append(km_2)
    #
    #             score.append(silhouette_score(cluster_signals, km_2.labels_, metric='cosine'))
    #
    #         s_score.append(score)
    #         model_list.append(sub_model_list)
    #
    #     s_score = np.array(s_score)
    #     models.append(model_list)
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(2, 15), s_score.T, marker='o')
    #     plt.legend(['C0', 'C1', 'C2', 'C3'])
    #     plt.xlabel('Number of clusters')
    #     plt.ylabel('Silhouette score')
    #     plt.tight_layout()
    #     plt.show()

