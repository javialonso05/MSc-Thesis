import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.data.data_processor import build_array, filter_data
from src.features.transformations import shift_signal

if __name__ == '__main__':
    import pickle
    from tqdm import tqdm

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
    best_v = data_info['Best velocity'].values

    shifted_signals = np.array([shift_signal(freq, intensity_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_residual = np.array([shift_signal(freq, residual_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_mapping = [mapping[i] for i in range(len(mapping)) if not np.isnan(best_v[i])]

    ## Filter data
    sigma_signal = filter_data(np.asarray(shifted_signals), 'sigma', shifted_residual)
    subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    savgol_signal = filter_data(np.asarray(shifted_signals), 'savgol')

    # PCA decomposition
    pca = PCA(n_components=30, random_state=42)
    pca_data = pca.fit_transform(shifted_signals)

    # UMAP decomposition
    up = umap.UMAP(n_neighbors=5, min_dist=0, random_state=42, metric='cosine')
    umap_data = up.fit_transform(shifted_signals)

    s_score = []
    for n in tqdm(range(2, 13)):
        score = []

        # PCA
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(pca_data)

        prob = gmm.predict_proba(pca_data).max(axis=1)
        high_confidence_mask = prob > 0.9
        score.append(silhouette_score(shifted_signals[high_confidence_mask],
                                      gmm.predict(pca_data[high_confidence_mask]), metric='cosine'))

        # UMAP
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(umap_data)

        prob = gmm.predict_proba(umap_data).max(axis=1)
        high_confidence_mask = prob > 0.9
        score.append(silhouette_score(shifted_signals[high_confidence_mask],
                                      gmm.predict(umap_data[high_confidence_mask]), metric='cosine'))

        s_score.append(score)

    s_score = np.array(s_score)

    plt.plot(range(2, 13), s_score, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Silhouette score')
    plt.xticks(ticks=range(2, 13, 2))
    plt.legend(['PCA', 'UMAP'])
    plt.tight_layout()
    plt.show()
