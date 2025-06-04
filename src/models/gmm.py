import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.data.data_processor import build_array, filter_data
from src.features.transformations import shift_signal


def plot_kde_gmm(data, bandwidth=None, gridsize=100, n_components=3):
    if data.shape[1] != 2:
        raise ValueError("Input data must be of shape (N,2)")

    x, y = data[:, 0], data[:, 1]

    # Perform kernel density estimation
    kde = gaussian_kde([x, y], bw_method=bandwidth)

    # Fit Gaussian Mixture Model (GMM)
    if n_components > 0:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(data)
    else:
        labels = np.ones(len(data))

    # Create grid for evaluation
    xmin, xmax = x.min() - 0.1, x.max() + 0.1
    ymin, ymax = y.min() - 0.1, y.max() + 0.1
    X, Y = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    XY_grid = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate KDE on the grid
    Z_kernel = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    if n_components > 0:
        Z = np.exp(gmm.score_samples(XY_grid))
        Z = Z.reshape(X.shape)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.pcolormesh(X, Y, Z_kernel, shading='auto', cmap='viridis')
    plt.colorbar(label='Density')

    scatter = ax.scatter(x, y, c=labels, cmap='tab10', s=30, alpha=0.6, label='Data Points')  # Data Points

    if n_components > 0:
        plt.contour(X, Y, Z, levels=10, cmap='coolwarm')  # GMM results

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig.suptitle("Kernel Density Estimation with GMM Classification")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    plt.show()
    if n_components > 0:
        return gmm


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

    # Find old data index
    shifted_mapping = [shifted_mapping[i][:-1] for i in range(len(shifted_mapping))]
    old_data_idx = np.load('Data/old_data_idx.npy')

    # Perform UMAP reduction with the old data
    import umap

    up = umap.UMAP(n_neighbors=5, metric='cosine', min_dist=0, random_state=42).fit(subtraction_signal[old_data_idx])
    old_umap_data = up.transform(subtraction_signal[old_data_idx])
    new_umap_data = up.transform(subtraction_signal)

    wide_signals = [['126223', 'core7'],
['586092', 'core39'],
['586092', 'core65'],
['641469', 'core14'],
['644284', 'core79'],
['667557', 'core74'],
['737588', 'core31'],
['801753', 'core41'],
['801753', 'core46'],
['801753', 'core51'],
['801753', 'core55'],
['801753', 'core56'],
['801753', 'core59'],
['801753', 'core60'],
['801753', 'core61'],
['801753', 'core64'],
['801753', 'core66'],
['801753', 'core68'],
['801753', 'core70'],
['G337.4050-00.4071A', 'core23']]
    wide_idx = [i for i in range(len(shifted_mapping)) if shifted_mapping[i] in wide_signals]


    # Remove very far-away clusters
    umap_data = new_umap_data[np.where(new_umap_data[:, 0] > 0)[0]]

    gmm = plot_kde_gmm(umap_data, n_components=7)

    prob = gmm.predict_proba(umap_data)
    labels = []
    for i in range(len(prob)):
        p = prob[i]

        if p[2] > 0.9:
            labels.append(0)
        elif p[4] > 0.9:
            labels.append(1)
        elif p[5] > 0.9:
            labels.append(2)
        elif sum(p[0, 1, 3, 6]) > 0.9:
            labels.append(3)
        else:
            labels.append(np.nan)

    hp_labels = labels[~np.isnan(labels)]
    hp_signals = subtraction_signal[np.where(new_umap_data[:, 0] > 0)[0]][~np.isnan(labels)]

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 6))

    cmap = plt.get_cmap('tab10')
    for i in range(4):
        ax[i].plot(freq, hp_signals[hp_labels==i].mean(axis=0), label=f'{hp_labels==i} signals')
        ax[i].legend()

    fig.supxlabel('Frequency [MHz]', fontsize=14)
    fig.supylabel('Intensity [Jy]', fontsize=14)
    plt.tight_layout()
    plt.show()
