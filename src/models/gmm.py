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

    scatter = ax.scatter(x, y, c=labels, cmap='tab10', s=30, alpha=0.6, label='Data Points')  # Data Points

    if n_components > 0:
        plt.contour(X, Y, Z, levels=10, cmap='coolwarm')  # GMM results

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    plt.show()
    if n_components > 0:
        return gmm

