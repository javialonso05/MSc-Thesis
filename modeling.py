import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from umap import UMAP
from seaborn import kdeplot
from scipy.stats import gaussian_kde


def plot_kde_gmm(data: np.ndarray, 
                 n_components: int = None, 
                 gmm: GaussianMixture = None, 
                 ax: plt.axes = None,
                 tol: float = 0.0,
                 labels: list = None,
                 label_probability: np.ndarray = None,
                 bandwidth = None,
                 gridsize: int = 100,
                 save_plot: str = False):
    """
    Plot Gaussian Mixtures in a 2D representation of the data over a kde of the density of points

    Args:
        data (_type_): _description_
        n_components (int, optional): _description_. Defaults to None.
        gmm (GaussianMixture, optional): __
        ax (_type_, optional): _description_. Defaults to None.
        tol (float, optional): Minimum probability of a point belonging to its class for it to be plotted. Defaults to 0.
        labels (list, optional): __
        label_probability (): __
        bandwidth (_type_, optional): _description_. Defaults to None.
        gridsize (int, optional): _description_. Defaults to 100.
        save_plot (str, optional): Flag for saving the plot. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Check inputs
    if data.shape[1] != 2:
        raise ValueError("Input data must be of shape (N,2)")
    
    if n_components is None and gmm is None and labels is None:
        raise ValueError("Either n_components, gmm or labels need to be supplied as inputs")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if labels is None:
        if n_components is None:
            n_components = gmm.n_components
            mask = np.max(gmm.predict_proba(data), axis=1) > tol
            labels = gmm.predict(data[mask])
        elif gmm is None and n_components > 0:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data)
            mask = np.max(gmm.predict_proba(data), axis=1) > tol
            labels = gmm.predict(data[mask])
        elif gmm is None and n_components == 0:
            mask = np.ones(len(data), dtype=bool)
            labels = np.ones(len(data))
    else:
        if label_probability is None:
            label_probability = np.ones((len(labels), 1))
        mask = np.max(label_probability, axis=1) > tol
        n_components = 0

    x, y = data[:, 0], data[:, 1]

    # Perform kernel density estimation
    kde = gaussian_kde([x, y], bw_method=bandwidth)

    # Create grid for evaluation
    xmin, xmax = x.min() - 0.1, x.max() + 0.1
    ymin, ymax = y.min() - 0.1, y.max() + 0.1
    X, Y = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    XY_grid = np.column_stack([X.ravel(), Y.ravel()])

    # # Evaluate KDE on the grid
    Z_kernel = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    if n_components > 0:
        Z = np.exp(gmm.score_samples(XY_grid))
        Z = Z.reshape(X.shape)

    # Plot result
    colors = plt.get_cmap("Set1")(labels[mask])
    ax.pcolormesh(X, Y, Z_kernel, shading='auto', cmap='viridis')
    print(len(x), len(y), len(colors))
    scatter = ax.scatter(x[mask], y[mask], color=colors, s=30, alpha=0.6)

    if n_components > 0:
        ax.contour(X, Y, Z, levels=10, cmap='coolwarm')  # GMM results

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # ax.add_artist(legend1)
    if save_plot is not None:
        fig.savefig(save_plot)


# Load data
freq = np.load("Data/frequencies.npy")
subtraction_signals = np.load("Data/Processed/subtraction_signals.npy")
subtraction_signals /= np.max(subtraction_signals, keepdims=True, axis=1)

# Reduce dimensionality
# pca_data = PCA(n_components=30, random_state=42).fit_transform(subtraction_signals)
# cosine_umap = UMAP(n_neighbors=5, metric='cosine', random_state=42).fit_transform(subtraction_signals)
# euclidean_umap = UMAP(n_neighbors=5, metric='euclidean', random_state=42).fit_transform(subtraction_signals)

cosine_umap = np.load("Data/Processed/cosine_umap.npy")
euclidean_umap = np.load("Data/Processed/euclidean_umap.npy")

# Save datasets
# np.save("Data/Processed/cosine_umap.npy", cosine_umap)
# np.save("Data/Processed/euclidean_umap.npy", euclidean_umap)

# Identify outlier groups and match them in the cosine signal
g1_mask = (euclidean_umap[:, 0] > 8) * (euclidean_umap[:, 1] > 2)
g6_mask = (2.3 < cosine_umap[:, 0]) * (cosine_umap[:, 0] < 2.7) * (8.6 < cosine_umap[:, 1]) * (cosine_umap[:, 1] < 9.2)
g7_mask = (-3.2 < euclidean_umap[:, 0]) * (euclidean_umap[:, 0] < -2.8) * (4.1 < euclidean_umap[:, 1]) * (euclidean_umap[:, 1] < 4.5)
outlier_mask = g1_mask + g6_mask + g7_mask

cmap = plt.get_cmap("Set1")
colors = cmap(np.unique(labels))
fig, ax = plt.subplots(5, 1, sharex=True, figsize=(12, 6))
for i in range(5):
    ax[i].plot(freq, subtraction_signals[~outlier_mask][labels == i], color=colors[i], label=f'{sum(labels == i)} signals')
    ax[i].legend(loc='upper left')

fig.supxlabel("Frequency [MHz]", fontsize=14)
fig.supylabel("Intensity [Jy]", fontsize=14)
fig.tight_layout()
fig.savefig("results/gmm/cosine_gmm_5_centroids.pdf")
plt.show()
    

# Train model
for n in range(4, 10):
    cosine_model = GaussianMixture(n_components=n, random_state=42).fit(cosine_umap[~outlier_mask])
    plot_kde_gmm(cosine_umap[~outlier_mask], gmm=cosine_model, save_plot="results/gmm/euclidean_8_clusters.pdf")
    plt.show()

# Retrieve labels and probabilities
labels = cosine_model.predict(cosine_umap[~outlier_mask])
label_prob = cosine_model.predict_proba(euclidean_umap[~outlier_mask])

# Calculate and plot means
means = np.array([subtraction_signals[~outlier_mask][labels == i].mean(axis=0) for i in range(8)])

cmap = plt.get_cmap("Set1")
colors = cmap(np.unique(labels))



# fig, ax = plt.subplots(8, 1, sharex=True, figsize=(12, 8))
# for i in range(8):
#     ax[i].plot(freq, means[i], color=colors[i], label=f"{sum(labels == i)} signals")
#     ax[i].legend(loc='upper left')

# fig.supxlabel("Frequency [MHz]", fontsize=14)
# fig.supylabel("Intensity [Jy]", fontsize=14)
# fig.tight_layout()
# fig.savefig("results/gmm/euclidean_gmm_centroids.pdf")
# plt.show()

# Merge clusters based on cosine distance
# dist = cosine_distances(means, means)
# plt.matshow(dist)
# for i in range(len(means)):
#     for j in range(len(means)):
#         plt.text(i, j, str(round(dist[i, j], 2)), va='center', ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
# plt.colorbar()
# plt.show()

# dist += np.eye(8)
# while np.min(dist) < 0.1:
#     # Find labels to merge
#     label_1, label_2 = np.where(dist == np.min(dist))
    
#     # Select just one if two signals are at the same distance
#     if len(label_1) > 1:
#         label_1 = label_1[0]
#     if len(label_2) > 1:
#         label_2 = label_2[0]

#     # Plot merged signals
#     fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
    
#     ax[0].plot(freq, subtraction_signals[~outlier_mask][labels == label_1].mean(axis=0))
#     ax[1].plot(freq, subtraction_signals[~outlier_mask][labels == label_2].mean(axis=0))
    
#     fig.suptitle(f"Cosine distance: {np.min(dist):.3f}")
#     fig.supxlabel("Frequency [MHz]", fontsize=14)
#     fig.supylabel("Intensity [Jy]", fontsize=14)
#     fig.tight_layout()
#     plt.show()

#     # Merge
#     labels[labels == label_1] = label_2
#     label_prob[:, label_2] += label_prob[:, label_1]
#     label_prob = np.delete(label_prob, label_1, axis=1)
    
#     # Reset labels
#     for i, label in enumerate(np.unique(labels)):
#         labels[labels == label] = i

#     # Compute new distances
#     means = np.array([subtraction_signals[~outlier_mask][labels == i].mean(axis=0) for i in range(len(np.unique(labels)))])
#     dist = cosine_distances(means, means) + np.eye(len(means))

# labels[labels == 3] = 1
# labels[labels == 7] = 1
# label_prob[:, 1] += label_prob[:, 3] + label_prob[:, 7]
# label_prob = np.delete(label_prob, [3, 7], axis=1)
# for i, label in enumerate(np.unique(labels)):
#     labels[labels == label] = i

# # Plot final arrangement
# plot_kde_gmm(euclidean_umap[~outlier_mask], labels=labels, label_probability=label_prob, tol=0.9, save_plot="results/gmm/final_clusters_umap.pdf")
# plt.show()


# prob_mask = np.max(label_prob, axis=1) > 0.9
# means = np.array([subtraction_signals[~outlier_mask][prob_mask][labels[prob_mask] == label].mean(axis=0) for label in np.unique(labels)])

# fig, ax = plt.subplots(len(means) + 3, 1, sharex=True, figsize=(12, 6))

# for i in range(len(means)):
#     ax[i].plot(freq, means[i], label=f"{sum(labels[prob_mask]==i)} signals", color=cmap.colors[i])
#     ax[i].legend(loc="upper left")

# ax[-3].plot(freq, subtraction_signals[g1_mask].mean(axis=0), color=cmap.colors[i+1], label=f"{sum(g1_mask)} signals")
# ax[-3].legend(loc="upper left")

# ax[-2].plot(freq, subtraction_signals[g6_mask].mean(axis=0), color=cmap.colors[i+2], label=f"{sum(g1_mask)} signals")
# ax[-2].legend(loc="upper left")

# ax[-1].plot(freq, subtraction_signals[g7_mask].mean(axis=0), color=cmap.colors[i+3], label=f"{sum(g1_mask)} signals")
# ax[-1].legend(loc="upper left")

# fig.supxlabel("Frequency [MHz]", fontsize=14)
# fig.supylabel("Intensity [Jy]", fontsize=14)
# fig.tight_layout()
# fig.savefig("results/gmm/final_high_likelihood_centroids.pdf")
# plt.show()

# copy_array = subtraction_signals[~outlier_mask][prob_mask].copy()
# copy_array[copy_array < -2] = -2
# plt.figure(figsize=(10, 4))

# plt.plot(freq, copy_array[labels[prob_mask] == 2].T, color='grey', alpha=0.3)
# plt.plot(freq, means[2], color=cmap.colors[2])

# plt.xlabel("Frequency [MHz]", fontsize=14)
# plt.ylabel("Intensity [Jy]", fontsize=14)
# plt.tight_layout()
# plt.savefig("results/gmm/cluster_2_signals.pdf")
# plt.show()
