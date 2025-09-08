import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.agglomerative import CustomAgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering


def between_cluster_similarity(signals, labels, plot=False):
    """

    Args:
        signals (_type_): _description_
        labels (_type_): _description_
        plot (bool, optional): _description_. Defaults to False.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    unique_labels = np.unique(labels)
    bcs = np.zeros((len(unique_labels), len(unique_labels)))
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i == j:
                continue
            signals_i = signals[labels == label_i]
            signals_j = signals[labels == label_j]
            bcs[i, j] = np.mean(cosine_similarity(signals_i, signals_j))

    if plot:
        plt.matshow(bcs)
        for i in range(len(bcs)):
            for j in range(len(bcs)):
                plt.text(i, j, np.round(bcs[i, j], 2), ha="center", va="center", color="white")
        plt.show()
    return bcs

# Load data
cosine_umap = np.load("Data\\Processed\\cosine_umap.npy")
euclidean_umap = np.load("Data\\Processed\\euclidean_umap.npy")
subtraction_signals = np.load("Data\\Processed\\subtraction_signals.npy")
freq = np.load("Data\\frequencies.npy")

