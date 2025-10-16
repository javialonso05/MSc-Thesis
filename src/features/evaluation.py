import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def within_cluster_similarity(data: np.ndarray, labels: list,
                              std: bool = False,
                              plot: bool = False,
                              title: str = None, 
                              save: str = None) -> np.ndarray:
    """
    Evaluate the within-cluster similarity of the data
    :param data: 2D numpy array containing the data for each signal
    :param labels: list containing the cluster labels_ for each signal
    :param batch_size: size of the batches that will be compared within each cluster
    :param std: boolean to calculate or not the standard deviation of the within-cluster similarity
    :param plot: boolean to plot the within-cluster similarity of each cluster in a bar plot
    :param title: string with the title of the plot
    :return: array containing the within-cluster similarity for each signal. If the cluster only contains one signal,
    the value is set to np.nan
    """
    
    if len(data) != len(labels):
        raise ValueError('Data and labels_ must have the same length')

    # Obtain unique labels and counts
    within_cluster_similarity = np.array([])
    within_cluster_std = np.array([])
    clusters, sizes = np.unique(labels, return_counts=True)

    # Loop over all different clusters
    for i, cluster in enumerate(clusters):
        # Append a similarity of 0 and continue if the cluster has only one element
        if sizes[i] == 1:
            within_cluster_similarity = np.append(within_cluster_similarity, 0)
            within_cluster_std = np.append(within_cluster_std, 0)
            continue

        # Initiate internal evaluation loop
        cluster_data = data[labels == cluster]
        cluster_similarity = cosine_similarity(cluster_data, cluster_data)
        cluster_similarity = cluster_similarity[~np.eye(cluster_similarity.shape[0], dtype=bool)].reshape(cluster_similarity.shape[0],-1)

        # Append the average similarity and its std for the cluster
        within_cluster_similarity = np.append(within_cluster_similarity, np.mean(cluster_similarity))
        within_cluster_std = np.append(within_cluster_std, np.std(cluster_similarity))


    if plot:
        # Calculate cluster size
        total_signals = len(labels)
        percentages = np.array([round((np.sum(labels == cluster) / total_signals) * 100) for cluster in clusters])
        
        # Sort labels by size
        sorted_wcs = within_cluster_similarity[np.argsort(sizes)[::-1]]
        sorted_percentages = percentages[np.argsort(sizes)[::-1]]
        
        # Calculate average wcs
        average_wcs = np.average(within_cluster_similarity, weights=sizes)
        
        # Plot the within-cluster similarity
        width = 0.8
        plt.figure(figsize=(8, 4))
        if std:
            plt.bar(np.arange(len(clusters)), sorted_wcs, width, yerr=within_cluster_std,
                    color='tab:blue')
        else:
            plt.bar(np.arange(len(clusters)), sorted_wcs, width, color='tab:blue')
        plt.xlabel('Cluster')
        plt.ylabel('Cosine Similarity')
        plt.axhline(average_wcs, linestyle='--', color='black')

        # Add percentage of signals that belong to each cluster
        for i, percentage in enumerate(sorted_percentages):
            if not std:
                plt.text(i, sorted_wcs[i], f'{percentage}%', ha='center', va='bottom')
            else:
                plt.text(i - width / 4, sorted_wcs[i], f'{percentage}%', ha='center', va='bottom')

        if save is not None:
            plt.savefig(save)

        plt.show()

    if std:
        return within_cluster_similarity[np.argsort(sizes)[::-1]], within_cluster_std[np.argsort(sizes)[::-1]]

    return within_cluster_similarity[np.argsort(sizes)[::-1]]


def between_cluster_similarity(
    signals: np.ndarray,
    labels: np.ndarray,
    wcs: np.ndarray = None,
    plot=False,
    text=False,
    save=None):
    """

    Args:
        signals (_type_): _description_
        labels (_type_): _description_
        plot (bool, optional): _description_. Defaults to False.
    """
    
    if wcs is None:
        wcs = np.ones(len(np.unique(labels)))
    
    unique_labels, sizes = np.unique(labels, return_counts=True)
    sorted_labels = unique_labels[np.argsort(sizes)[::-1]]
    bcs = np.zeros((len(unique_labels), len(unique_labels)))
    for i, label_i in enumerate(sorted_labels):
        for j, label_j in enumerate(sorted_labels):
            if i == j:
                continue
            signals_i = signals[labels == label_i]
            signals_j = signals[labels == label_j]
            bcs[i, j] = np.mean(cosine_similarity(signals_i, signals_j)) / wcs[i]

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(bcs, cmap='viridis', vmin=0, vmax=1, origin='upper')
        if text:
            for i in range(len(bcs)):
                for j in range(len(bcs)):
                    plt.text(j, i, np.round(bcs[i, j], 2), ha="center", va="center", color="white")
        else:
            plt.colorbar()
        
        if save is not None:
            plt.savefig(save, bbox_inches='tight')
        plt.show()
    return bcs

