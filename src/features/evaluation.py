import matplotlib.pyplot as plt
import numpy as np


# TODO: review this function
def within_cluster_similarity(data: np.ndarray, labels: list,
                              std: bool = False,
                              plot: bool = False,
                              title: str = None) -> np.ndarray:
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
    from sklearn.metrics.pairwise import cosine_similarity
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
        # Plot the within-cluster similarity
        width = 0.8
        plt.figure(figsize=(8, 4))
        if std:
            plt.bar(np.arange(len(clusters)), within_cluster_similarity, width, yerr=within_cluster_std,
                    color='tab:blue')
        else:
            plt.bar(np.arange(len(clusters)), within_cluster_similarity, width, color='tab:blue')
        plt.xlabel('Cluster')
        plt.ylabel('Cosine Similarity')
        plt.title('Within-Cluster Similarity') if title is None else plt.title(title)

        # Add percentage of signals that belong to each cluster
        total_signals = len(labels)
        percentages = [round((np.sum(labels == cluster) / total_signals) * 100) for cluster in clusters]
        for i, percentage in enumerate(percentages):
            if not std:
                plt.text(i, within_cluster_similarity[i], f'{percentage}%', ha='center', va='bottom')
            else:
                plt.text(i - width / 4, within_cluster_similarity[i], f'{percentage}%', ha='center', va='bottom')

        plt.show()

    if std:
        return within_cluster_similarity, within_cluster_std

    return within_cluster_similarity
