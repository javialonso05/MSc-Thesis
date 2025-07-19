import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


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


def piecewise_cosine_distance(data,
                              window=None,
                              stride=None,
                              norm_type='max'):
    """
    Custome distance metric that evaluates the cosine distance at several windows
    :param data:
    :param window:
    :param stride:
    :param norm_type:
    :return:
    """
    # Assign default values
    if window is None:
        window = len(data[0]) // 4
    if stride is None:
        stride = window // 2
    # Check inputs
    if window > data.shape[1]:
        raise ValueError('Window must be smaller than the signal length')

    # Normalize inputs
    if norm_type is not None:
        if norm_type == 'max':
            data /= np.max(data, axis=1).reshape(-1, 1)
        elif norm_type == 'int':
            data /= np.linalg.norm(data, axis=1).reshape(-1, 1)
        elif norm_type == '13CO':
            data = np.array([data[i] / data[i][16993] for i in range(len(data))])
        else:
            raise ValueError(f'norm_type "{norm_type}" not recognized"')


    # Start comparison
    distance = None
    weights = None
    for idx in tqdm(range(0, data.shape[1] - window, stride)):
        signals = data[:, idx:idx + window]
        dist = 1 - cosine_similarity(signals, signals)
        w0 = np.tile(np.linalg.norm(signals, axis=1), (len(dist), 1))

        # distance.append(dist)
        # # weights.append(np.minimum(w0, w0.T))
        # weights.append(w0 + w0.T)

        if distance is None:
            distance = dist * (w0 + w0.T)
            weights = w0 + w0.T
        else:
            distance += dist * (w0 + w0.T)
            weights += w0 + w0.T

    distance /= weights
    # similarity = np.array(distance)
    # weights = np.array(weights)
    #
    # dist = np.zeros_like(dist)
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         dist[i, j] = np.average(similarity[:, i, j], weights=weights[:, i, j])

    return distance


def test_piecewise_noise_sensitivity(test_signal):
    """

    :param test_signal:
    :return:
    """
    cosine_dist = []
    piecewise_dist = []
    for s in np.linspace(0, np.max(test_signal), 1000):
        noisy_signal = test_signal + s * np.random.rand(len(test_signal))
        cosine_dist.append(1 - cosine_similarity(test_signal.reshape(1, -1),
                                                 noisy_signal.reshape(1, -1))[0][0])
        piecewise_dist.append(piecewise_cosine_distance(np.vstack((test_signal, noisy_signal)), window=500, stride=250)[0, 1])

    return cosine_dist, piecewise_dist


def test_window_size(test_signals):
    cosine_dist = 1 - cosine_similarity(test_signals[0].reshape(1, -1),
                                        test_signals[1].reshape(1, -1))[0]
    piecewise_dist = []
    for window in np.arange(250, 5001, 50):
        piecewise_dist.append(piecewise_cosine_distance(test_signals, window=window, stride=250)[0, 1])

    plt.plot(np.arange(250, 5001, 50), piecewise_dist)
    plt.axhline(cosine_dist, linestyle='--', color='black')
    plt.xlabel('Window size')
    plt.ylabel('Distance')
    plt.show()

    return cosine_dist, piecewise_dist


def clustering_overlap_score(labels1, labels2):
    """
    Computes the best label alignment between two clustering results
    and returns the fraction of points that are assigned the same label.
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    assert labels1.shape == labels2.shape, "Label arrays must be the same length"
    # Compute the confusion matrix
    cm = confusion_matrix(labels1, labels2)
    # Use the Hungarian algorithm to find the best alignment
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize total matches
    # Total correctly matched labels
    matched = cm[row_ind, col_ind].sum()
    return matched / len(labels1)


def overlapping_indices(labels1, labels2):
    """
    Returns the indices where the optimally permuted labels2 matches labels1.
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    assert labels1.shape == labels2.shape, "Label arrays must be the same length"
    # Compute the confusion matrix
    cm = confusion_matrix(labels1, labels2)

    # Find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Build a mapping from labels2 to labels1
    label_map = {col: row for row, col in zip(row_ind, col_ind)}

    # Apply the mapping to labels2
    aligned_labels2 = np.array([label_map[label] for label in labels2])

    # Find indices where aligned labels match
    overlap_idx = np.where(aligned_labels2 == labels1)[0]

    return overlap_idx.tolist()


if __name__ == '__main__':
    from src.data.data_processor import build_array, filter_data
    from src.features.transformations import shift_signal

    import pickle
    import pandas as pd
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
    best_v = data_info['Best velocity 2'].values

    shifted_signals = np.array([shift_signal(freq, intensity_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_residual = np.array([shift_signal(freq, residual_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    shifted_mapping = [mapping[i] for i in range(len(mapping)) if not np.isnan(best_v[i])]

    # Filter data
    subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)

    # Test functions
    signals = np.vstack((subtraction_signal[5], subtraction_signal[7]))
    cosine_dis = 1 - cosine_similarity(signals[0].reshape(1, -1), signals[1].reshape(1, -1))[0]
    values = []
    for norm in ['max', 'int']:
        norm_values = []
        for window_size in tqdm(range(500, 5000, 500)):
            stride_length = window_size // 2
            psim = piecewise_cosine_distance(signals, window=window_size, stride=stride_length, norm_type=norm)
            norm_values.append(psim[0, 1])
        values.append(norm_values)

    plt.plot(range(500, 5000, 500), np.array(values).T)
    plt.axhline(cosine_dis, linestyle='--', color='black')
    plt.legend(['max', 'int', 'cosine'])
    plt.xlabel('Window size')
    plt.ylabel('Distance')
    plt.show()

