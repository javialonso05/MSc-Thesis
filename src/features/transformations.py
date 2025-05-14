import numpy as np


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors or a vector and a matrix
    :param a: 1D vector
    :param b: 1D vector or 2D matrix
    :return: cosine similarity
    """
    if len(np.shape(a)) != 1 and len(np.shape(b)) != 1:
        raise ValueError("At least one input must be a 1D vector")

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def shift_signal(frequency, signal, velocity) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift a signal by a given velocity
    :param frequency: 1D array with the frequency channels
    :param signal: 1D array with the intensities
    :param velocity: source velocity in km/s
    :return: shifted_signal, shifted_frequency
    """

    c = 5e5
    shifted_frequency = frequency * (1 - velocity / c)
    shifted_signal = np.interp(shifted_frequency, frequency, signal, left=0, right=0)
    return shifted_signal, shifted_frequency
