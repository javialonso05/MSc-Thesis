import numpy as np


def shift_signal(frequency, signal, velocity):
    """
    Shift a signal by a given velocity
    :param frequency: 1D array with the frequency channels
    :param signal: 1D array with the intensities
    :param velocity: source velocity in km/s
    :return: shifted_signal, shifted_frequency
    """

    c = 3e5  # km/s
    shifted_frequency = frequency * (1 - velocity / c)
    shifted_signal = np.interp(shifted_frequency, frequency, signal, left=0, right=0)
    return shifted_signal


def sigma_filter(intensity_array: np.ndarray, residual_array: np.ndarray, threshold: int = 3):
    """
    Filter the signal by removing all data-points smaller than a given threshold based on the residual
    :param intensity_array:
    :param residual_array:
    :param threshold:
    :return: filtered_data
    """

    # Check input
    if np.shape(intensity_array) != np.shape(residual_array):
        raise ValueError('Shapes of intensity and residual must be equal')


    sigma = np.std(intensity_array, axis=1)
    filtered_data = np.zeros_like(intensity_array)
    for i in range(len(sigma)):
        signal = np.array(intensity_array[i])
        signal[signal < threshold * sigma[i]] = 0

        while np.all(signal == 0):
            threshold -= 0.1
            signal = np.array(intensity_array[i])
            signal[signal < threshold * sigma[i]] = 0

        filtered_data[i] = signal

    return filtered_data


def savgol_filter(data_array: np.ndarray, window_length: int = 200, polyorder: int = 3):
    """
    Use a Savitzky-Golay filter to smooth the signal
    :param data_array: data to smooth
    :param window_length: number of points to consider in the smoothing. Default is 25 based on Ishikawa et al. (2013)
    :param polyorder: order of the polynomial to use. Default is 3.
    :return: filtered_data
    """

    from scipy.signal import savgol_filter

    filtered_signal = savgol_filter(data_array, window_length, polyorder)
    return filtered_signal
    
