import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.features.transformations import shift_signal, cosine_similarity


def load_data(base_dir: str):
    """
    Load data from the original file format into a dictionary with pandas dataframes
    :param base_dir: path to the folder with the extracted data
    :return: data_dict
    """

    # Create empty dictionary to hold the data
    data_dict = {}

    for folder_name in tqdm(os.listdir(base_dir), desc="Extracting data"):
        # Update folder path
        folder_path = os.path.join(base_dir, folder_name)

        # Check if there are any files within the folder
        if any(os.listdir(folder_path)):
            # Create a sub-dictionary for each region
            data_dict[folder_name] = {}

            # Iterate through each file to find the relevant data
            for file_name in os.listdir(folder_path):
                if file_name.endswith("mean.K.txt"):  # Temperature file
                    if file_name.startswith(f"{folder_name}_spw0"):  # Channel 0
                        # Retrieve core and create a dictionary for each core if it does not yet exist
                        core = next(item for item in file_name.split('.') if 'core' in item)
                        if core not in data_dict[folder_name]:
                            data_dict[folder_name][core] = {
                                # Channel: [[Freq], [Temp], [Intensity], [Residual]]
                                0: [[], [], [], []],
                                1: [[], [], [], []]
                            }

                        # Read temperature data
                        file_path = os.path.join(folder_path, file_name)
                        data = pd.read_csv(file_path, sep=' ', header=None, names=['Frequency', 'Temperature'])

                        # Update frequency and temperature lists
                        data_dict[folder_name][core][0][0] = list(data.values[:, 0])
                        data_dict[folder_name][core][0][1] = list(data.values[:, 1])

                    if file_name.startswith(f"{folder_name}_spw1"):  # Channel 1
                        # Retrieve core and create a dictionary for each core if it does not yet exist
                        core = next(item for item in file_name.split('.') if 'core' in item)
                        if core not in data_dict[folder_name]:
                            data_dict[folder_name][core] = {
                                # Channel: [[Freq], [Temp], [Intensity], [Residual]]
                                0: [[], [], [], []],
                                1: [[], [], [], []]
                            }

                        # Read temperature data
                        file_path = os.path.join(folder_path, file_name)
                        data = pd.read_csv(file_path, sep=' ', header=None, names=['Frequency', 'Temperature'])

                        # Update frequency and temperature lists
                        data_dict[folder_name][core][1][0] = list(data.values[:, 0])
                        data_dict[folder_name][core][1][1] = list(data.values[:, 1])

                if file_name.endswith("mean.Jy.txt"):  # Intensity/residual file
                    if file_name.startswith(f"{folder_name}_spw0"):  # Channel 0
                        # Retrieve core and create a dictionary for each core if it does not yet exist
                        core = next(item for item in file_name.split('.') if 'core' in item)
                        if core not in data_dict[folder_name]:
                            data_dict[folder_name][core] = {
                                # Channel: [[Freq], [Temp], [Intensity], [Residual]]
                                0: [[], [], [], []],
                                1: [[], [], [], []]
                            }

                        # Read intensity/residual data
                        file_path = os.path.join(folder_path, file_name)
                        data = pd.read_csv(file_path, sep=' ', header=None, names=['Frequency', 'Temperature'])

                        # Update intensity/residual lists
                        idx =  2 if 'residual' not in file_name else 3
                        data_dict[folder_name][core][0][idx] = list(data.values[:, 1])

                    if file_name.startswith(f"{folder_name}_spw1"):  # Channel 1
                        # Retrieve core and create a dictionary for each core if it does not yet exist
                        core = next(item for item in file_name.split('.') if 'core' in item)
                        if core not in data_dict[folder_name]:
                            data_dict[folder_name][core] = {
                                # Channel: [[Freq], [Temp], [Intensity], [Residual]]
                                0: [[], [], [], []],
                                1: [[], [], [], []]
                            }

                        # Read temperature data
                        file_path = os.path.join(folder_path, file_name)
                        data = pd.read_csv(file_path, sep=' ', header=None, names=['Frequency', 'Temperature'])

                        # Update intensity/residual lists
                        idx = 2 if 'residual' not in file_name else 3
                        data_dict[folder_name][core][1][idx] = list(data.values[:, 1])

            # Update lists to Data Frames
            for core in data_dict[folder_name].keys():
                data_dict[folder_name][core][0] = pd.DataFrame(data_dict[folder_name][core][0]).T
                data_dict[folder_name][core][0].columns = ['Frequency', 'Temperature', 'Intensity', 'Residual']

                data_dict[folder_name][core][1] = pd.DataFrame(data_dict[folder_name][core][1]).T
                data_dict[folder_name][core][1].columns =['Frequency', 'Temperature', 'Intensity', 'Residual']

    return data_dict


def interpolate_data(data_dict: dict):
    """
    Retrieve all frequencies for spw0 and spw1 and interpolate the data to a common base
    :param data_dict: original data dict
    :return: interpolated data dict
    """

    interpolated_data = {}
    common_frequency = []
    for spw in range(2):
        min_f = None
        max_f = None
        # Iterate through the frequency data of all regions for each spw
        for region in data_dict.keys():
            for core in data_dict[region].keys():
                freq_array = data_dict[region][core][spw]['Frequency']
                if min_f is None:
                    min_f = freq_array.min()
                else:
                    min_f = min(min_f, freq_array.min())

                if max_f is None:
                    max_f = freq_array.max()
                else:
                    max_f = max(max_f, freq_array.max())

        # Establish new frequency range and the new common base
        common_frequency.append(np.linspace(min_f, max_f, 10000))

    # Interpolate the data based on this new frequency
    keys = ['Temperature', 'Intensity', 'Residual']
    for region in tqdm(data_dict.keys(), desc=f'Interpolating data'):
        interpolated_data[region] = {}
        for core in data_dict[region].keys():
            interpolated_data[region][core] = {}
            for spw in range(2):
                # Interpolate data
                freq_array = np.flip(data_dict[region][core][spw]['Frequency'])
                data = np.array(common_frequency[spw])
                for key in keys:
                    data = np.vstack((data, np.interp(common_frequency[spw], freq_array, np.flip(data_dict[region][core][spw][key]), left=0, right=0)))

                interpolated_data[region][core][spw] = pd.DataFrame(data.T, columns=data_dict[region][core][spw].columns)

    return interpolated_data


def build_array(data_dict: dict, category: str = 'Intensity', spw: int = 1, return_log: bool = True):
    """
    Build a numpy array with all the data for a given variable
    :param data_dict: original data dict
    :param category: the variable to be searched for
    :param spw: the spw to be stored
    :param return_log: flag for returning the mapping of the array
    :return: variable_array (and mapping)
    """

    variable_array = []
    if return_log:
        mapping = []

    for source in data_dict.keys():
        for core in data_dict[source].keys():
            variable_array.append(data_dict[source][core][spw][category])
            if return_log:
                mapping.append([source, core, spw])

    return np.array(variable_array) if not return_log else np.array(variable_array), mapping


def red_shift_correction(frequency: np.ndarray, signal_array: np.ndarray, reference_signal: np.ndarray = None,
                         plot_reference: bool = False, v_list = np.linspace(-250, 250, 500)):
    """
    Calculate each signal's speed based on their correlation to a reference signal
    :param frequency: 1D array with the frequencies of the signals in signal_array
    :param signal_array: 2D array with all the signals to be corrected
    :param reference_signal: signal to be matched by the other signals
    :param plot_reference: flag for plotting the reference signal
    :param v_list: list with the velocities to be searched in the broad search phase
    :return: corrected signals and velocity list
    """

    # Create the reference signal if it is not passed as an input
    if reference_signal is None:
        # Determine the frequencies at which peaks are most likely expected
        peaks = [
            220398.684,  # 13CO(2-1)
            219949.433,  # SO(5,6-4,5)
            219560.358   # C18O(2-1)
        ]
        peak_loc = [np.argmin(np.abs(frequency - peak)) for peak in peaks]

        reference_signal = np.zeros_like(frequency)
        for peak_location in peak_loc:
            width = 20
            reference_signal += np.exp(-((np.arange(len(frequency)) - peak_location) ** 2 / (2 * width ** 2)))

    if plot_reference:
        plt.figure(figsize=(10, 6))
        plt.plot(frequency, reference_signal/np.max(reference_signal))
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Intensity [Jy]')

        plt.axvline(220398.684, color='black', linestyle='--', label=r'$^{13}$CO(2-1)')
        plt.axvline(219949.433, color='green', linestyle='--', label=r'SO(5,6-4,5)')
        plt.axvline(219560.358, color='purple', linestyle='--', label=r'C$^{18}$O(2-1)')

        plt.legend()
        plt.show()

    corrected_signals = np.zeros_like(signal_array)
    signal_velocities = []
    for i in tqdm(range(len(signal_array)), desc='Shifting signals'):
        signal = signal_array[i]
        signal[signal < 0] = 0  # Remove negative values

        # Find broad alignment
        max_sim = -1
        best_vel = 0
        for v in v_list:
            shifted_signal = shift_signal(frequency, signal, v)[0]
            similarity = cosine_similarity(shifted_signal, reference_signal)

            if similarity > max_sim:
                max_sim = similarity
                best_vel = v

        # Fine-tune alignment
        max_sim = -1
        dv = 1 / 5
        v = best_vel - (1 - dv)
        while np.abs(dv) > 1e-10:
            v += dv
            sim = cosine_similarity(reference_signal, shift_signal(frequency, signal, v)[0])

            if sim < max_sim:
                dv /= -2

            max_sim = sim

        corrected_signals[i] = shift_signal(frequency, signal, v)[0]
        signal_velocities.append(v)

    return corrected_signals, signal_velocities


if __name__ == "__main__":
    # raw_data = pickle.load(open("Data/Raw/data_dict.pkl", "rb"))
    # interpolated_data = interpolate_data(raw_data)
    #
    # with open('Data/Raw/interpolated_data_dict.pkl', 'wb') as f:
    #     pickle.dump(interpolated_data, f)

    interpolated_data = pickle.load(open('Data/Raw/interpolated_data_dict.pkl', 'rb'))
    intensity_array, mapping = build_array(interpolated_data)
    f1 = interpolated_data['100132']['core1'][1]['Frequency']

    corrected_signals, red_shift_velocities = red_shift_correction(frequency=f1, signal_array=intensity_array)
