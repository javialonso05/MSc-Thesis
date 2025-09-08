import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from src.features.transformations import shift_signal, transform_signal


class RedShiftCorrector:
    def __init__(self,
                 frequency: np.ndarray,
                 reference_signal: np.ndarray = None,
                 tolerance: float = 0.5,
                 max_velocity: float = 200.0):
        
        self.frequency = frequency.flatten()
        self.tol = tolerance
        self.max_v = max_velocity
        self.results = None

        # Create reference signal
        if reference_signal is None:
                peaks = [
                    218222.192,  # H2CO
                    218475.632,  # H2CO
                    220398.684,  # 13CO(2-1)
                    219949.433,  # SO(5,6-4,5)
                    219560.358   # C18O(2-1)
                ]
                peak_loc = [np.argmin(np.abs(self.frequency - peak)) for peak in peaks]

                reference_signal = np.zeros_like(self.frequency)
                for peak_location in peak_loc:
                    reference_signal += np.exp(-((np.arange(len(self.frequency)) - peak_location) ** 2 / (2 * 10 ** 2)))
        self.reference_signal = reference_signal.reshape(1, -1)
    
    def _calculate_velocity(self,
                            signal_array: np.ndarray,
                            method: str,
                            dv_0: float = 1.0,
                            dv_1: float = 0.01):
        """
        Calculate the velocity of each source
        :param signal_array - Array with the signals to be shifted
        :param dv_0 - Velocity increments for the broad alignment
        """
        
        
        # Initialize velocity list
        signal_velocities = []
        if signal_array.shape[1] != self.reference_signal.shape[1]:  # Use only spw1
            raise ValueError(f"The shape of the input ({signal_array[0].shape[1]}) and reference signal ({self.reference_signal.shape[1]}) do not match")
        
        # Start iteration
        for i in tqdm(range(len(signal_array)), desc='Shifting signals'):
            signal = np.array(signal_array[i])
            signal[signal < 0] = 0  # Remove negative values

            # Find broad alignment
            max_sim = -1
            best_vel = self._grid_search(signal=signal,
                                         v_min=-250,
                                         v_max=250,
                                         dv=1,
                                         best_sim=max_sim,
                                         best_v=0)

            # Fine-tune alignment
            sim = cosine_similarity(self.reference_signal, shift_signal(self.frequency, signal, best_vel).reshape(1, -1))
            if method == 'grid':  # grid search
                best_vel = self._grid_search(signal=signal,
                                    v_min=-best_vel - 1,
                                    v_max=best_vel + 1,
                                    dv=0.01,
                                    best_sim=sim,
                                    best_v=best_vel)
            else:  # recursive bisection
                best_vel = self._recursive_bisection(
                    signal=signal,
                    best_v=best_vel,
                    best_sim=sim
                )
            
            signal_velocities.append(best_vel)

        return signal_velocities
    
    def _grid_search(self,
                     signal: np.ndarray,
                     v_min: float,
                     v_max: float,
                     dv: float,
                     best_sim: float,
                     best_v: float):
        """

        Args:
            signal (np.ndarray): _description_
            v_min (float): _description_
            v_max (float): _description_
            dv (float): _description_
            current_sim (float): _description_
        """
        for v in np.arange(v_min, v_max + dv, dv):
            shifted_signal = shift_signal(self.frequency, signal, v)
            similarity = cosine_similarity(shifted_signal.reshape(1, -1), self.reference_signal)
            
            if similarity > best_sim:
                best_sim = similarity
                best_v = v
        
        return best_v
        ...
    
    def _recursive_bisection(self,
                             signal: np.ndarray,
                             best_v: float,
                             best_sim: float,
                             tol: float = 1e-3,
                             dv: float = 0.1):
        """

        Args:
            signal (np.ndarray): _description_
            best_v (float): _description_
            best_sim (float): _description_
            tol (float, optional): _description_. Defaults to 1e-10.
            dv (float, optional): _description_. Defaults to 0.1.
        """
        
        v = best_v - dv
        current_similarity = -1
        while abs(dv) > tol:
            v += dv
            
            shifted_signal = shift_signal(self.frequency, signal, v)
            similarity = cosine_similarity(shifted_signal.reshape(1, -1), self.reference_signal)
            
            if similarity < current_similarity:
                dv /= -2
            
            current_similarity = similarity
            
        if current_similarity > best_sim:  
            # The algorithm found a more exact velocity
            return v
        else:
            # The algorithm got trapped in a relative maximum
            return best_v
            

    def fit(self,
            raw_data: np.ndarray,
            mapping: list,
            residual: np.ndarray = None,
            filtering: list = ['None', 'subtraction', 'sigma'],
            method: str = "grid") -> pd.DataFrame:
        """
        Find the velocities of all sources according to a majority-voting scheme

        Args:
            raw_data (np.ndarray): array with the interpolated raw signals.
            mapping (list): ordered list corresponding to the sources and cores of each signal in raw_data.
            residual (np.ndarray): array with each signals residual. Defaults to None.
            filtering (list, optional): list of filtering methods to be used. Defaults to [None, 'subtraction', 'sigma'].
            method (str, optional): method to use during the fine search. Acceptable values are "grid" and "bisection". Defaults to "grid".
            
        Raises:
            ValueError: Residual is required if "subtraction" or "sigma" are in "filtering"
            ValueError: Signal and residual data shape do not match
            ValueError: Unknown filtering method. Accepted methods are "subtraction", "sigma" and "savgol"
            ValueError: Length of data array and mapping do not match
            ValueError: Unknown search method. Acceptable methods are "grid" and "bisection".
            Warning: Majority voting can lead to errors if an even number of methods is used.

        Returns:
            pd.DataFrame: DataFrame with each signal's data, the velocity for each method and the best velocity.
        """

        # Check inputs
        if residual is None and ("subtraction" in filtering or "sigma" in filtering):
            raise ValueError('Residual is required if "subtraction" or "sigma" are in "filtering"')
        if residual is not None:
            if raw_data.shape != residual.shape:
                raise ValueError('Signal and residual data shape do not match')
        if len(filtering) % 2 == 0:
            raise Warning('Majority voting can lead to errors if an even number of methods is used.')
        if len(mapping) != len(raw_data):
            raise ValueError('Length of data array and mapping do not match')
        if method != "bisection" and method != "grid":
            raise ValueError('Unknown search method. Acceptable methods are "grid" and "bisection".')
        
        # Create output dataframe
        results = {
            'Source': [],
            'Core': [],
            'Best_velocity': []
        }
        
        # Find the velocity for each filtering method
        velocity_matrix = []
        for method in filtering:
            print(f'Filtering method: {method}')
            # Calculate velocity
            if method == 'None':
                signal_data = raw_data
            else:
                signal_data = filter_data(data=raw_data, filter_type=method, residual=residual)
            velocity_list = self._calculate_velocity(signal_data, method)
            
            # Define tag\
            tag = f'{method}_velocity'
            
            # Store results
            results[tag] = velocity_list
            velocity_matrix.append(velocity_list)
        
        # Find best velocity according to majority voting
        velocity_matrix = np.array(velocity_matrix).T
        for i, row in enumerate(velocity_matrix):
            # Count how many velocities are within self.tol of each other
            counts = []
            for v in row:
                count = np.sum(np.abs(row - v) <= self.tol)
                counts.append(count)
            
            # Pick the velocity with the highest count (most common within tolerance)
            best_idx = np.argmax(counts)
            
            # If counts = 0, best_v will be NaN
            if np.all(counts == 0):
                results['Best_velocity'].append(np.nan)
            else:
                best_v = np.mean(row[np.abs(row - row[best_idx]) <= self.tol])
                if np.abs(best_v) > 200:
                    results['Best_velocity'].append(np.nan)
                else:
                    results['Best_velocity'].append(best_v)
            
            # Append mapping data
            results['Source'].append(mapping[i][0])
            results['Core'].append(int(mapping[i][1][4:]))
        
        # Convert to dataframe and return
        results = pd.DataFrame(results)
        self.results = results
        return results      
                   

# Loading data functions
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

    if not return_log:
        return np.array(variable_array)
    return np.array(variable_array), mapping


# Red-shift correction functions
def red_shift_correction(frequency: np.ndarray, signal_array: np.ndarray, reference_signal: np.ndarray = None,
                         plot_reference: bool = False, v_list = np.linspace(-250, 250, 500),
                         width: float = 10):
    """
    Calculate each signal's speed based on their correlation to a reference signal
    :param frequency: 1D array with the frequencies of the signals in signal_array
    :param signal_array: 2D array with all the signals to be corrected
    :param reference_signal: signal to be matched by the other signals
    :param plot_reference: flag for plotting the reference signal
    :param v_list: list with the velocities to be searched in the broad search phase
    :param width: width of the peaks in the reference signal
    :return: velocity list
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
        signal = np.array(signal_array[i])
        signal[signal < 0] = 0  # Remove negative values

        # Find broad alignment
        max_sim = -1
        best_vel = 0
        for v in v_list:
            shifted_signal = shift_signal(frequency, signal, v)
            similarity = cosine_similarity(shifted_signal, reference_signal)

            if similarity > max_sim:
                max_sim = similarity
                best_vel = v

        # Fine-tune alignment
        max_sim = -1
        dv = 0.5 / 10
        v = best_vel - (1 - dv)
        n = 0
        while np.abs(dv) > 1e-10:
            v += dv
            sim = cosine_similarity(reference_signal, shift_signal(frequency, signal, v))

            if sim < max_sim:
                dv /= -2

            max_sim = sim
            n += 1

            # Add exit
            if v >= best_vel + 1 or v <= best_vel - 1:
                v = best_vel
                break
            if n >= 1000:
                v = best_vel
                break

        corrected_signals[i] = shift_signal(frequency, signal, v)
        signal_velocities.append(v)

    return signal_velocities


def best_velocity(velocity_array: np.ndarray, threshold: float = 2):
    """
    Find the best velocity amongst all the possible options
    :param velocity_array: 2D array where every row corresponds to a signal and every column to [Raw, Sigma, Sub, ...]
    :param threshold: value below which it is considered a match
    :return: best_v
    """

    best_v = []
    for i in range(len(velocity_array)):
        v = velocity_array[i]
        v_diff = v - v.reshape(-1, 1)

        if np.abs(v_diff[0, 2]) < threshold and np.abs(v[0]) < 200:
            best_v.append(v[0])
            continue
        elif np.abs(v_diff[0, 1]) < threshold and np.abs(v[0]) < 200:
            best_v.append(v[0])
            continue
        elif np.abs(v_diff[1, 2]) < threshold and np.abs(v[1]) < 200:
            best_v.append(v[2])
            continue
        elif np.abs(v_diff[2, 3]) < threshold and np.abs(v[2]) < 200:
            best_v.append(v[2])
            continue
        else:
            best_v.append(np.nan)

    return np.array(best_v)


def probability_find_v(find_samples, no_find_samples, x2 = np.linspace(0, 10, 300)):
    """
    Find and plot the probability of finding the signal's velocity given its SNR
    :param find_samples: 1D array with the SNR of the signals whose velocity was found
    :param no_find_samples: 1D array with the SNR of the signals whose velocity was not found
    :param x2: SNR where to evaluate the probability of finding the signal's velocity
    :return: probability of finding the velocity given x2
    """

    from sklearn.neighbors import KernelDensity

    find_kde = KernelDensity(kernel='gaussian').fit(find_samples)
    no_find_kde = KernelDensity(kernel='exponential').fit(no_find_samples)

    f1 = np.exp(find_kde.score_samples(x2.reshape(-1, 1)))  # Find v pdf
    f2 = np.exp(no_find_kde.score_samples(x2.reshape(-1, 1)))  # Not find v pdf
    p1 = len(find_samples) / (len(find_samples) + len(no_find_samples))  # Find v prior
    p2 = len(no_find_samples) / (len(find_samples) + len(no_find_samples))  # Not find v prior
    p = p1 * f1 / (p1 * f1 + p2 * f2)

    # Plot
    plt.figure(figsize=(5, 4))

    plt.plot(x2, f1, color='tab:green', label='Find v PDF')
    plt.plot(x2, f2, color='tab:red', label='Not find v PDF')

    plt.xlabel('SNR [dB]')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.plot(x2, p, color='tab:blue', label='P(Find v | SNR)')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return p


# Data transformation functions
def filter_data(data: np.ndarray, filter_type: str, residual = None, **kwargs):
    """

    :param data: 1D/2D array with the data to be filtered
    :param filter_type: type of filter to perform
    :param residual: 1D array with the residual of the data
    :param kwargs: additional arguments passed to the filter function
    :return: filtered data
    """
    if filter_type is None:
        return data

    if filter_type == 'sigma':
        # Check residual
        if residual is None:
            raise ValueError("Residual cannot be None if the filter type is sigma")

        from src.features.transformations import sigma_filter

        filtered_data = sigma_filter(data, residual, **kwargs)

    elif filter_type == 'subtraction':
        # Check arguments
        if residual is None:
            raise ValueError("Residual cannot be None if the filter type is subtraction")
        if np.shape(data) != np.shape(residual):
            raise ValueError("Residual and data must have the same shape")

        filtered_data = data - residual

    elif filter_type == 'savgol':
        from src.features.transformations import savgol_filter

        filtered_data = savgol_filter(data, **kwargs)
        filtered_data[filtered_data < 0] = 0

    else:
        raise ValueError(f'{filter_type} is not recognised as an established filter type')

    return filtered_data


if __name__ == "__main__":
    # raw_dict = load_data('Data/Raw/sources_extracted')
    # with open("Data/Raw/data_dict.pkl", "wb") as f:
    #     pickle.dump(raw_dict, f)

    # raw_data = pickle.load(open("Data/Raw/data_dict.pkl", "rb"))
    # interpolated_data = interpolate_data(raw_data)

    # with open('Data/Raw/interpolated_data_dict.pkl', 'wb') as f:
    #     pickle.dump(interpolated_data, f)

    """
    Code for the scatter + density graphs:
        # Create the figure and axes using GridSpec
        fig = plt.figure(figsize=(10, 6))
        grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)
        main_ax = fig.add_subplot(grid[1:, :-1])  # Scatter plot
        x_density = fig.add_subplot(grid[0, :-1], sharex=main_ax)  # Top density plot
        y_density = fig.add_subplot(grid[1:, -1], sharey=main_ax)  # Right density plot
        # Scatter plot of the vectors
        main_ax.scatter(data_info['SNR'][idx_2], v_diff_2, label='1', alpha=0.6)
        main_ax.scatter(data_info['SNR'][idx_1], v_diff_1, label='2', alpha=0.6)
        main_ax.scatter(data_info['SNR'][idx_0], v_diff_0, label='3', alpha=0.6)
        main_ax.legend(title='# mismatches')
        main_ax.set_xlabel('SNR [dB]')
        main_ax.set_ylabel(r'$\Delta$V [km/s]')
        # KDE plot for x-axis
        sns.kdeplot(data_info['SNR'][idx_2], ax=x_density, fill=False, color='tab:blue')
        sns.kdeplot(data_info['SNR'][idx_1], ax=x_density, fill=False, color='tab:orange')
        sns.kdeplot(data_info['SNR'][idx_0], ax=x_density, fill=False, color='tab:green')
        # x_density.axis('off')
        x_density.set_xlabel('')
        # KDE plot for y-axis (horizontal)
        sns.kdeplot(y=v_diff_2, ax=y_density, fill=False, color='tab:blue')
        sns.kdeplot(y=v_diff_1, ax=y_density, fill=False, color='tab:orange')
        sns.kdeplot(y=v_diff_0, ax=y_density, fill=False, color='tab:green')
        # y_density.axis('off')
        plt.show()
    """

    # Load signals
    interpolated_data = pickle.load(open('Data/Raw/interpolated_data_dict.pkl', 'rb'))

    # Build arrays
    f0 = interpolated_data['100132']['core1'][0]['Frequency']
    f1 = interpolated_data['100132']['core1'][1]['Frequency']
    freq = np.hstack((f0, f1))
    print(min(f1), max(f1))

    spw1_array, mapping = build_array(interpolated_data, category='Intensity')
    spw0_array = build_array(interpolated_data, category='Intensity', spw=0, return_log=False)
    intensity_array = np.hstack((spw0_array, spw1_array))

    residual_array_spw1 = build_array(interpolated_data, category='Residual', return_log=False)
    residual_array = np.hstack((build_array(interpolated_data, category='Residual', return_log=False, spw=0), residual_array_spw1))

    # Shift signals
    data_info = pd.read_csv('Data/data_info.csv')
    best_v = data_info['Best velocity 2'].values
    
    # Print reference signal
    peaks = [
        220398.684,  # 13CO(2-1)
        219949.433,  # SO(5,6-4,5)
        219560.358   # C18O(2-1)
    ]
    peak_loc = [np.argmin(np.abs(freq - peak)) for peak in peaks]

    reference_signal = np.zeros_like(freq)
    for peak_location in peak_loc:
        reference_signal += np.exp(-((np.arange(len(freq)) - peak_location) ** 2 / (2 * 10 ** 2)))

    plt.figure(figsize=(10, 6))
    plt.plot(f1, reference_signal/np.max(reference_signal))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Intensity [Jy]')

    plt.axvline(220398.684, color='black', linestyle='--', label=r'$^{13}$CO(2-1)')
    plt.axvline(219949.433, color='green', linestyle='--', label=r'SO(5,6-4,5)')
    plt.axvline(219560.358, color='purple', linestyle='--', label=r'C$^{18}$O(2-1)')

    plt.legend()
    plt.show()


    # Create reference signal
    # peaks = [
    #     220398.684,  # 13CO(2-1)
    #     219949.433,  # SO(5,6-4,5)
    #     219560.358  # C18O(2-1)
    # ]
    # peak_loc = [np.argmin(np.abs(f1 - peak)) for peak in peaks]
    #
    # reference_signal = np.zeros_like(f1)
    # for peak_location in peak_loc:
    #     reference_signal += np.exp(-((np.arange(len(f1)) - peak_location) ** 2 / (2 * 10 ** 2)))
    #
    #
    # v_diff = []
    # v_list = np.linspace(-5, 5, 1000)
    # for filter_method in [None, 'sigma', 'subtraction', 'savgol']:
    #     if filter_method is None:
    #         data = spw1_array
    #         label = 'raw'
    #     else:
    #         data = filter_data(spw1_array, filter_type=filter_method, residual=residual_array_spw1)
    #         label = f'{filter_method}-filtered'
    #
    #     v_change = []
    #     for i in tqdm(range(len(data)), desc=f'Re-shifting {label} signals'):
    #         v = best_v[i]
    #         if np.isnan(v):
    #             v_change.append(np.nan)
    #             continue
    #
    #         sim_list = []
    #         for dv in v_list:
    #             shifted_signal = shift_signal(f1, data[i], v + dv)
    #             sim_list.append(cosine_similarity(shifted_signal.reshape(1, -1), reference_signal.reshape(1, -1))[0][0])
    #
    #         sim_list = np.array(sim_list)
    #         v_change.append(v_list[np.argmax(sim_list)])
    #
    #     v_diff.append(v_change)

    # shifted_signals = np.array([shift_signal(freq, intensity_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    # shifted_residual = np.array([shift_signal(freq, residual_array[i], best_v[i]) for i in range(len(intensity_array)) if not np.isnan(best_v[i])])
    # shifted_mapping = [mapping[i] for i in range(len(mapping)) if not np.isnan(best_v[i])]
    #
    # subtraction_signal = filter_data(np.asarray(shifted_signals), 'subtraction', shifted_residual)
    # sigma_signal = filter_data(np.asarray(shifted_signals), 'sigma', shifted_residual)
    # savgol_signal = filter_data(np.asarray(shifted_signals), 'savgol')
    #
    # # Perform UMAP reduction with the old data
    # import umap
    #
    # subtraction_umap = umap.UMAP(n_neighbors=5, metric='cosine', min_dist=0, random_state=42).fit_transform(subtraction_signal)
    # sigma_umap = umap.UMAP(n_neighbors=5, metric='cosine', min_dist=0, random_state=42).fit_transform(sigma_signal)
    # savgol_umap = umap.UMAP(n_neighbors=5, metric='cosine', min_dist=0, random_state=42).fit_transform(savgol_signal)
    #
    #
    # fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    #
    # ax[0].scatter(subtraction_umap[:, 0], subtraction_umap[:, 1])
    # ax[0].set_title("Subtraction-filtered")
    #
    # ax[1].scatter(sigma_umap[:, 0], sigma_umap[:, 1])
    # ax[1].set_title("Sigma-filtered")
    #
    # ax[2].scatter(savgol_umap[:, 0], savgol_umap[:, 1])
    # ax[2].set_title("SavGol-filtered")
    #
    # plt.show()
