import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from src.features.transformations import shift_signal


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
            filtering: list = ['None', 'subtraction', 'threshold'],
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
    regions = sorted(list(data_dict.keys()))
    for spw in range(2):
        min_f = None
        max_f = None
        # Iterate through the frequency data of all regions for each spw
        for region in regions:
            for core in sorted(list(data_dict[region].keys())):
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
    for region in tqdm(regions, desc=f'Interpolating data'):
        interpolated_data[region] = {}
        for core in sorted(list(data_dict[region].keys())):
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
