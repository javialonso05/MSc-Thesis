import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cluster_distributions(core_data: pd.DataFrame,
                               core_info: pd.DataFrame,
                               variable: str,
                               min_cluster_size: int = 10,
                               cluster_labels: list = None,
                               ):
    """
    Plot boxplots for a given property and the relevant clusters
    Args:
        core_data (pd.DataFrame): physical properties of the cores
        core_info (pd.DataFrame): label assignment and core info (source + coreID)
        variable (str): variable from core_data to test
        min_cluster_size (int, optional): Minimum cluster size to consider. Defaults to 10.
        cluster_labels (list, optional): Clusters to consider. Defaults to None.

    Raises:
        ValueError: {variable} not in core_data
        ValueError: Either min_cluster_size or cluster_labels must be specified.
    """
    
    labels = core_info['Labels'].values
    
    # Check inputs    
    if cluster_labels is None:  # Clusters to plot
        cluster_labels = []
        if min_cluster_size is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if count >= min_cluster_size:
                    cluster_labels.append(label)
        else:
            raise ValueError("Either min_cluster_size or cluster_labels must be specified.")
        
    
    if variable not in core_data.columns:
        raise ValueError(f"'{variable}' not found in the DataFrame.")
    
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Find mapping: core_info -> core_data
    mapping = []
    for i in range(len(core_info)):
        idx = core_info[(core_info['Source'].iloc[i] == core_data['CLUMP']) & (core_info['Core'].iloc[i] == core_data['ID'])].index
        if len(idx) > 0:
            mapping.append([i, idx[0]])
    
    # Find mask
    data_mask = []
    info_mask = []
    for i in range(len(mapping)):
        info_idx, data_idx = mapping[i]
        
        if core_info['Labels'].iloc[info_idx] in cluster_labels:
            data_mask.append(data_idx)
            info_mask.append(info_idx)
    
    # Select data
    core_data = core_data.iloc[data_mask]
    core_info = core_info.iloc[info_mask]
    
    # Plot properties
    df = pd.DataFrame()
    df[variable] = core_data[variable]
    df['Labels'] = core_info['Labels']
    df.boxplot(column=variable, by='Labels')
    plt.ylabel(variable)
    plt.show()


def find_mapping(signal_data: np.ndarray, 
                 core_data: pd.DataFrame,
                 core_mapping: dict) -> list:
    """
    Find the mapping of signal indices to data values.
    :param data: Array with signal data
    :param core_mapping: Mapping from signal indices to source and core tags
    :return: List of mapped cluster labels
    """
    
    mapping = []
    for i in range(len(signal_data)):
        # Extract region and core from the mapping
        region, core = core_mapping[i]
        
        # Locate datapoint in core_data
        core_idx = core_data[(core_data['CLUMP'] == region) & (core_data['ID'] == core[4:])].index
        if len(core_idx) == 0:
            mapping.append(np.nan)
        else:
            mapping.append(core_idx[0])
    
    return mapping



if __name__ == '__main__':
    # Load core data
    data_info = pd.read_csv('Data/Raw/7MTM2TM1_Core_catalogue_out_official_v3_run6Apr23+16Jan24_sn-5_may24_clump_cat.txt', sep='\s+', comment='\\')
    data_info = data_info.drop(data_info.index[0]).reset_index(drop=True)
    data_info.to_csv('Data/data_info.csv', index=False)