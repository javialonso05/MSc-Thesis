import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cluster_distributions(data: pd.DataFrame,
                               labels: np.ndarray,
                               variable: str,
                               min_cluster_size: int = 10,
                               cluster_labels: list = None,
                               ):
    """
    Plot the distribution of physical properties for clusters in the dataset as boxplots
    :param data: DataFrame containing the cluster data
    :param variable: List of columns to plot. If None, all columns will be used
    :param min_cluster_size: Minimum size of clusters to be included in the plot
    :param cluster_labels: List of cluster labels to filter the data. None if min_cluster_size is specified
    :return: None
    """
    
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
        
    
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' not found in the DataFrame.")
    
    # Make box plots for each cluster for the given variable
    
    # Filter data for the specified clusters
    idx_mask = [i for i, label in enumerate(labels) if label in cluster_labels]
    filtered_data = data.iloc[idx_mask]
    
    print(filtered_data.head(5))
    
    # Create boxplot
    ax = filtered_data.boxplot(column=variable, by='Cluster', figsize=(10, 6))
    ax.set_xlabel('Cluster')
    ax.set_ylabel(variable)
    
    # Show plot
    plt.suptitle('')
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