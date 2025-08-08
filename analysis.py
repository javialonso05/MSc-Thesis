# %% Import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# %% Load data
freq = np.load('Data/frequencies.npy')
if freq.shape == (2, 10000):
    print(f'Freq is stored separately')
    freq = np.hstack((freq[0], freq[1]))

subtraction_signal = np.load('Data/subtraction_shifted_array.npy')

data_info = pd.read_csv('Data/data_info.csv')
velocity_info = pd.read_csv('Data/velocity_info.csv')

# Delete data from H_II regions
data_info = data_info[data_info['RADIO_MATCH'] == 0]

# for label_group in ['CC_label', 'AC_label', 'CP_label', 'AP_label']:
for label_group in ['CC_label']:
    subtraction_signal = np.load('Data/subtraction_shifted_array.npy')

    # %% Define dataframe
    variables = ['DIST', 'Lclump', 'Mclump', 'Tclump', 'n(H2)', 'Dcore', 'Lclump/Mclump', 'Surfd_nd']

    idx = []
    labels = []
    sc_pair = []

    # Filter data from data_info not in velocity_info
    for i in range(len(data_info)):
        source = data_info['CLUMP'].iloc[i]
        core = data_info['ID'].iloc[i]
        
        index = velocity_info[(velocity_info['Source'] == source) & (velocity_info['Core'] == core)].index.to_list()
        if len(index) > 0:
            idx.append(index[0])
            labels.append(velocity_info[label_group].iloc[index[0]])
            
            sc_pair.append((source, core))

    df = data_info.iloc[idx]

    # Filter data from velocity_info not in data_info
    sc_pair_2 = [(velocity_info['Source'].iloc[i], velocity_info['Core'].iloc[i]) for i in range(len(velocity_info))]
    mask = [True if sc_pair_2[i] in sc_pair else False for i in range(len(sc_pair_2))]
    subtraction_signal = subtraction_signal[mask]

    # Add label column
    df['Label'] = velocity_info[label_group][mask].copy()
    labels, sizes = np.unique(df['Label'], return_counts=True)
    relevant_labels = [labels[i] for i in range(len(labels)) if sizes[i] >= 10]


    # %% Analyze whether cores from the same source belong to the same cluster
    cores_per_source = np.array([len(df['ID'][df['CLUMP'] == source].unique()) for source in df['CLUMP'].unique()])
    n_cores_per_source, instances_core = np.unique(cores_per_source, return_counts=True)

    clusters_per_source = np.array([len(df['Label'][df['CLUMP'] == source].unique()) for source in df['CLUMP'].unique()])
    n_clusters_per_source, instances_clusters = np.unique(clusters_per_source, return_counts=True)

    plt.scatter(cores_per_source, clusters_per_source)
    plt.plot([1, max(cores_per_source)], [1, max(cores_per_source)], linestyle='--', color='black')
    plt.axhline(len(relevant_labels), linestyle='--', color='black')
    plt.xlabel('Cores per source')
    plt.ylabel('Clusters per source')
    plt.savefig('results/Plots/cores_and_clusters.pdf')
    plt.show()


    # %% Analyze the distribution of data
    from src.analysis.cluster_analysis import Analyzer
    analyzer = Analyzer(minimum_samples=10)
    analyzer.test_normality(df, plot_results=True, variables=variables, save=False)
    results = analyzer.test_median_difference(df[df['Label'].isin(relevant_labels)], variables=variables, show=False, 
                                              save=None)
    posthoc_results = analyzer.posthoc_test(df[df['Label'].isin(relevant_labels)], results)
    analyzer.visualize_results(data=df[df['Label'].isin(relevant_labels)],
                               posthoc_results=posthoc_results,
                               spectra=subtraction_signal[df['Label'].isin(relevant_labels)],
                               frequency=freq, save=None)
    analyzer.visualize_cluster_distribution_split(data=df[df['Label'].isin(relevant_labels)], 
                                                  spectra=subtraction_signal[df['Label'].isin(relevant_labels)],
                                                  frequency=freq, variables=variables, save=None)
