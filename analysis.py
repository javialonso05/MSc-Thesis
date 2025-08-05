# %% Import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
Variables of interest:
    - N_FRAG
    - DIST
    - Mclump
    - Lclump
    - Tclump
    - Mcore
    - Lcore
    - Tcore
"""

# %% Load data
freq = np.load('Data/frequencies.npy')
if freq.shape == (2, 10000):
    print(f'Freq is stored separately')
    freq = np.hstack((freq[0], freq[1]))

subtraction_signal = np.load('Data/subtraction_shifted_array.npy')

data_info = pd.read_csv('Data/data_info.csv')
velocity_info = pd.read_csv('Data/velocity_info.csv')

labels, sizes = np.unique(velocity_info['CC_label'], return_counts=True)
relevant_labels = [labels[i] for i in range(len(labels)) if sizes[i] >= 20]

# %% Define dataframe
variables = ['DIST', 'Lclump', 'Mclump', 'Tclump', 'Tcore', 'Mcore', 'Dcore', 'Lclump/Mclump']

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
        labels.append(velocity_info['CC_label'].iloc[index[0]])
        
        sc_pair.append((source, core))

df = data_info.iloc[idx]

# Filter data from velocity_info not in data_info
sc_pair_2 = [(velocity_info['Source'].iloc[i], velocity_info['Core'].iloc[i]) for i in range(len(velocity_info))]
mask = [True if sc_pair_2[i] in sc_pair else False for i in range(len(sc_pair_2))]
subtraction_signal = subtraction_signal[mask]

# Add label column
df['Label'] = velocity_info['CC_label'][mask]


# %% Analyze whether cores from the same source belong to the same cluster
cores_per_source = np.array([len(df['ID'][df['CLUMP'] == source].unique()) for source in df['CLUMP'].unique()])
n_cores_per_source, instances_core = np.unique(cores_per_source, return_counts=True)

clusters_per_source = np.array([len(df['Label'][df['CLUMP'] == source].unique()) for source in df['CLUMP'].unique()])
n_clusters_per_source, instances_clusters = np.unique(clusters_per_source, return_counts=True)

ratio = clusters_per_source / cores_per_source
mean_ratio = [np.mean(ratio[cores_per_source == n_cores_per_source[i]]) for i in range(len(n_cores_per_source))]

from scipy.stats import linregress

# Linear regression between n_cores_per_source and mean_ratio
slope, intercept, r_value, p_value, std_err = linregress(n_cores_per_source, mean_ratio)
print(f'R2 = {r_value**2}')

plt.scatter(n_cores_per_source, mean_ratio, label='Data')
plt.plot(n_cores_per_source, intercept + slope * n_cores_per_source, color='black', linestyle='--', label=f'y={slope:.3f}x + {intercept:.3f}')

plt.xlabel('Cores per source')
plt.ylabel('Mean(N_clusters / N_cores)')
plt.legend()
plt.show()


# %% Analyze the distribution of data
from src.analysis.cluster_analysis import Analyzer
analyzer = Analyzer(minimum_samples=20)
# analyzer.test_normality(df, plot_results=False, variables=['Mcore', 'Tcore', 'Lclump'])
results = analyzer.test_median_difference(df[df['Label'].isin(relevant_labels)], variables=variables, show=False, save=True)
posthoc_results = analyzer.posthoc_test(df[df['Label'].isin(relevant_labels)], results)
analyzer.visualize_results(data=df[df['Label'].isin(relevant_labels)], posthoc_results=posthoc_results, spectra=subtraction_signal[df['Label'].isin(relevant_labels)], frequency=freq)

# %% TODO: correlate variables to clusters

# variables = ['DIST', 'Lclump', 'Mclump', 'Tclump', 'Tcore', 'Mcore', 'Dcore', 'Lclump/Mclump']
# for var in variables:
#     # Filter relevant data
#     subset = df[df['Label'].isin(relevant_labels)]

#     # Compute mean per label and sort
#     means = subset.groupby('Label')[var].median().sort_values()
#     sorted_labels = means.index[::-1]

#     # Reorder 'Label' as a categorical with sorted order
#     subset = subset.copy()
#     subset['Label'] = pd.Categorical(subset['Label'], categories=sorted_labels, ordered=True)

#     # Plot boxplot
#     subset.boxplot(column=var, by='Label', grid=False, figsize=(12, 5))

#     plt.title(f'Boxplot of {var} by Label (sorted by mean)')
#     plt.suptitle('')
#     plt.xlabel('Label')
#     plt.ylabel(var)
#     plt.tight_layout()
#     plt.show()
    
# %%
