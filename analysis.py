# %% Import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import kdeplot


mapping = np.load("Data/mapping.npy")
freq = np.load('Data/frequencies.npy')
subtraction_signals = np.load('Data/Processed/subtraction_signals.npy')
shifted_signals = np.load("Data/Processed/shifted_signals.npy")
shifted_residual = np.load("Data/Processed/shifted_residual.npy")

data_info = pd.read_csv('Data/data_info.csv')  # Provided
velocity_info = pd.read_csv('Data/velocity_info.csv')  # Calculated
velocity_info = velocity_info.dropna(subset=["Automatic velocity"]).reset_index(drop=True)
n_lines = pd.read_csv("Data/Processed/almagal_nlines_all.csv")

CO_idx = np.argmin(np.abs(freq - 220398.42455376702))

# Load and re-order labels such that largest cluster is cluster 0 and smalles is cluster -1
labels_10 = np.load("Data/Processed/labels10.npy")
labels_23 = np.load("Data/Processed/labels23.npy")

labels_mask_10 = labels_10 != -1  # Mask for clustered signals
labels_mask_23 = labels_23 != -1  # Mask for clustered signals

ordered_labels_10 = labels_10.copy()
ordered_labels_23 = labels_23.copy()

unique_labels, sizes = np.unique(labels_10[labels_mask_10], return_counts=True)
for i, label in enumerate(unique_labels[np.argsort(sizes)[::-1]]):
    ordered_labels_10[labels_10 == label] = i
    
unique_labels, sizes = np.unique(labels_23[labels_mask_23], return_counts=True)
for i, label in enumerate(unique_labels[np.argsort(sizes)[::-1]]):
    ordered_labels_23[labels_23 == label] = i

# Match data from tables to data from available sources
sc_pair = []
data_index_list = []
v_index_list = []
nlines_index_list = []

# Filter data from data_info not in velocity_info
velocity_info = velocity_info.dropna(subset=["Automatic velocity"]).reset_index(drop=True)
for i in range(len(data_info)):
    source = data_info['CLUMP'].iloc[i]
    core = data_info['ID'].iloc[i]
    
    v_index = velocity_info[(velocity_info["Source"] == source) * (velocity_info["Core"] == core)].index.to_list()
    nlines_index = n_lines[(n_lines["Region"] == source) * (n_lines["Core"] == core)].index.to_list()
    
    if len(v_index) > 0 and len(n_lines) > 0:
        if len(v_index) > 1:
            print(f"v_index: {v_index}")
        if len(nlines_index) > 1:
            print(f"n_lines: {nlines_index}")
            
        v_index_list.append(v_index[0])
        nlines_index_list.append(nlines_index[0])
        data_index_list.append(i)
        
        sc_pair.append((source, core))


velocity_info = velocity_info.iloc[v_index_list].reset_index(drop=True)
n_lines = n_lines.iloc[nlines_index_list].reset_index(drop=True)
data_info = data_info.iloc[data_index_list].reset_index(drop=True)

assert np.all(velocity_info["Source"].values == n_lines["Region"].values)
assert np.all(velocity_info["Source"].values == data_info["CLUMP"].values)
assert np.all(velocity_info["Core"].values == n_lines["Core"].values)
assert np.all(velocity_info["Core"].values == data_info["ID"].values)

ordered_labels_10 = ordered_labels_10[v_index_list]
ordered_labels_23 = ordered_labels_23[v_index_list]

subtraction_signals = subtraction_signals[v_index_list]
subtraction_signals /= np.max(subtraction_signals, axis=1, keepdims=True)

# Delete data from H_II regions
h2_mask = np.array(data_info['RADIO_MATCH'] == 0)

#%%
print(f"There are {sum(h2_mask)} H2 regions")

clusters_10_h2, count_10_h2 = np.unique(ordered_labels[h2_mask], return_counts=True)
for i in range(len(clusters_10_h2)):
    print(f"Cluster {clusters_10_h2[i]}: {count_10_h2[i]}/{sum(ordered_labels == clusters_10_h2[i])} signals ({count_10_h2[i]/sum(ordered_labels == clusters_10_h2[i]) * 100:.2f}%) belong to HII regions")

plt.bar(clusters_10_h2, count_10_h2)
plt.xticks(ticks=np.unique(ordered_labels))
plt.xlabel("Cluster label")
plt.ylabel("Number of HII-region signals")
plt.tight_layout()
plt.show()

variables = ['DIST', 'Lclump', 'Mclump', 'Tclump', 'n(H2)', 'Dcore', 'Lclump/Mclump', 'Surfd_nd']

# Add label column
df['Labels_10'] = ordered_labels_10
df["Labels_23"] = ordered_labels_23

cores_per_source = np.array([len(df['ID'][df['CLUMP'] == source].unique()) for source in df['CLUMP'].unique()])
n_cores_per_source, instances_core = np.unique(cores_per_source, return_counts=True)

clusters_per_source = np.array([len(df['Label'][df['CLUMP'] == source].unique()) for source in df['CLUMP'].unique()])
for i, source in enumerate(df["CLUMP"].unique()):
    if -1 in df['Label'][df['CLUMP'] == source]:
        clusters_per_source[i] -= 1
n_clusters_per_source, instances_clusters = np.unique(clusters_per_source, return_counts=True)

fig, ax = plt.subplots(figsize=(7, 4))
kdeplot(x=cores_per_source, y=clusters_per_source, fill=True, cmap="viridis", clip=[(1, 50), (0, 8)], ax=ax, thresh=0)
ax.plot([1, len(np.unique(ordered_labels)) - 1], [1, len(np.unique(ordered_labels)) - 1], linestyle='--', color='lightgrey', linewidth=3)
ax.scatter(cores_per_source, clusters_per_source)
plt.ylim((0, 8))
plt.xlabel('Cores per source', fontsize=14)
plt.ylabel('Clusters per source', fontsize=14)
plt.savefig('results/analysis/cores_and_clusters_label10.pdf')
plt.show()


# %% Analyze the distribution of data
from src.analysis.cluster_analysis import Analyzer
analyzer = Analyzer(minimum_samples=20)
analyzer.test_normality(df, plot_results=True, variables=variables, save=False)
results = analyzer.test_median_difference(df[df['Label'].isin(ordered_labels)], variables=variables, show=False, 
                                            save=None)
posthoc_results = analyzer.posthoc_test(df[df['Label'].isin(ordered_labels)], results)
analyzer.visualize_results(data=df[df['Label'].isin(ordered_labels)],
                            posthoc_results=posthoc_results,
                            spectra=subtraction_signals[df['Label'].isin(ordered_labels)],
                            frequency=freq, save=None)
analyzer.visualize_cluster_distribution_split(data=df[df['Label'].isin(ordered_labels)], 
                                                spectra=subtraction_signals[df['Label'].isin(ordered_labels)],
                                                frequency=freq, variables=variables, save=None)
