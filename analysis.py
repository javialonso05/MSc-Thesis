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

# Load data
freq = np.load('Data/frequencies.npy')
if freq.shape == (2, 10000):
    print(f'Freq is stored separately')
    freq = np.hstack((freq[0], freq[1]))

subtraction_signal = np.load('Data/subtraction_shifted_array.npy')

data_info = pd.read_csv('Data/data_info.csv')
velocity_info = pd.read_csv('Data/velocity_info.csv')

labels, sizes = np.unique(velocity_info['CC_label'], return_counts=True)
relevant_labels = [labels[i] for i in range(len(labels)) if sizes[i] >= 20]

# Define dataframe
variables = ['Mcore', 'Tcore', 'Lclump']
df = data_info[variables]

idx = []
labels = []
for i in range(len(data_info)):
    source = data_info['CLUMP'].iloc[i]
    core = data_info['ID'].iloc[i]
    
    index = velocity_info[(velocity_info['Source'] == source) & (velocity_info['Core'] == core)].index.to_list()
    if len(index) > 0:
        idx.append(index[0])
        labels.append(velocity_info['CC_label'].iloc[index[0]])

df = df.iloc[idx]
df['Label'] = labels

# Test analyzer
from src.analysis.cluster_analysis import Analyzer
analyzer = Analyzer(minimum_samples=20)
results = analyzer.test_median_difference(df)
posthoc_results = analyzer.posthoc_test(df, results)


print('\nPosthoc results:')
for var in posthoc_results.keys():
    print(f'{var}:')
    
    groups = posthoc_results[var].keys()
    print(f'\t{len(groups)} groups')
    
    sizes = [len(posthoc_results[var][group]) for group in groups]
    print(f'\tSizes: {sizes}')
    
analyzer.visualize_results(data=df, posthoc_results=posthoc_results,
                           spectra=subtraction_signal, frequency=freq)

        
# TODO: check splits with other labelling assignments
# TODO: correlate observed properties to the spectra