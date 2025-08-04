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

# Test for differences
variables = ['Mcore', 'Tcore', 'Lclump']
df = data_info[variables]
df['Label'] = velocity_info['CC_label']

# Test analyzer
from src.analysis.cluster_analysis import Analyzer
analyzer = Analyzer(minimum_samples=20)
results, posthoc_results = analyzer.test_median_difference(df)


print('\nPosthoc results:')
for var in posthoc_results.keys():
    print(f'{var}:')
    
    groups = posthoc_results[var].keys()
    print(f'\t{len(groups)} groups')
    
    sizes = [len(posthoc_results[var][group]) for group in groups]
    print(f'\tSizes: {sizes}')



print('\n')
for var in posthoc_results.keys():
    print(f'{var}:')
    for group in posthoc_results[var].keys():
        if len(posthoc_results[var][group]) < 5:
            print(f'\nGroup {group}: {posthoc_results[var][group]}')
        else:
            print(f'\nGroup {group}: {posthoc_results[var][group][:5]}...')
        
