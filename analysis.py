import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# Load data
freq = np.load('Data/frequencies.npy')
if freq.shape == (2, 10000):
    print(f'Freq is stored separately')
    freq = np.hstack((freq[0], freq[1]))

subtraction_signal = np.load('Data/subtraction_shifted_array.npy')

data_info = pd.read_csv('Data/data_info.csv')
velocity_info = pd.read_csv('Data/velocity_info.csv')

labels, sizes = np.unique(velocity_info['CC_label'], return_counts=True)
print(f'{sum(sizes >= 20)} clusters with more than 10 signals')

# Plot distributions
from src.analysis.cluster_analysis import plot_cluster_distributions
core_info = velocity_info[['Source', 'Core', 'CC_label']]
core_info.columns = ['Source', 'Core', 'Labels']

plot_cluster_distributions(data_info, core_info, variable='Mcore', min_cluster_size=20)