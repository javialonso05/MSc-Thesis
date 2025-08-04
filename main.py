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
