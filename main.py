import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# Red-shift correction
import numpy as np
import pandas as pd

## Load data
freq = np.load('Data/frequencies.npy')
if freq.shape == (2, 10000):
    print(f'Freq is stored separately')
    freq = np.hstack((freq[0], freq[1]))

intensity_array = np.load('Data/intensity_array.npy')
residual_array = np.load('Data/residual_array.npy')

## Correct velocity
from src.data.data_processor import RedShiftCorrector
corrector = RedShiftCorrector(freq[10000:])
velocity_info = corrector.fit(intensity_array, mapping=np.load('Data/mapping.npy'), residual=residual_array)
velocity_info.to_csv('Data/velocity_info.csv', index=False)
