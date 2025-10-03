# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_processor import shift_signal, filter_data

# Load data
freq = np.load("Data/frequencies.npy")
signals = np.load("Data/intensity_array.npy")
residuals = np.load("Data/residual_array.npy")

peaks = [
            218222.192,  # H2CO
            218475.632,  # H2CO
            219560.358,   # C18O(2-1)
            219949.433,  # SO(5,6-4,5)
            220398.684,  # 13CO(2-1)
        ]

peak_labels = [
    r"H$_2$CO",
    r"H$_2$CO",
    r"C$^{18}$O",
    r"SO",
    r"$^{13}$CO",
]


# Read velocity table
velocity_table = pd.read_csv("Data/velocity_info.csv")

# %%
# from src.data.data_processor import RedShiftCorrector
# new_velocities = RedShiftCorrector(frequency=freq).fit(signals, mapping, residuals)


v_diff = velocity_table["Manual velocity"] - velocity_table["Automatic velocity"]
v_diff = v_diff.dropna()
print(f"{np.sum(np.abs(v_diff < 5))}/{len(v_diff)} ({np.round(100*np.sum(np.abs(v_diff < 5))/len(v_diff), 3)})")

from src.features.transformations import shift_signal

v0 = velocity_table["Manual velocity"]
v1 = velocity_table["Automatic velocity"]

shifted_0 = np.array([shift_signal(freq, signals[i], v0.iloc[i]) for i in range(len(v0)) if not np.isnan(v0.iloc[i])])
shifted_1 = np.array([shift_signal(freq, signals[i], v1.iloc[i]) for i in range(len(v1))])

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
ax[0].plot(freq, shifted_0.mean(axis=0))
ax[1].plot(freq, shifted_1.mean(axis=0))
fig.supxlabel('Frequency [MHz]', fontsize=14)
fig.supylabel('Intensity [Jy]', fontsize=14)
fig.tight_layout()
plt.show()


# %%
from moviepy import ImageSequenceClip

editor = ImageSequenceClip("Data/velocity_correction.mp4")


