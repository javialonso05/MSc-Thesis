import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_processor import shift_signal, filter_data

# Load data
freq = np.load("Data/frequencies.npy")
signals = np.load("Data/intensity_array.npy")
residuals = np.load("Data/residual_array.npy")
mapping = np.load("Data/mapping.npy")

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
# velocity_table = pd.read_csv('Data/velocity_table.csv')
# velocity_table.columns = ['source', 'nameAG', 'core_ID', 'v_clump', 'v_core', 'line', 'chi2', 'v_mean_g', 'group', 'v_mean_all', 'v_std']
# velocity_table.to_csv("Data/velocity_table.csv", index=False)

# from src.data.data_processor import RedShiftCorrector
velocities = pd.read_csv('Data/velocity_info.csv')
# corrector = RedShiftCorrector(freq)
# bisection_velocities = RedShiftCorrector(frequency=freq).fit(raw_data=signals, mapping=mapping, residual=residual, method='bisection')
# bisection_velocities.to_csv("Data/velocity_info_bisection.csv", index=False)

shifted_signals = np.load("Data/shifted_signals.npy")
shifted_residual = np.load("Data/shifted_residual.npy")

subtraction_signals = filter_data(shifted_signals, "subtraction", shifted_residual)
threshold_signals = filter_data(shifted_signals, "sigma", shifted_residual)

from umap import UMAP
import seaborn as sns

umap_raw = UMAP(n_neighbors=5, metric='cosine', random_state=42).fit_transform(shifted_signals)
umap_sig = UMAP(n_neighbors=5, metric='cosine', random_state=42).fit_transform(threshold_signals)
umap_sub = UMAP(n_neighbors=5, metric='cosine', random_state=42).fit_transform(subtraction_signals)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))

ax[0, 0].set_title("Raw")
ax[0, 0].scatter(umap_raw[:, 0], umap_raw[:, 1], color='tab:blue', alpha=0.3)

ax[0, 1].set_title("Threshold-filtered")
ax[0, 1].scatter(umap_sig[:, 0], umap_sig[:, 1], color='tab:orange', alpha=0.3)

ax[0, 2].set_title("Subtraction-filtered")
ax[0, 2].scatter(umap_sub[:, 0], umap_sub[:, 1], color='tab:green', alpha=0.3)

sns.kdeplot(x=umap_raw[:, 0], y=umap_raw[:, 1], fill=True, cmap="Blues", ax=ax[1, 0])
sns.kdeplot(x=umap_sig[:, 0], y=umap_sig[:, 1], fill=True, cmap="Oranges", ax=ax[1, 1])
sns.kdeplot(x=umap_sub[:, 0], y=umap_sub[:, 1], fill=True, cmap="Greens", ax=ax[1, 2])

fig.tight_layout()
fig.savefig("results/umap_projection.pdf")
plt.show()