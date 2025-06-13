import matplotlib.pyplot as plt
import numpy as np
import mne  # EEG analysis library

# --------------------------------------------------
# Load EEG data from EDF file (subject S001)
# --------------------------------------------------
name = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2/S001/S001R01.edf'
raw = mne.io.read_raw_edf(name, verbose=None)

# Plot the raw EEG to visually inspect it
raw.plot()

# Load the data into memory
raw.load_data()

# --------------------------------------------------
# Set the electrode layout (montage)
# --------------------------------------------------
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005")

# (Optional) pick channels for visualizing specific artifacts (empty here)
artifact_picks = mne.pick_channels(raw.ch_names, [])
# raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# --------------------------------------------------
# Apply a high-pass filter to remove slow drift (< 1 Hz)
# --------------------------------------------------
filt_raw = raw.copy().filter(l_freq=1, h_freq=None)

# --------------------------------------------------
# Perform ICA decomposition to identify independent sources
# --------------------------------------------------
ica = mne.preprocessing.ICA(n_components=5, max_iter="auto", random_state=97)
ica.fit(filt_raw)

# Visualize sources (time series and topographic layout)
ica.plot_sources(filt_raw)
ica.plot_components()

# --------------------------------------------------
# Mark components for exclusion (based on visual inspection)
# --------------------------------------------------
ica.exclude = [0]  # Component 0 identified as artifact (e.g., blink)

# --------------------------------------------------
# Apply ICA to remove the selected components from original data
# --------------------------------------------------
reconst_raw = raw.copy()
ica.apply(reconst_raw)

# --------------------------------------------------
# (Optional) Save the reconstructed data without artifacts
# --------------------------------------------------
# fname = name[:-4] + '_wo_artifacts.edf'
# mne.export.export_raw(fname, reconst_raw, overwrite=True)

# --------------------------------------------------
# Plot original and cleaned EEG for comparison (optional)
# --------------------------------------------------
# reconst_raw.plot()
