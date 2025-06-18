import mne                      # Library used to work with EEG data
import matplotlib.pyplot as plt # For plotting the EEG signals

# --- 1. Load the EEG files (with and without artifacts) ---
file_with_artifacts = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2/S026/S026R01.edf'             # original/raw EEG signal
file_wo_artifacts   = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2/S026/S026R01_wo_artifacts.edf' # same signal after preprocessing

# Load both files using MNE
raw_with_artifacts = mne.io.read_raw_edf(file_with_artifacts, preload=True, verbose=False)
raw_wo_artifacts   = mne.io.read_raw_edf(file_wo_artifacts, preload=True, verbose=False)

# --- 2. Select the same channel in both datasets (Fp1) ---
# Important: in the raw file the channel name has a dot at the end
channel_with_artifacts = 'Fp1.'
channel_wo_artifacts   = 'Fp1'

# Extract the signal and corresponding time points
signal_with_artifacts, times = (
    raw_with_artifacts.copy()
    .pick_channels([channel_with_artifacts])
    .get_data(return_times=True)
)
signal_wo_artifacts = (
    raw_wo_artifacts.copy()
    .pick_channels([channel_wo_artifacts])
    .get_data()
)

# --- 3. Plot both signals to visually compare them ---
plt.figure(figsize=(16, 8))
plt.plot(times[:1000], signal_with_artifacts[0, :1000], label='With artifacts', linewidth=3)
plt.plot(times[:1000], signal_wo_artifacts[0, :1000], label='Without artifacts', linewidth=3)

# Add axis labels, title, and legend (with large font sizes for visibility in the TFG)
plt.xlabel('Time (s)', fontsize=28)
plt.ylabel('Amplitude (µV)', fontsize=28)
plt.title('EEG Signal at Fp1 - Subject S026 (Eyes Open)', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout()

# --- 4. Save the figure in high resolution ---
output_path = '/.../Fp1_S026_EO_HighRes.png'
plt.savefig(output_path, dpi=400)
plt.show()
