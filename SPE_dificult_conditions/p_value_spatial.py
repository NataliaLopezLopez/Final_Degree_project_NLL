from scipy import stats
import numpy as np
import multiprocess as mp
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
from egg_utils_2 import eeg  # Custom EEG analysis class

# Define analysis parameters
number_of_subjects = 109
filt_mode = 'raw'        # Use raw EEG data
word_length = 3          # Pattern length for entropy
lag = 1                  # Lag between points in a pattern
analysis_mode = 'spatial'  # Choose 'spatial' analysis

# Create EEG objects for EO and EC
eeg_open = eeg(number_of_subjects, filt_mode, run=1)   # Eyes Open
eeg_closed = eeg(number_of_subjects, filt_mode, run=2) # Eyes Closed

# Set attributes for both EEG conditions
for eeg_obj in [eeg_open, eeg_closed]:
    eeg_obj.L = word_length
    eeg_obj.lag = lag
    eeg_obj.file_path = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2'
    eeg_obj.cut_up = 30
    eeg_obj.cut_low = 12

# Load EEG data
eeg_open.load_data()
eeg_closed.load_data()

if analysis_mode == 'spatial':
    startTime = datetime.now()

    # Use multiprocessing to calculate SPE for each subject
    if __name__ == '__main__':
        with mp.Pool(mp.cpu_count()) as pool:
            # Horizontal SPE calculation for EO and EC
            eeg_open.set_mode("horizontal")
            eeg_closed.set_mode("horizontal")
            spe_hor_open = pool.map(eeg_open.par_spatial_2, range(eeg_open.subjects))
            spe_hor_closed = pool.map(eeg_closed.par_spatial_2, range(eeg_closed.subjects))

            # Vertical SPE calculation for EO and EC
            eeg_open.set_mode("vertical")
            eeg_closed.set_mode("vertical")
            spe_ver_open = pool.map(eeg_open.par_spatial_2, range(eeg_open.subjects))
            spe_ver_closed = pool.map(eeg_closed.par_spatial_2, range(eeg_closed.subjects))

    print('Spatial Analysis completed.')
    print('Time elapsed:', str(datetime.now() - startTime))

    # Convert lists to NumPy arrays
    spe_hor_open = np.stack([i for i in spe_hor_open])
    spe_hor_closed = np.stack([i for i in spe_hor_closed])
    spe_ver_open = np.stack([i for i in spe_ver_open])
    spe_ver_closed = np.stack([i for i in spe_ver_closed])

    # Perform t-tests between EO and EC for both configurations
    t_hor, p_hor = stats.ttest_ind(spe_hor_open, spe_hor_closed, equal_var=False)
    print(f'T-test Horizontal: t={t_hor}, p={p_hor}')

    t_ver, p_ver = stats.ttest_ind(spe_ver_open, spe_ver_closed, equal_var=False)
    print(f'T-test Vertical: t={t_ver}, p={p_ver}')

    print("T-tests for spatial analysis completed.")

    # Save results as .npy files
    np.save("spe_hor_open_raw_wo_31.npy", spe_hor_open)
    np.save("spe_hor_closed_raw_wo_31.npy", spe_hor_closed)
    np.save("spe_ver_open_raw_wo_31.npy", spe_ver_open)
    np.save("spe_ver_closed_raw_wo_31.npy", spe_ver_closed)
