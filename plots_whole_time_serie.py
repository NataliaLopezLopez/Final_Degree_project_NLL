from tqdm import tqdm  # For progress bar during loops
import numpy as np
import multiprocess as mp  # To parallelize entropy calculations
from datetime import datetime
import matplotlib.pyplot as plt
from egg_utils_2 import eeg  # Custom class to load and process EEG data
import mne

# --------------------------------------------------
# 1. Define analysis parameters
# --------------------------------------------------

number_of_subjects = 109  # Total number of EEG subjects to process (max 109)
# Subjects 97 and 109 may be excluded due to invalid signal data

filt_mode = 'raw'  # Choose 'raw' or 'filt' if filtering in alpha band

word_length = 4  # Word length for permutation patterns
lag = 1  # Time lag between samples used for pattern generation

analysis_mode = 'spatial'  # Analysis mode: 'ensemble', 'spatial', or 'temporal'

# --------------------------------------------------
# 2. Create EEG objects for EO and EC conditions
# --------------------------------------------------

eeg_open = eeg(number_of_subjects, filt_mode, run=1)    # Eyes Open data
eeg_closed = eeg(number_of_subjects, filt_mode, run=2)  # Eyes Closed data

# Set parameters for both objects
for eeg_obj in [eeg_open, eeg_closed]:
    eeg_obj.L = word_length
    eeg_obj.lag = lag
    eeg_obj.file_path = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2'
    eeg_obj.cut_up = 30
    eeg_obj.cut_low = 12

# --------------------------------------------------
# 3. Load EEG data for both conditions
# --------------------------------------------------

eeg_open.load_data()
eeg_closed.load_data()

# --------------------------------------------------
# 4. Analyze entropy at different time windows
# --------------------------------------------------

open_promedio = []  # Mean PE for EO
open_sd = []        # Std dev PE for EO

closed_promedio = []  # Mean PE for EC
closed_sd = []        # Std dev PE for EC

time = [10, 20, 30, 40, 50, 59]  # Window sizes in seconds

for i in tqdm(time):
    eeg_open.max_time = i * 160     # Set window size in samples (160 Hz)
    eeg_closed.max_time = i * 160

    if analysis_mode == 'spatial':
        startTime = datetime.now()

        # Compute PE for EO using multiprocessing
        if __name__ == '__main__':
            pool = mp.Pool(mp.cpu_count())
            pe_eyes_open = pool.map(eeg_open.PE_chanel, range(eeg_open.subjects))
            pool.close()
            pool.join()
        
        open_promedio.append(np.mean(pe_eyes_open))
        open_sd.append(np.std(pe_eyes_open))

        # Compute PE for EC using multiprocessing
        if __name__ == '__main__':
            pool = mp.Pool(mp.cpu_count())
            pe_eyes_closed = pool.map(eeg_closed.PE_chanel, range(eeg_closed.subjects))
            pool.close()
            pool.join()

        closed_promedio.append(np.mean(pe_eyes_closed))
        closed_sd.append(np.std(pe_eyes_closed))

# --------------------------------------------------
# 5. Print execution time and status
# --------------------------------------------------

print('Time elapsed:' + str(datetime.now() - startTime))
print('Process completed.')

# --------------------------------------------------
# 6. Plot mean PE over time for EO and EC
# --------------------------------------------------

plt.fill_between(time, np.array(open_promedio) - np.array(open_sd), np.array(open_promedio) + np.array(open_sd), alpha=0.5)
plt.plot(time, open_promedio, 'b', label='Eyes Open')

plt.fill_between(time, np.array(closed_promedio) - np.array(closed_sd), np.array(closed_promedio) + np.array(closed_sd), alpha=0.5, color='r')
plt.plot(time, closed_promedio, 'r', label='Eyes Closed')

plt.xlabel('Time window (s)')
plt.ylabel('Mean Permutation Entropy')
plt.title('Evolution of PE over different time durations')
plt.legend()

# Save plot
plt.savefig("eeg_plot.png", dpi=300, bbox_inches='tight')

# --------------------------------------------------
# 7. Save final PE results for later use
# --------------------------------------------------

np.save('EC_PE_RAW_WO_50', np.array(pe_eyes_closed))
np.save('EO_PE_RAW_WO_50', np.array(pe_eyes_open))