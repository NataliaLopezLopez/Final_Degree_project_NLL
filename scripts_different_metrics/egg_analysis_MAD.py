import numpy as np
import multiprocess as mp
from datetime import datetime
import matplotlib.pyplot as plt
from egg_utils_2 import eeg
import mne

# 1. Here we are defining the parameters 

number_of_subjects = 109
filt_mode = 'raw' 
word_length = 4
lag = 1 
analysis_mode = 'temporal' 

# 2. Here what we are doing is to create the objects 

eeg_open = eeg(number_of_subjects, filt_mode, run=1)  # Open eyes 
eeg_closed = eeg(number_of_subjects, filt_mode, run=2)  # Closed eyes 

# Parameters 
for eeg_obj in [eeg_open, eeg_closed]:
    eeg_obj.L = word_length
    eeg_obj.lag = lag
    eeg_obj.file_path = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2'
    eeg_obj.cut_up = 8
    eeg_obj.cut_low = 4

# 3. Here what we are doing is to load the data. 

eeg_open.load_data()
eeg_closed.load_data()

# 4. Here we analyze the data

if analysis_mode == 'temporal':
    startTime = datetime.now()

    # open 
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count())
        mad_eyes_open = pool.map(eeg_open.mad_channel, range(eeg_open.subjects))
        pool.close()
        pool.join()
    av_open = np.mean(np.array(mad_eyes_open),axis=0)
    print('MAD Eyes Open - mean =', np.mean(mad_eyes_open), ', std =', np.std(mad_eyes_open))

    # closed 
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count())
        mad_eyes_closed = pool.map(eeg_closed.mad_channel, range(eeg_closed.subjects))
        pool.close()
        pool.join()
    av_closed = np.mean(np.array(mad_eyes_closed),axis=0)
    print('MAD Eyes Closed - mean =', np.mean(mad_eyes_closed), ', std =', np.std(mad_eyes_closed))

    # time of the execution 
    print('Time elapsed:' + str(datetime.now() - startTime))
    print('Process completed.')


np.save('EC_mad_raw',np.array(mad_eyes_closed))
np.save('EO_mad_raw',np.array(mad_eyes_open))

# 5. VISUALIZATION mne.viz.plot_topomap

# Assign the average MAD values for Eyes Open (EO) and Eyes Closed (EC)
mad_eyes_open_avg = av_open
mad_eyes_closed_avg = av_closed

# Get electrode positions
positions = np.array(eeg_open.get_pos())[:, :-1]

# Compute the difference
mad_dif = mad_eyes_open_avg - mad_eyes_closed_avg

# Determine color range
vmin_eo_ec = min(mad_eyes_open_avg.min(), mad_eyes_closed_avg.min())
vmax_eo_ec = max(mad_eyes_open_avg.max(), mad_eyes_closed_avg.max())

vmin_diff, vmax_diff = mad_dif.min(), mad_dif.max()

fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

# Graph 1: EO 
im_eo, _ = mne.viz.plot_topomap(
    data=mad_eyes_open_avg,
    pos=positions,
    cmap='plasma',
    contours=10,
    image_interp='cubic',
    vlim=(vmin_eo_ec, vmax_eo_ec),
    axes=axes[0],
    show=False
)
axes[0].set_title("EO (Eyes Open)")

# Graph 2: EC 
im_ec, _ = mne.viz.plot_topomap(
    data=mad_eyes_closed_avg,
    pos=positions,
    cmap='plasma',
    contours=10,
    image_interp='cubic',
    vlim=(vmin_eo_ec, vmax_eo_ec),
    axes=axes[1],
    show=False
)
axes[1].set_title("EC (Eyes Closed)")

# Graph 3: Difference
im_diff, _ = mne.viz.plot_topomap(
    data=mad_dif,
    pos=positions,
    cmap='coolwarm',
    contours=10,
    image_interp='cubic',
    vlim=(vmin_diff, vmax_diff),
    axes=axes[2],
    show=False
)
axes[2].set_title("Difference EO and EC")

cbar = fig.colorbar(im_eo, ax=[axes[0], axes[1]], orientation='horizontal', fraction=0.05, pad=0.2)
cbar.set_label("MAD EEG Values (EO and EC)")

cbar_diff = fig.colorbar(im_diff, ax=axes[2], orientation='horizontal', fraction=0.05, pad=0.2)
cbar_diff.set_label("MAD EEG Values (Difference)")

plt.tight_layout()
plt.savefig("MAD_topomap_corrected_range_raw.png", dpi=300)
plt.show()
