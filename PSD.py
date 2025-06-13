import numpy as np
import multiprocess as mp
from datetime import datetime
import matplotlib.pyplot as plt
from egg_utils_2 import eeg

# 1. Here we are defining the parameters 

number_of_subjects = 1 #shouldn't be larger that 108, dataset has 109 but subject 109 has some not valid values at the end (this should be automated by some error control)
# if larger than 97, the actual number of subjects is number_of_subjects-1, since subject 97 is removed for the same reasons as subject 109
filt_mode = 'notch' # 'raw' or 'filt', for considering only the alpha band
word_length = 3 # Word length
lag = 1 # Spatial lag
analysis_mode = 'temporal' # Controls the symbol construction (temporal or spatial) 
# and how the averages are performed
# 'ensemble' : spatial analysis, but for each time, the mean value of SPE in all subjects if provided
# This analysis corresponds to the one of Boaretto et al. (2023), and outputs the data used to
# plot Fig.4 of Gancio et al. (2024). It provides SPE for diferente arrengement:
# linear: straight foward, as come from the dataset, ordering (first approa of Boaretto et al. (2023))
# boaretto's best: best arrengement of electrodes find by Boaretto et al. (2023)
# horizontal symbols
# vertical symbosl
#
# 'spatial': Spatial analysis as performed by Gancio et al. (2024), an average in time of the SPE values is provided 
# (one averaged quantity for each subject). This provied the data for Fig. 5,and partial of Figs. 7 and 8 of  Gancio et al. (2024)
#
# 'temporal':  Temporal analysis as performed by Gancio et al. (2024), an average in space of the Permutation Entropy (PE)
#  values is provided (one averaged quantity for each subject). This provied the data for Fig. 6, and the additional
# data for Figs. 7 and 8 of  Gancio et al. (2024)


# 2. Here what we are doing is to create the objects 

eeg_open = eeg(number_of_subjects, filt_mode, run=1)  # Open eyes 
eeg_closed = eeg(number_of_subjects, filt_mode, run=2)  # Closed eyes 

# Parameters 
for eeg_obj in [eeg_open, eeg_closed]:
    eeg_obj.L = word_length
    eeg_obj.lag = lag
    eeg_obj.file_path = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2'
    eeg_obj.cut_up = 12
    eeg_obj.cut_low = 8


# 3. Here what we are doing is to load the data. 

eeg_open.load_data()
eeg_closed.load_data()

# psd. 

from scipy.fft import fft, fftfreq, ifft

y = eeg_closed.data[0][30,:]
#y = y - np.mean(y)

N=len(y)
T= 1/160 # periodo de muestreo o el intervalo. 
yf= np.abs(fft(y))**2 # magnitude 

xf = fftfreq(N, T)[:N//2] # frequence value 

plt.semilogy(xf[1:N//2], 1.0/N * yf[1:N//2], '-b')
#plt.plot([12.5e9,12.5e9],[1,.01],'r-')
plt.xlim((0,50))
plt.ylim((1e-15,1e-7))
plt.title('PSD')
plt.grid(); 

plt.savefig("psd_output.png", dpi=300, bbox_inches='tight')  # Guardar antes de mostrar
plt.show()

