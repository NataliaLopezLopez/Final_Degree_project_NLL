"""
This script defines a class called `eeg` that I use to load, preprocess, and analyse EEG data
from multiple subjects. It's one of the main components of my Bachelor's Thesis project,
which focuses on comparing Eyes Open (EO) vs Eyes Closed (EC) brain states using
Spatial Permutation Entropy (SPE).

Key functionalities:
--------------------
- Load EEG signals from .edf files using MNE (raw, filtered, or with notch filtering).
- Reorganize electrode data into spatial grids (64, 31, or 17 channels) to apply spatial analysis.
- Compute different variants of Spatial Permutation Entropy (SPE), including pooled and time-resolved versions.
- Calculate basic statistical features per channel (mean, variance, skewness, kurtosis, MAD, IQR, autocorrelation, etc).
- Contains helper functions for calculating ordinal patterns and entropy values.

Usage:
------
This file is imported and used in my main analysis scripts, where I compute entropy measures
and compare different preprocessing or spatial arrangements.

Data:
-----
The EEG data comes from the PhysioNet dataset. Each subject has recordings in two conditions:
1 = Eyes Open (EO), 2 = Eyes Closed (EC).


"""

import mne
import math
import numpy as np
from scipy.stats import skew, kurtosis # important to do the skwness and kurtosis


class eeg:

    def __init__(self,subjects,mode,run):
        self.subjects = subjects #number of subjects
        self.mode=mode #raw or filt
        self.run=run #number of experiment: 1 corresponds to Eyes Open, and 2 to Eyes Closed
        #self.max_time=9600 #maximum time fo the experiment
        self.max_time= 9440
        self.i=[]
        self.j=[]
        self.structured_data=[]
        self.L=3
        self.lag = 1
        self.Lx=self.L
        self.Ly=1
        self.file_path='/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2'

        self.cut_low = []
        self.cut_up = []
        self.raw=[]

    def load_data(self):
        R=self.run
        self.data=[]
        if self.mode=='raw':
            for subject_number in range(self.subjects):
                   
                if subject_number>=99:
                    name=self.file_path+"/S"+str(subject_number+1)+"/S"+str(subject_number+1)+"R0"+str(R)+".edf"
                elif subject_number>=9:
                    name=self.file_path+"/S0"+str(subject_number+1)+"/S0"+str(subject_number+1)+"R0"+str(R)+".edf"
                else:
                    name=self.file_path+"/S00"+str(subject_number+1)+"/S00"+str(subject_number+1)+"R0"+str(R)+".edf"
                raw = mne.io.read_raw_edf(name,verbose=None)

                self.raw = raw
                self.data=self.data+[raw.get_data()]
    

        elif self.mode=='filt':
            for subject_number in range(self.subjects):
                        
                '''if subject_number == 96:
                  
                    continue '''
                if subject_number>=99:
                    name=self.file_path+"/S"+str(subject_number+1)+"/S"+str(subject_number+1)+"R0"+str(R)+".edf"
                elif subject_number>=9:
                    name=self.file_path+"/S0"+str(subject_number+1)+"/S0"+str(subject_number+1)+"R0"+str(R)+".edf"
                else:
                    name=self.file_path+"/S00"+str(subject_number+1)+"/S00"+str(subject_number+1)+"R0"+str(R)+".edf"
                raw = mne.io.read_raw_edf(name,verbose=None)

                self.data=self.data+[mne.filter.filter_data(data=raw.get_data(), sfreq=160, l_freq= self.cut_low, h_freq= self.cut_up)]
        
        elif self.mode=='notch':
            freqs = self.cut_low - ((self.cut_low - self.cut_up)/2)
            ancho = -(self.cut_low - self.cut_up)

            for subject_number in range(self.subjects):
                        
                '''if subject_number == 96
                  
                    continue '''
                if subject_number>=99:
                    name=self.file_path+"/S"+str(subject_number+1)+"/S"+str(subject_number+1)+"R0"+str(R)+".edf"
                elif subject_number>=9:
                    name=self.file_path+"/S0"+str(subject_number+1)+"/S0"+str(subject_number+1)+"R0"+str(R)+".edf"
                else:
                    name=self.file_path+"/S00"+str(subject_number+1)+"/S00"+str(subject_number+1)+"R0"+str(R)+".edf"
                raw = mne.io.read_raw_edf(name,verbose=None)

                self.data=self.data+[mne.filter.notch_filter(raw.get_data(), 160, freqs= freqs, notch_widths= ancho )]
        

        else:
            raise Exception("Load mode not specified or incorrect, Mode has to be 'raw' or 'filt'")
            
        '''if self.subjects>=96:
             self.subjects=self.subjects-1
             
             print('Subjects number changed to: '+str(self.subjects))'''
        
    def create_data_struc(self,data):
        #This function gives the grid arrangement as in 
        #Gancio, J., Masoller, C., & Tirabassi, G. (2024). Permutation entropy analysis of EEG signals for distinguishing eyes-open and eyes-closed brain states: Comparison of different approaches. Chaos: An Interdisciplinary Journal of Nonlinear Science, 34(4).
        
        row_len=[3,5,9,9,11,9,9,5,3,1]
        new_order=[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
                   1,2,3,4,5,6,7,40,43,41,8,9,10,11,12,13,14,42,44,45,15,
                   16,17,18,19,20,21,46,47,48,49,50,51,52,53,54,55,56,57,
                   58,59,60,61,62,63,64]
        
        new_data=[math.nan]*(len(row_len)*max(row_len))
        new_data=np.array(new_data).reshape(len(row_len),max(row_len))
        mid=(max(row_len)-1)/2-1
        counter=0
        for j in range(len(row_len)):
            for i in range(row_len[j]):
                new_data[j,int(mid-(row_len[j]-1)/2+i+1)]=data[new_order[counter]-1]
                counter+=1
        return new_data
    
    def create_data_struc_31(self, data):
        # Esta función organiza 30 electrodos en una rejilla según una disposición definida manualmente.
        row_len = [3, 5, 5, 5, 5, 5, 3]
        new_order = [22, 23, 24, 30, 32, 34, 36, 38, 39,
                    2, 4, 6, 40, 43, 9, 11, 13, 44, 45,
                    16, 18, 20, 46, 56, 49, 51, 53, 55,
                    61, 62, 63]

        new_data = [math.nan] * (len(row_len) * max(row_len))
        new_data = np.array(new_data).reshape(len(row_len), max(row_len))
        mid = (max(row_len) - 1) / 2 - 1
        counter = 0
        for j in range(len(row_len)):
            for i in range(row_len[j]):
                new_data[j, int(mid - (row_len[j] - 1) / 2 + i + 1)] = data[new_order[counter] - 1]
                counter += 1
        return new_data
    
    def create_data_struc_17(self, data):
        row_len = [3,3,5,3,3]
        new_order = [22,23,24,32,34,36,41,9,11,13,42,49,51,53,61,62,63]
        new_data = [math.nan] * (len(row_len) * max(row_len))
        new_data = np.array(new_data).reshape(len(row_len), max(row_len))
        
        mid = (max(row_len) - 1) / 2 - 1
        counter = 0
        
        for j in range(len(row_len)):
            for i in range(row_len[j]):
                col = int(mid - (row_len[j] - 1) / 2 + i + 1)
                new_data[j, col] = data[new_order[counter] - 1]
                counter += 1
        return new_data

    

    
    def set_mode(self,mode):
        if mode == 'vertical':
            self.Lx=1
            self.Ly=self.L
        elif mode == 'horizontal':
            self.Lx=self.L
            self.Ly=1
        else:
            raise Exception("Analysis mode not specified or incorrect, Mode has to be 'horizontal' or 'vertical'")
      
    
    def spatial_code(self,data):
        code=[]
        for j in range(data.shape[0]-(self.Ly-1)*self.lag):
            for i in range(data.shape[1]-(self.Lx-1)*self.lag):

                word=data[j+np.arange(self.Ly)*self.lag,i+np.arange(self.Lx)*self.lag]

                if not(np.isnan(word).any()):
                    code.extend(perm_indices(word,self.L,lag=1))
             
        return code
    
    def par_spatial(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc(new_data)
                    
            code=self.spatial_code(structured_data)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return np.mean(Ht)
    
    ########### para el montage de 30 electrodos. 
    def par_spatial_31_elect(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc_31(new_data)
                    
            code=self.spatial_code(structured_data)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return np.mean(Ht)
    
    
    def par_spatial_17_elect(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc_17(new_data)
                    
            code=self.spatial_code(structured_data)
    
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return Ht
    



    
    def par_pool_SPE(self,j):
        #Gets pooled spatial entropy (PSPE) of subject j
                    
        code=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc_17(new_data)
            code.extend(self.spatial_code(structured_data))
        
        probs=probabilities(code,self.L)
        
        return entropy(probs)/np.log(math.factorial(self.L))

    
    def par_spatial_2(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc_312(new_data)
                    
            code=self.spatial_code(structured_data)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return Ht
    
    def boaretto_best(self,data):
        #This function orders the data according to the best ordering reported in 
        #Boaretto, B. R., Budzinski, R. C., Rossi, K. L., Masoller, C., & Macau, E. E. (2023). Spatial permutation entropy distinguishes resting brain states. Chaos, Solitons & Fractals, 171, 113453.
        
        new_order=[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
                   1,2,3,4,5,6,7,40,43,41,8,9,10,11,12,13,14,42,44,45,15,
                   16,17,18,19,20,21,46,47,48,49,50,51,52,53,54,55,56,57,
                   58,59,60,61,62,63,64]
        
        new_data=[math.nan]*(len(new_order))

        for i in range(len(data)):
            
            new_data[i]=data[new_order[i]-1]
                
        return new_data
    
    def par_spatial_boaretto(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            structured_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.boaretto_best(structured_data)
                    
            code=perm_indices(structured_data,self.L,self.lag)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return np.mean(Ht)
    
    def par_PE(self,j):
        # Gets average usual permutation entropy (PE) of subject j
        Ht=[]
        for i in range(64):
            selected_data=self.data[j][i,:self.max_time] #Get channels i for all times 
            code=perm_indices(selected_data,self.L,self.lag)

            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        return np.mean(Ht)
    
    
    def PE_chanel(self,j):
        # Gets average usual permutation entropy (PE) of subject j
        Ht=[]
        for i in range(64):
            selected_data=self.data[j][i,:self.max_time] #Get channels i for all times 
            code=perm_indices(selected_data,self.L,self.lag)

            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        return Ht
    
    def mean_channel(self, j):
        mean_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  # Get channel i for all times
            mean_values.append(np.mean(selected_data))  # Compute mean
        return mean_values

    def variance_channel(self, j):
        variance_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  # Get channel i for all times
            variance_values.append(np.var(selected_data))  # Compute variance
        return variance_values

    def mad_channel(self, j):
        # Computes the Median Absolute Deviation (MAD) of each EEG channel for subject j
        mad_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  # Get channel i for all times
            mad_values.append(np.median(np.abs(selected_data - np.median(selected_data))))  # Compute MAD
        return mad_values
    
    def iqr_channel(self, j):
        # Computes the Interquartile Range (IQR) of each EEG channel for subject j
        iqr_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  # Get channel i for all times
            q75, q25 = np.percentile(selected_data, [75 ,25])
            iqr_values.append(q75 - q25)  # Compute IQR
        return iqr_values

    def skewness_channel(self, j):
        # Computes the skewness of each EEG channel for subject j
        skewness_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  # Get channel i for all times
            skewness_values.append(skew(selected_data))  # Compute skewness
        return skewness_values
    
    def kurtosis_channel(self, j):
        # Computes the kurtosis of each EEG channel for subject j
        kurtosis_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  # Get channel i for all times
            kurtosis_values.append(kurtosis(selected_data))  # Compute kurtosis
        return kurtosis_values
    
    def autocorr_channel(self, j):
        # Calcula la autocorrelación de cada canal EEG para el sujeto j
        autocorr_values = []
        for i in range(64):
            selected_data = self.data[j][i, :self.max_time]  
            autocorr_values.append(autocorr(selected_data, 2)[1]) 
        return autocorr_values
    
        
    
    def get_pos(self):
        subject_number = 100
        R=1 
        name=self.file_path+"/S"+str(subject_number+1)+"/S"+str(subject_number+1)+"R0"+str(R)+".edf"
        raw = mne.io.read_raw_edf(name,verbose=None)
        montage = mne.channels.make_standard_montage("biosemi64")
        dic = montage.get_positions()["ch_pos"]
        dic_new = dict()
        for i in dic:
            dic_new[i.upper()]= dic[i]
        pos = []
        for i in raw.ch_names:
            key= i[:-1].upper()
            if key[-1]== ".":
                key = key[:-1]
            if key == "T9":
                key = "P9"
            if key == "T10":
                key = "P10"
            pos.append(dic_new[key])
        return pos 
    

def autocorr(x,lags):

    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)
    return corr[:lags]

#autocorr(data,2)[1]

def perm_indices(ts, wl, lag):
    m = len(ts) - (wl - 1)*lag
    indcs = np.zeros(m, dtype=int)
    for i in range(1,wl):
        st = ts[(i - 1)*lag : m + ((i - 1)*lag)]
        for j in range(i,wl):
            zipped=zip(st,ts[j*lag : m+j*lag])
            indcs += [x > y for (x, y) in zipped]
        indcs*= wl - i
    return indcs + 1


def entropy(probs):
    h=0
    for i in range(len(probs)):
        if probs[i]==0:
            continue
        else:
            h=h-probs[i]*np.log(probs[i])
    return h

def probabilities(code,L):
    get_indexes = lambda x, xs: [k for (y, k) in zip(xs, range(len(xs))) if x == y]
    probs=[]
    for i in range(1,math.factorial(L)+1):
            
                
        probs=probs + [len(get_indexes(i,code))/len(code)
                   ]
        
        #print(self.entropy(probabilities))
    return probs 



