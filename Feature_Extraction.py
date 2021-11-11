# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 19:42:50 2021

@author: Tejaswini, Dev & Anuj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.integrate as integrate
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
from antropy import spectral_entropy
import glob

datafile = open(os.getcwd() + "\\OptimalData.txt")
signals = datafile.read()
datafile.close()
signals = list(signals.split("\n"))
signals.remove(signals[len(signals) - 1])

# Create an empty list to store the extracted features
data_list = []  # For the final data storage
temp_list = []  # For temporary data storage

for i in range(len(signals)):
    
    loc = os.getcwd() + "\\Perfect PPG Data\\" + signals[i]
    f = open(loc, "r")
    y = list(f.read().split(","))
    
    # Introducing a try/except statement for error handling
    try:
        for i in range(0, len(y)):
            y[i] = float(y[i])
    
    except: continue
    
    # Finding the FFT of the signal to find F-Domain features
    fft_vals = np.fft.fft(y)
    mags = abs(fft_vals)/2100
    mags = mags[0 : int(2100/2 + 1)]
    mags[0] = mags[0]/2
    freq_plot = np.fft.fftfreq(2100, d = 0.001)
    freq_plot = freq_plot[0 : int(2100/2 + 1)]
    
    """
    # Plotting the FFT of the signal
    plt.figure(dpi = 120)
    plt.plot(freq_plot, mags, color = 'royalblue', linewidth = 2, label = 'FFT')
    plt.legend()
    plt.xlim(0, 10)
    plt.xlabel("Frequencies")
    plt.ylabel("Amplitude")
    plt.title("FFT of the signal")
    plt.show()
    """
    
    # =========================================================================
    # AREA UNDER THE CURVE
    freq_list = []
    for i in freq_plot:
        freq_list.append(round(i))
    index_02 = freq_list.index(2)
    index_05 = freq_list.index(5)
    
    # Finding out the area under the curves
    A0_2 = integrate.trapezoid(mags[0:index_02], freq_plot[0:index_02])
    A2_5 = integrate.trapezoid(mags[index_02:index_05], freq_plot[index_02: index_05])
    
    # Finding the ratio
    Ratio_A = A0_2/A2_5
    # =========================================================================
    
    # =========================================================================
    # PEAK INFORMATION
    peaks_indices = find_peaks(mags, height = 0.001)
    peaks_position = freq_plot[peaks_indices[0]]
    peaks_values = mags[peaks_indices[0]]
    
    """
    # Plotting the peaks
    plt.figure(dpi = 120)
    plt.plot(freq_plot, mags, color = 'royalblue', linewidth = 2)
    plt.scatter(peaks_position[0:3], peaks_values[0:3], color = 'orangered', 
                marker = 'o', s = 17, label = 'Peaks', zorder = 10)
    plt.legend()
    plt.xlim(0, 10)
    plt.xlabel("Frequencies")
    plt.ylabel("Power Spectral Density")
    plt.title("FFT of the filtered signal")
    plt.show()
    """

    P_12 = peaks_values[0]/peaks_values[1]
    f_12 = peaks_position[0]/peaks_position[1]
    
    # Not present for every signal
    # P_13 = peaks_values[0]/peaks_values[2]
    # f_13 = peaks_position[0]/peaks_position[2]
    
    fmax = peaks_position[0]
    mag_fmax = peaks_values[0]
    # =========================================================================
    
    # =========================================================================
    # STATISTICAL FEATURES
    mean = np.mean(y)
    median = np.median(y)
    st_dev = np.std(y)
    perc_25 = np.percentile(y, 25)
    perc_50 = np.percentile(y, 50)
    perc_75 = np.percentile(y, 75)
    mad = np.mean(np.absolute(y - np.mean(y)))
    
    Q1 = np.percentile(y, 25, interpolation = 'midpoint')
    Q3 = np.percentile(y, 75, interpolation = 'midpoint')
    IQR = Q3-Q1
    QD = IQR/2
    
    skewness = stats.skew(y)
    kurtosis = stats.kurtosis(y)
    
    pd_series = pd.Series(y)
    counts = pd_series.value_counts()
    shannon_entropy = stats.entropy(counts)
    
    spec_entropy = spectral_entropy(y, sf = 1000)
    # =========================================================================
    
    # =========================================================================
    # TIME DOMAIN FEATURES - Anuj
    
    def local_min(ys):
        '''
        ys: Input signal to be traversed
        Returns the local minimum of a given numeric array
        '''
        return [y for i, y in enumerate(ys)
                if ((i == 0) or (ys[i - 1] >= y))
                and ((i == len(ys) - 1) or (y < ys[i+1]))]
    
    def time_domain_extractor(loc):
        '''
        loc: List of data directories
        Returns a dictionary of data structures with Time domain features
        '''
        features = {
            "Systolic": {
                "Peak_indices": [],
                "Peak_values": [],
                "crest_factor": [],
                "impulse_factor": [],
                "clearance_factor": []},
            
            "Diastolic": {
                "Peak_indices": [],
                "Peak_values": [],
                "crest_factor": [],
                "impulse_factor": [],
                "clearance_factor": []},
            
            "Overall": {
                "coeff_of_var": [],
                "shape_factor": [],
                "P2P": [],
                "local_minima": [],
                "RMS": []}
            }
        
        for index in loc:
            
            # Indexing one signal sample at a time
            file = open(index)
            temp_reader = list(file.read().split(','))
            file.close()
            
            # Try/except for empty file error handling
            try:
                y = []
                for i in range(len(temp_reader)):
                    y.append(float(temp_reader[i]))
            except: continue
            
            # Calculating the root-mean square of the signal
            rms = np.sqrt(np.mean(np.square(y)))                
            features["Overall"]["RMS"].append(rms)
            
            # Coefficient of variance: Deviation of signal from it's mean
            features["Overall"]["coeff_of_var"].append(np.std(y)/np.mean(y))
            
            # Approximate Systolic Peak indices and amplitude
            [syspeak, prop_sys] = find_peaks(y,
                                             # Specified a boundary condition for Systolic peak
                                             height=np.mean(y),
                                             distance=1)    
            syspeak_amp = prop_sys['peak_heights']
            # syspeak_amp = [y[x] for x in syspeak]
            
            features["Systolic"]["Peak_indices"].append(syspeak)
            features["Systolic"]["Peak_values"].append(syspeak_amp)
            
            # Approximate diastolic peak indices and amplitude
            [diapeak, prop_dia] = find_peaks(y,
                                             # Specified a boundary condition for diastolic peak
                                             height=(0, np.mean(y)),    
                                             distance=1)
            diapeak_amp = prop_dia['peak_heights']         
            
            features["Diastolic"]["Peak_indices"].append(diapeak)
            features["Diastolic"]["Peak_values"].append(diapeak)
            
            # Crest Factor: deviation of peaks from the RMS value
            features["Systolic"]["crest_factor"].append(syspeak_amp/rms)
            features["Diastolic"]["crest_factor"].append(diapeak_amp/rms)
            
            # Clearance factor: Peak to Square of mean error for oscillatory systems
            features["Systolic"]["clearance_factor"].append(syspeak_amp/np.square(np.mean(np.sqrt(y))))
            features["Diastolic"]["clearance_factor"].append(diapeak_amp/np.square(np.mean(np.sqrt(y))))
            
            features["Overall"]["shape_factor"].append(rms/np.mean(y))
            features["Overall"]["P2P"].append(max(y) - min(y))
            features["Overall"]["local_minima"].append(local_min(y))
            
            # Impulse factor: Instantaneous amplitude ratio of Peak to signal's mean
            features["Systolic"]["impulse_factor"].append(syspeak_amp/np.mean(y))
            features["Diastolic"]["impulse_factor"].append(diapeak_amp/np.mean(y))
            
        return features
    
    file_loc = glob.glob(os.getcwd() + "\\Perfect PPG Data" + "\\*.txt")
    time_features = time_domain_extractor(file_loc)
    
    # =========================================================================
    
    # =========================================================================
    # FIRST AND SECOND DERIVATIVE FEATURES
    # Tejaswini put stuff here
    # =========================================================================
    
    # =========================================================================
    # Creating the dataframe
    
    temp_list = [A0_2, A2_5, Ratio_A, fmax, mag_fmax, mean, 
                 median, st_dev, perc_25, perc_50, perc_75, mad, IQR, QD,
                 skewness, kurtosis, shannon_entropy, spec_entropy]
    
    data_list.append(temp_list)
    

