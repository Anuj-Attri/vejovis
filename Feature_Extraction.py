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
import re
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.kernel_regression import KernelReg

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
    
    '''
    Introducing a try/except statement for Type conversion 
    and Missing data error handling.
    '''    
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
    try:
        P_12 = peaks_values[0]/peaks_values[1]
        f_12 = peaks_position[0]/peaks_position[1]
        
    except:
        P_12 = float("NaN")     # Assign NaN value in case of missing data
        f_12 = float("NaN")
    
    try:
        P_13 = peaks_values[0]/peaks_values[2]
        f_13 = peaks_position[0]/peaks_position[2]
        
    except:
        P_13 = float("NaN")     # Assign NaN value in case of missing data
        f_13 = float("NaN")
    
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
    
def derivative_features_extractor(loc):
    derivative_features = {
    "First_derivative": {
        "Peaks":[],
        "Valleys": []},
    "Second_derivative": {
        "Peaks":[],
        "Valleys": []},
    "Time_intervals": {
        "Peak_to_peak":[],
        "First_derivative_peak": [],
        "Second_derivative_peak": [],
        "First_derivative_valley": [],
        "Second_derivative_valley": []},
    "Ratios": {
        "b1_a1":[],
        "b2_a2": [],
        "ta1_tpp": [],
        "ta2_tpp": [],
        "tb1_tpp": [],
        "tb2_tpp": [],
        "ta12_tpp": [],
        "tb12_tpp": []} 
    }
    
    # Create an empty list to store the extracted features
    for i in range(len(signals)):
        loc = os.getcwd() + "\\Perfect PPG Data\\" + signals[i]
        f = open(loc, "r")
        y = list(f.read().split(","))
        
        try:
            for i in range(0, len(y)):
            y[i] = float(y[i])
            
        except: continue
        
        m=y.shape[0]
        x = np.linspace(0, 2.1, m)

        cs = CubicSpline(x, y)
        d = cs(x,1)
        dd = cs(x,2)

        kr1 = KernelReg(y,x,'c')
        kr2 = KernelReg(d,x,'c')
        kr3 = KernelReg(dd,x,'c')
        f, g = kr1.fit(x)
        dy, dg = kr2.fit(x)
        ddy, ddg = kr3.fit(x)

        '''
        #Plotting derivatives
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1, 1, 1)
        plt.figure(figsize=(15,5))
        ax.plot(x, f)
        ax.plot(x, dy)
        ax.plot(x, ddy)
        plt.show()
        '''

        #Finding features
        peak1, _ = find_peaks(dy, distance = 800)      #array of peaks of first derivative
        peak2, _ = find_peaks(ddy, distance = 800)     #array of peaks of second derivative
        peak3, _ = find_peaks(y, distance = 800)       #array of peaks of ppg
        valley1, _ = find_peaks(-dy, distance = 800)   #array of valleys of first derivative
        valley2, _ = find_peaks(-ddy, distance = 800)  #array of valleys of second derivative
        valley3, _ = find_peaks(-y, distance = 800)    ##array of valleys of ppg

        '''
        #Plotting features
        plt.figure(figsize=(15,5))

        plt.plot(peak1, dy[peak1], "yo");
        plt.plot(valley1, dy[valley1], "sy");
        plt.plot(dy, label='$First Derivative$');

        plt.plot(peak2, ddy[peak2], "go");
        plt.plot(valley2, ddy[valley2], "sg");
        plt.plot(ddy, label='$Second Derivative$');

        plt.plot(peak3, y[peak3], "bo");
        plt.plot(valley3, y[valley3], "sb");
        plt.plot(y, label='$PPG Signal$');

        plt.grid()
        plt.legend()
        plt.show()
        '''

        #Feature values
        a1 = dy[peak1[0]]               # first maximum peak from the first derivative
        p = x[peak1[0]]

        for i in peak2:                 #first maximum peak from the second derivative after a1
            if i>peak1[0]:
                a2 = ddy[i]
                q = x[i]
                break

        for i in valley1:               #first minimum peak from the first derivative after a1
            if i>peak1[0]:
                b1 = dy[i]
                r = x[i]
                break

        for i in valley2:               #first minimum peak from the second derivative after a2
            if i>q:
                b2 = ddy[i]
                s = x[i]
                break

        for i in valley3:               #foot of ppg
            if i>p:
                f1 = x[i]
                break

        for i in valley3:               #foot of ppg
            if i>q:
                f2 = x[i]
                break

        ta1 = f1 - p                 #time interval from the foot to the time at which a1 occurred
        ta2 = f2 - q                 #time interval from the foot to the time at which a2 occurred
        tb1 = f1 - r                 #time interval from the foot to the time at which b1 occurred
        tb2 = f2 - s                 #time interval from the foot to the time at which b2 occurred
        tpp = x[peak3[1]]-x[peak3[0]]      #peak to peak time interval

        #Ratios
        b2_a2 = b2/a2
        b1_a1 = b1/a1
        ta1_tpp = ta1/tpp
        ta2_tpp = ta2/tpp
        tb1_tpp = tb1/tpp
        tb2_tpp = tb2/tpp
        ta12_tpp = (ta1-ta2)/tpp
        tb12_tpp = (tb1-tb2)/tpp

        #Appending values
        derivative_features["First_derivative"]["Peaks"].append(a1)
        derivative_features["First_derivative"]["Valleys"].append(b1)
        derivative_features["Second_derivative"]["Peaks"].append(a2)
        derivative_features["Second_derivative"]["Valleys"].append(b2)
        derivative_features["Time_intervals"]["Peak_to_peak"].append(tpp)
        derivative_features["Time_intervals"]["First_derivative_peak"].append(ta1)
        derivative_features["Time_intervals"]["Second_derivative_peak"].append(ta2)
        derivative_features["Time_intervals"]["First_derivative_valley"].append(tb1)
        derivative_features["Time_intervals"]["Second_derivative_valley"].append(tb2)
        derivative_features["Ratios"]["b1_a1"].append(b1_a1)
        derivative_features["Ratios"]["b2_a2"].append(b2_a2)
        derivative_features["Ratios"]["ta1_tpp"].append(ta1_tpp)
        derivative_features["Ratios"]["ta2_tpp"].append(ta2_tpp)
        derivative_features["Ratios"]["tb1_tpp"].append(tb1_tpp)
        derivative_features["Ratios"]["tb2_tpp"].append(tb2_tpp)
        derivative_features["Ratios"]["ta12_tpp"].append(ta12_tpp)
        derivative_features["Ratios"]["tb12_tpp"].append(tb12_tpp)

return derivative_features

file_loc = glob.glob(os.getcwd() + "\\Perfect PPG Data" + "\\*.txt")
derivative_ppg_features = derivative_features_extractor(file_loc)
    
    # =========================================================================
    
    # =========================================================================
    # Creating the dataframe
    
    temp_list = [A0_2, A2_5, Ratio_A, fmax, mag_fmax, mean, 
                 median, st_dev, perc_25, perc_50, perc_75, mad, IQR, QD,
                 skewness, kurtosis, shannon_entropy, spec_entropy]
    
    data_list.append(temp_list)
    

