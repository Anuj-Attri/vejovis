# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:53:03 2021

@author: Tejaswini, Dev & Anuj
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt
import sys
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Defining the baseline extraction function
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

# Defining the min-max normalization function 
def normalize_minmax(arr, range_min = 0, range_max = 1):
    temp_arr = []
    diff = range_max - range_min;
    arr_diff = max(arr) - min(arr)
    
    for i in arr:
        temp_arr.append((i-min(arr))*diff/arr_diff + range_min)
    
    temp_arr = np.array(temp_arr)
    return temp_arr

# Defining the plot function
def plot(y):
    # # Getting the current time to save the figure plot as a unique figure
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    
    # Display the data and save the graph in the Graphs folder
    x = np.linspace(0, 2.1, 2100)   
    plt.figure(dpi = 1200)    # High resolution plot for better visibility
    plt.plot(x, y)
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Voltage Amplitude")
    plt.title("PPG Signal")
    plt.savefig(os.getcwd() + "\Graphs\Figure_" + current_time + ".png")
    plt.close()
    
# Defining the Baseline Removal function
def Remove_Baseline(y):
    baseline = baseline_als(y, lam = 100000000, p = 0.0047)
    y_corrected = np.array(y - baseline)
    y_corrected = normalize_minmax(y_corrected)
    
    return y_corrected
    


# EXECUTED THIS JUST ONCE TO FIND THE OPTIMAL SIGNALS(OptimalData.txt)
#==============================================================================
# Creating a new text file to store the most optimal signals

file = open("OptimalData.txt", "w")

# Using the Skewness SQI(SSQI) to find the most optimal signal segment
dataset = pd.read_excel(os.getcwd() + "\Table1.xlsx", engine = 'openpyxl')
df = dataset.iloc[:, 1:5]
subject_id = list(df.iloc[:, 0])

for i in range(0, len(subject_id)):
    row = list(df.iloc[i, 1:4])
    indexmax = str(row.index(max(row)) + 1)
    filename = str(subject_id[i]) + "_" + indexmax + ".txt"
    file.write(filename+"\n")
    
file.close()


#==============================================================================    

# EXECUTED THIS JUST ONCE TO CLEAN THE SIGNALS AND SAVE THEM (Filtered Data)
# =============================================================================
# Reading the OptimalData.txt file to find the best signals
datafile = open("OptimalData.txt")
signals = datafile.read()
datafile.close()
signals = list(signals.split("\n"))
signals.remove(signals[len(signals)-1])

# Initializing the signal details for signal processing
sr = 1000   # Sampling rate
t = 2.1     # Time duration

# Setting up the Chebyshev filter
sos = signal.cheby2(10, 60, 5, btype = "lowpass", fs = 1000, output = 'sos')

# Open the data files one by one from the OptimalData file
for i in range(0, 1):
    loc = os.getcwd() + "\Data File\\0_subject\\" + signals[i]
    f = open(loc, "r")
    filtered_signals = open(os.getcwd() + "\\Filtered Data\\" + signals[i], 'w')
    y = list(f.read().split("\t"))
    y.remove(y[len(y)-1])
    
    for i in range(0, len(y)):
        y[i] = float(y[i])
    
    # ==================== FILTERING SECTION STARTS HERE ======================
    
    # Normalize the signal for easier preprocessing
    y = np.array(y)
    minmax_normal_y = normalize_minmax(y, 0, 1)
    
    # Using a Chebyshev-II filter for obtaining a filtered curve
    filtered = signal.sosfilt(sos, minmax_normal_y)
    filtered = normalize_minmax(filtered, 0, 1)
    
    # Converting the filtered signal to a string to store it in a file
    filtered = str(list(filtered))
    filtered = filtered[1 : -1]
    filtered_signals.write(filtered)
    
    # Close all files
    f.close()
    filtered_signals.close()    
    
    # =================== FILTERING SECTION ENDS HERE =========================
    
# =============================================================================
"""

"""
# BASELINE REMOVAL FROM THE SIGNALS AND SAVING THEM

# =============================================================================

datafile = open("OptimalData.txt")
signals = datafile.read()
datafile.close()
signals = list(signals.split("\n"))
signals.remove(signals[len(signals)-1])

for i in range(0, 219):
    loc = os.getcwd() + "\\Filtered Data\\" + signals[i]
    perfect_signals = open(os.getcwd() + "\\Perfect PPG Data\\" + signals[i], 'w')
    f = open(loc, "r")
    y = list(f.read().split(","))
    
    for i in range(0, len(y)):
        y[i] = float(y[i])
    
    # Baseline removal
    y = Remove_Baseline(y)
    
    perfect_ppg = str(list(y))
    perfect_ppg = perfect_ppg[1 : -1]
    perfect_signals.write(perfect_ppg)

    perfect_signals.close()   
    f.close()

# =============================================================================

        
    




