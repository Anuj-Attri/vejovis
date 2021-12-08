import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.integrate as integrate
from scipy.signal import find_peaks
from scipy import stats
from antropy import spectral_entropy
import glob
from scipy.signal import savgol_filter
import pickle

# Initializing the signal details for signal processing
sr = 1000   # Sampling rate
t = 2.1     # Time duration

# Setting up the Chebyshev filter
sos = signal.cheby2(10, 60, 5, btype = "lowpass", fs = 1000, output = 'sos')

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

# Defining the Baseline Removal function
def Remove_Baseline(y):
    baseline = baseline_als(y, lam = 100000000, p = 0.0047)
    y_corrected = np.array(y - baseline)
    y_corrected = normalize_minmax(y_corrected)
    
    return y_corrected
    

def local_min(ys):
    '''
    ys: Input signal to be traversed
    Returns the local minimum of a given numeric array
    '''
    return [y for i, y in enumerate(ys)
            if ((i == 0) or (ys[i - 1] >= y))
            and ((i == len(ys) - 1) or (y < ys[i+1]))]

def feature_extractor(y):
    '''
    
    dataframe = feature_extractor(loc)
    
    loc: List of data directories
    Returns a DataFrame of extracted features from all files present in loc.
    '''
    features = {
        "Systolic": {
            "SysPeakInd": [],
            "SysPeakVal": [],
            "SysCrest": [],
            "SysImp": [],
            "SysClear": []},
        
        "Diastolic": {
            "DiaPeakInd": [],
            "DiaPeakVal": [],
            "DiaCrest": [],
            "DiaImp": [],
            "DiaClear": []},
        
        "Overall": {
            "coeff_of_var": [],
            "shape_factor": [],
            "P2P": [],
            "local_minima": [],
            "RMS": []}
        }

    data_list = []  # For the final data storage
    temp_list = []  # For temporary data storage

   
        
        # Finding the FFT of the signal to find F-Domain features
    fft_vals = np.fft.fft(y)
    mags = abs(fft_vals)/2100
    mags = mags[0 : int(2100/2 + 1)]
    mags[0] = mags[0]/2
    freq_plot = np.fft.fftfreq(2100, d = 0.001)
    freq_plot = freq_plot[0 : int(2100/2 + 1)]
    
    
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
    
    temp_list = [A0_2, A2_5, Ratio_A, fmax, mag_fmax, mean, 
         median, st_dev, perc_25, perc_50, perc_75, mad, IQR, QD,
         skewness, kurtosis, shannon_entropy, spec_entropy]

    data_list.append(temp_list)

    
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
    
    features["Systolic"]["SysPeakInd"].append(syspeak)
    features["Systolic"]["SysPeakVal"].append(syspeak_amp)
    
    # Approximate diastolic peak indices and amplitude
    [diapeak, prop_dia] = find_peaks(y,
                                     # Specified a boundary condition for diastolic peak
                                     height=(0, np.mean(y)),    
                                     distance=1)
    diapeak_amp = prop_dia['peak_heights']         
    
    features["Diastolic"]["DiaPeakInd"].append(diapeak)
    features["Diastolic"]["DiaPeakVal"].append(diapeak)
    
    # Crest Factor: deviation of peaks from the RMS value
    features["Systolic"]["SysCrest"].append(syspeak_amp/rms)
    features["Diastolic"]["DiaCrest"].append(diapeak_amp/rms)
    
    # Clearance factor: Peak to Square of mean error for oscillatory systems
    features["Systolic"]["SysClear"].append(syspeak_amp/np.square(np.mean(np.sqrt(y))))
    features["Diastolic"]["DiaClear"].append(diapeak_amp/np.square(np.mean(np.sqrt(y))))
    
    features["Overall"]["shape_factor"].append(rms/np.mean(y))
    features["Overall"]["P2P"].append(max(y) - min(y))
    features["Overall"]["local_minima"].append(local_min(y))
    
    # Impulse factor: Instantaneous amplitude ratio of Peak to signal's mean
    features["Systolic"]["SysImp"].append(syspeak_amp/np.mean(y))
    features["Diastolic"]["DiaImp"].append(diapeak_amp/np.mean(y))
        

        
    stat_columns = ['A0_2', 'A2_5', 'Ratio_A', 'fmax', 'mag_fmax', 'mean', 
             'median', 'st_dev', 'perc_25', 'perc_50', 'perc_75', 'mad', 'IQR',
             'QD', 'skewness', 'kurtosis', 'shannon_entropy', 'spec_entropy']
    
    stat_df = pd.DataFrame(data_list, columns=stat_columns)
    
    time_df = pd.concat([pd.DataFrame(features["Overall"]),
                         pd.DataFrame(features["Systolic"]),
                         pd.DataFrame(features["Diastolic"])], axis = 1)
    
    STfeatures_df = pd.concat([stat_df, time_df], axis=1)
    
    return STfeatures_df

def derivative_features_extractor(y):
    
    derivative_features = {
        "First_derivative": {
            "DyPeaks":[],
            "DyValleys": []},
        "Second_derivative": {
            "DDyPeaks":[],
            "DDyValleys": []},
        "Time_intervals": {
            "P2PTime":[],
            "DyPeakTime": [],
            "DDyPeakTime": [],
            "DyValleyTime": [],
            "DDyValleyTime": []},
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

    m=len(y)
    x = np.linspace(0, 2.1, m)

    dy = np.zeros(y.shape,np.float64)
    dy[0:-1] = np.diff(y)/np.diff(x)
    dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    dy = savgol_filter(dy, 351, 3)


    ddy = np.zeros(dy.shape,np.float64)
    ddy[0:-1] = np.diff(dy)/np.diff(x)
    ddy[-1] = (dy[-1] - dy[-2])/(x[-1] - x[-2])
    ddy = savgol_filter(ddy, 355, 3)

    
    #Finding features
    peak1, _ = find_peaks(dy, distance = 800)      #array of peaks of first derivative
    peak2, _ = find_peaks(ddy, distance = 800)     #array of peaks of second derivative
    peak3, _ = find_peaks(y, distance = 800)       #array of peaks of ppg
    valley1, _ = find_peaks(-1*dy, distance = 800)   #array of valleys of first derivative
    valley2, _ = find_peaks(-1*ddy, distance = 800)  #array of valleys of second derivative
    valley3, _ = find_peaks(-1*y, distance = 800)    ##array of valleys of ppg


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

    ta1 = f1 - p                        #time interval from the foot to the time at which a1 occurred
    ta2 = f2 - q                        #time interval from the foot to the time at which a2 occurred
    tb1 = f1 - r                        #time interval from the foot to the time at which b1 occurred
    tb2 = f2 - s                        #time interval from the foot to the time at which b2 occurred
    tpp = x[peak3[1]]-x[peak3[0]]       #peak to peak time interval

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
    derivative_features["First_derivative"]["DyPeaks"].append(a1)
    derivative_features["First_derivative"]["DyValleys"].append(b1)
    derivative_features["Second_derivative"]["DDyPeaks"].append(a2)
    derivative_features["Second_derivative"]["DDyValleys"].append(b2)
    derivative_features["Time_intervals"]["P2PTime"].append(tpp)
    derivative_features["Time_intervals"]["DyPeakTime"].append(ta1)
    derivative_features["Time_intervals"]["DDyPeakTime"].append(ta2)
    derivative_features["Time_intervals"]["DyValleyTime"].append(tb1)
    derivative_features["Time_intervals"]["DDyValleyTime"].append(tb2)
    derivative_features["Ratios"]["b1_a1"].append(b1_a1)
    derivative_features["Ratios"]["b2_a2"].append(b2_a2)
    derivative_features["Ratios"]["ta1_tpp"].append(ta1_tpp)
    derivative_features["Ratios"]["ta2_tpp"].append(ta2_tpp)
    derivative_features["Ratios"]["tb1_tpp"].append(tb1_tpp)
    derivative_features["Ratios"]["tb2_tpp"].append(tb2_tpp)
    derivative_features["Ratios"]["ta12_tpp"].append(ta12_tpp)
    derivative_features["Ratios"]["tb12_tpp"].append(tb12_tpp)
    
    deriv_df = pd.concat([pd.DataFrame(derivative_features["First_derivative"]),
                  pd.DataFrame(derivative_features["Second_derivative"]),
                  pd.DataFrame(derivative_features["Time_intervals"]),
                  pd.DataFrame(derivative_features["Ratios"])], axis=1)
        
    return deriv_df

# Indexing one signal sample at a time
file = open(os.getcwd() + "\\2_1.txt")
temp_reader = list(file.read().split('\t'))
temp_reader.remove(temp_reader[len(temp_reader)-1])
file.close()

# Try/except for empty file error handling

y = []
for i in range(len(temp_reader)):
    y.append(float(temp_reader[i]))


# Initializing the signal details for signal processing
sr = 1000   # Sampling rate
t = 2.1     # Time duration

# Setting up the Chebyshev filter
sos = signal.cheby2(10, 60, 5, btype = "lowpass", fs = 1000, output = 'sos')

y = np.array(y)
minmax_normal_y = normalize_minmax(y, 0, 1)

# Using a Chebyshev-II filter for obtaining a filtered curve
filtered = signal.sosfilt(sos, minmax_normal_y)
filtered = normalize_minmax(filtered, 0, 1)

y = Remove_Baseline(filtered)

df1 = feature_extractor(y)
df2 = derivative_features_extractor(y)
data2 = pd.concat([df1, df2], axis=1)
data1 = pd.read_excel("PPG-BP dataset.xlsx")

data = pd.DataFrame()

data["Age"] = [data1["Age(year)"][0]]
data["Heart rate"] = [data1["Heart Rate(b/m)"][0]]
data["Weight"] = [data1["Weight(kg)"][0]]
data["Height"] = [data1["Height(cm)"][0]]
data["BMI"] = [data1["BMI(kg/m^2)"][0]]

temp = ['mean',
       'median', 'st_dev', 'perc_25', 'perc_50', 'perc_75','b1_a1', 'b2_a2', 'ta1_tpp', 'ta2_tpp', 'tb1_tpp', 'tb2_tpp',
        'ta12_tpp', 'tb12_tpp']

data["A2_5"] = data2["A2_5"]
data["weight_tpp"] = data["Weight"]/data2["P2PTime"]
data["BMI_tpp"] = data["BMI"]/data2["P2PTime"]
data["weight_dytime"] = data["Weight"]/data2["DyPeakTime"]
data["weight_ValleyTime"] = data["Weight"]/data2["DyValleyTime"]

data["RMS"] = data2.RMS
data["shape_factor"] = data2.shape_factor

for i in temp:
    data[i] = data2[i]
    
data_dia = data.drop(labels=["weight_dytime", "weight_ValleyTime", "A2_5"], axis=1)
data_dia["A0_2"] = data2.A0_2

Systolic = data1["Systolic Blood Pressure(mmHg)"]
Diastolic = data1["Diastolic Blood Pressure(mmHg)"]

systolic_path = os.getcwd() + "\\Models\\systolic_model.sav"
diastolic_path = os.getcwd() + "\\Models\\diastolic_model.sav"

systolic_model = pickle.load(open(systolic_path, 'rb'))
diastolic_model = pickle.load(open(diastolic_path, 'rb'))

print("Systolic Pressure: ", systolic_model.predict(data))
print("Diastolic Pressure: ", diastolic_model.predict(data_dia))