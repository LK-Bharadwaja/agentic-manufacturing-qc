#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pywt
import re
import statsmodels.api as sm
import scipy.signal as sp_signal  
from itertools import combinations
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import get_window
from scipy.fftpack import fft
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from tabulate import tabulate  # Optional for debugging
import xlsxwriter
import os


# Set the sampling rate
sampling_rate = 50000  # Hz
min_frequency_of_interest = 50  # Hz

# Samples per window
window_size = int(sampling_rate / min_frequency_of_interest)  
overlap = window_size // 2  # 50% overlap
freq_resolution = 50
stft_window_size = window_size // 2
channels = ['Channel1 [g]', 'Channel2 [g]', 'Channel3 [g]']

# Y_True values for all 27 parts
y_true = [2.024, 1.983, 1.945, 2.384, 2.439, 2.467, 2.752, 2.812, 2.795, 2.185,
          2.024, 1.931, 2.405, 2.408, 2.481, 2.781, 2.748, 2.745, 2.085, 2.002,
          2.023, 2.441, 2.478, 2.440, 2.776, 2.693, 2.757]

# Define file paths for 27 parts

file_paths = [f"part {i}.xlsx" for i in range(1, 28)]
print(file_paths)

# Load the data
data_frames = []  # Creating List to store DataFrames

for file_path in file_paths:
    try:
        df = pd.read_excel(file_path)  # Read CSV file
        print(f"Loaded file {file_path}")
        data_frames.append(df)
#print(f"Loaded: {file_path} | Shape: {df.shape}")  # Debugging statement
    except Exception as e:
        print(f"Error loading {file_path}: {e}")  # Error Handling


# In[2]:


def create_windows(signal, window_size, overlap):
    step = window_size - overlap
    return [
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ]


# In[3]:


def calculate_time_domain_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms
    signal_kurtosis = kurtosis(signal)
    return rms, peak, crest_factor, signal_kurtosis

def calculate_frequency_features(signal_data):
    windows = create_windows(signal_data, window_size, overlap)
    psd_vals, peak_freq_vals, bandwidth_vals, spectral_entropy_vals = [], [], [], []
    
    for window in windows:
        fft_transformed = np.abs(fft(window * get_window('hann', window_size)))
        freqs = np.fft.fftfreq(len(window), d=1/sampling_rate)
        psd = fft_transformed ** 2
        total_psd = np.sum(psd)
        peak_freq = freqs[np.argmax(fft_transformed)]
        threshold = 0.1 * np.max(psd)
        significant_freqs = freqs[psd >= threshold]
        bandwidth = significant_freqs[-1] - significant_freqs[0] if len(significant_freqs) > 0 else 0
        psd_norm = psd / np.sum(psd) if np.sum(psd) != 0 else psd
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        psd_vals.append(total_psd)
        peak_freq_vals.append(peak_freq)
        bandwidth_vals.append(bandwidth)
        spectral_entropy_vals.append(spectral_entropy)
    
    return np.mean(psd_vals), np.mean(peak_freq_vals), np.mean(bandwidth_vals), np.mean(spectral_entropy_vals)


def calculate_stft_features(signal_data, fs, win_size):
    f, t, Zxx = sp_signal.stft(signal_data, fs=fs, nperseg=win_size)
    magnitude = np.abs(Zxx) + 1e-10
    centroid = np.sum(f[:, None] * magnitude, axis=0) / np.sum(magnitude, axis=0)
    bandwidth = np.sqrt(np.sum((f[:, None] - centroid)**2 * magnitude, axis=0) / np.sum(magnitude, axis=0))
    entropy_val = entropy(magnitude, axis=0)
    kurt_val = kurtosis(magnitude, axis=0, fisher=False)
    return (
        np.nanmean(centroid),
        np.nanmean(bandwidth),
        np.nanmean(entropy_val),
        np.nanmean(kurt_val)
    )


# In[4]:


all_results = []

column_headers = ["PART NO.", "PART NO. NUM"]  # Start with Part identifiers

# Loop through 3 channels to define feature names
for ch in range(1, 4):  
    column_headers.extend([
        f"Channel {ch} RMS", f"Channel {ch} PEAK", f"Channel {ch} CREST FACTOR", f"Channel {ch} KURTOSIS",
        f"Channel {ch} PSD", f"Channel {ch} Dominant Frequency", f"Channel {ch} Total Energy", f"Channel {ch} Spectral Entropy",
        f"Channel {ch} Spectral Centroid", f"Channel {ch} Spectral Bandwidth",
        f"Channel {ch} Spectral Entropy (TF)", f"Channel {ch} Spectral Kurtosis"
    ])

# Add Y_true column at the end
column_headers.append("Y_true")


# In[5]:


for part_idx, df in enumerate(data_frames):
    part_no = f"Part {part_idx + 1}"
    part_no_num = part_idx + 1  # Numeric identifier for sorting
    part_features = [part_no, part_no_num]  # Start with PART NO. and PART NO. NUM
#     window_size = int(sampling_rate / (2 * freq_resolution))
#     stft_window_size = window_size // 2

    for ch in range(3):  # Assuming 3 channels
        if channels[ch] not in df.columns:
            print(f"Warning: {channels[ch]} not found in {part_no}, skipping...")
            continue

        signal_data = df[channels[ch]].dropna().values  # Extract channel data

        if len(signal_data) == 0:
            print(f"Warning: {channels[ch]} in {part_no} has no valid data, skipping...")
            continue

        # Compute time-domain features
        rms, peak, crest_factor, kurt = calculate_time_domain_features(signal_data)

        # Compute frequency-domain features
        psd, dominant_freq, Total_energy, spectral_entropy = calculate_frequency_features(signal_data)

        # Compute wavelet features
        spectrai_centroid, spectral_bandwidth, spectral_entropy_feature, spectral_kutosis = calculate_stft_features(signal_data, sampling_rate, stft_window_size)

        # Store all features for the current channel
        part_features.extend([
            rms, peak, crest_factor, kurt,
            psd, dominant_freq, Total_energy, spectral_entropy,
            spectrai_centroid, spectral_bandwidth, spectral_entropy_feature, spectral_kutosis
        ])

    # Append Y_true value
    if part_idx < len(y_true):
        part_features.append(y_true[part_idx])
    else:
        print(f"Warning: No Y_true value for {part_no}, assigning NaN.")
        part_features.append(np.nan)

    # Store results
    all_results.append(part_features)
   


# In[6]:


features_df = pd.DataFrame(all_results, columns=column_headers)

features_df.sort_values(by="PART NO. NUM", inplace=True, ascending=True)
features_df.drop(columns=["PART NO. NUM"], inplace=True)
features_df.reset_index(drop=True, inplace=True)


# In[7]:


# Independent variables
X = features_df.drop(columns=["PART NO.", "Y_true"])

# Dependent variable
Y = features_df["Y_true"]  

valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]


# In[8]:


r2_values = [sm.OLS(Y, sm.add_constant(X[[col]])).fit().rsquared for col in X.columns]

r2_row = pd.DataFrame([{**{col: r2 for col, r2 in zip(X.columns, r2_values)}, 'PART NO.': "R-squared Values", 'Y_true': None}])


# In[9]:


features_df = pd.concat([features_df, r2_row], ignore_index=True)


# In[10]:


feature_groups = ["RMS", "PEAK", "CREST FACTOR", "KURTOSIS", 
                  "PSD","Peak Frequency", "Spectral Entropy", "Bandwidth", 
                  "Wavelet Mean Energy", "Wavelet Kurtosis", "Wavelet Skewness", "Wavelet Entropy"]

selected_features = []


# In[11]:


# Independent variables
X = features_df.drop(columns=["PART NO.", "Y_true"])

# Dependent variable
Y = features_df["Y_true"]  

valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]


# In[12]:


for feature in feature_groups:
    relevant_cols = [col for col in X.columns if feature in col]
    if relevant_cols:
        # Select the feature with the highest R² value in this group
        top_feature = max(relevant_cols, key=lambda col: sm.OLS(Y, sm.add_constant(X[[col]])).fit().rsquared)
        
        # Ensure uniqueness
        if top_feature not in selected_features:
            selected_features.append(top_feature)

# Define output file path
output_file_path = "Vibration_Analysis_Combined_27_Parts.xlsx"


# In[13]:


selected_features_df = features_df[["PART NO."] + selected_features + ["Y_true"]]


# In[14]:


with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
    features_df.to_excel(writer, sheet_name="Combined_Results", index=False)
    selected_features_df.to_excel(writer, sheet_name="Top_Features", index=False)
print(f"Results saved in {output_file_path} with all feature combinations listed in an additional sheet.")


# In[15]:


all_combinations = list(combinations(selected_features, 4))


# In[16]:


file_path = "Vibration_Analysis_Combined_27_Parts.xlsx"
xls = pd.ExcelFile(file_path)


# In[17]:


top_features_df = xls.parse('Top_Features')
top_12_features = selected_features


# In[18]:


top_features_df = top_features_df.dropna(subset=['Y_true'])
X_full = top_features_df[top_12_features]
Y_true = top_features_df['Y_true']


# In[19]:


best_combination = None
best_r2_score = float('-inf')
best_accuracy = float(0)
results = []
ypred_dict = {}


# In[20]:


def custom_regression_accuracy(Y_true, Y_pred):
#     Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    
#     # Avoid division by zero by applying a mask
#     mask = Y_true != 0
#     accuracy_values = np.zeros(len(Y_true))  # Initialize with zeros
    
#     # Apply formula where Y_true is not zero
#     accuracy_values[mask] = 1 - (np.abs(Y_true[mask] - Y_pred[mask]) / np.abs(Y_true[mask]))
    accuracy_values = 1 - np.abs(Y_true - Y_pred) / np.abs(Y_pred)

    return np.mean(accuracy_values)


# In[21]:


# Iterate over all combinations of 4 features
for combo in all_combinations:
    X = X_full[list(combo)]
    model = LinearRegression()
    model.fit(X, Y_true)
    Y_pred = model.predict(X)
    r2 = r2_score(Y_true, Y_pred)
    
    # Store results
    results.append({'Feature Combination': combo, 'R² Score': r2})
    ypred_dict[str(combo)] = Y_pred
    
    accuracy = (custom_regression_accuracy(Y_true, Y_pred))
#     print(combo)
#     print(accuracy)
    # Track best combination
#     if r2 > best_r2_score:
#         best_r2_score = r2
#         best_combination = combo
#         print(best_combination)
#         best_Y_pred = Y_pred
#         best_model = model
        
    if accuracy > best_accuracy:
        best_r2_score = r2
        best_accuracy = accuracy
        best_combination = combo
        best_Y_pred = Y_pred
        best_model = model
# Calculate Accuracies and Average accuracy
# accuracy1 = custom_regression_accuracy(Y_true, best_Y_pred)
# average_accuracy = np.mean(accuracy1)


# In[22]:


best_results_df = X_full[list(best_combination)].copy()
best_results_df["Y_true"] = Y_true


# In[23]:


intercept = best_model.intercept_
coefficients = best_model.coef_


# In[24]:


equation_terms = [f"{coeff:.4f} * {feature}" for coeff, feature in zip(coefficients, best_combination)]
equation = f"Y = {' + '.join(equation_terms)} + {intercept:.4f}"


# In[25]:


output_file = "Feature_Combinations_Results_27_Parts.xlsx"


# In[26]:


with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    # Save the full feature dataset
    selected_features_df.to_excel(writer, sheet_name="Top_Features", index=False)
    
    # Save all possible feature combinations
    pd.DataFrame(all_combinations, columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"]).to_excel(writer, sheet_name="Feature_Combinations", index=False)
    
    # Save the best feature combination with Y_pred
    best_results_df.to_excel(writer, sheet_name="Best_Combination", index=False)


# In[27]:


output_message = f"""
Best 4-feature combination: {best_combination}\n
Average Accuracy: {accuracy}\n
Highest R² score: {best_r2_score}\n
Results saved in {output_file} with all feature combinations listed in an additional sheet.\n
Regression Equation: {equation}
"""


# In[28]:


print(output_message)


# In[29]:


file_paths = [f"part {i}.xlsx" for i in range(28, 44)]


# Load the data
data_frames = []  # Creating List to store DataFrames

for file_path in file_paths:
    try:
        df = pd.read_excel(file_path)  # Read CSV file
        #print(f"Loaded file {file_path}")
        data_frames.append(df)
#print(f"Loaded: {file_path} | Shape: {df.shape}")  # Debugging statement
    except Exception as e:
        print(f"Error loading {file_path}: {e}")  # Error Handling




y_true = [2.748, 2.059, 2.699, 2.037, 1.997, 2.658, 2.079, 2.059,
               2.039, 2.672, 2.101, 2.075, 2.569, 2.571, 2.573, 2.682]
all_results = []

column_headers = ["PART NO.", "PART NO. NUM"]  # Start with Part identifiers

# Loop through 3 channels to define feature names
for ch in range(1, 4):  
    column_headers.extend([
        f"Channel {ch} RMS", f"Channel {ch} PEAK", f"Channel {ch} CREST FACTOR", f"Channel {ch} KURTOSIS",
        f"Channel {ch} PSD", f"Channel {ch} Peak Frequency", f"Channel {ch} Bandwidth", f"Channel {ch} Spectral Entropy",
        f"Channel {ch} Wavelet Mean Energy", f"Channel {ch} Wavelet Kurtosis",
        f"Channel {ch} Wavelet Skewness", f"Channel {ch} Wavelet Entropy"
    ])

# Add Y_true column at the end
column_headers.append("Y_true")

for part_idx, df in enumerate(data_frames):
    part_no = f"Part {part_idx + 1}"
    part_no_num = part_idx + 1  # Numeric identifier for sorting
    part_features = [part_no, part_no_num]  # Start with PART NO. and PART NO. NUM

    for ch in range(3):  # Assuming 3 channels
        if channels[ch] not in df.columns:
            print(f"Warning: {channels[ch]} not found in {part_no}, skipping...")
            continue

        signal = df[channels[ch]].dropna().values  # Extract channel data

        if len(signal) == 0:
            print(f"Warning: {channels[ch]} in {part_no} has no valid data, skipping...")
            continue

        # Compute time-domain features
        rms, peak, crest_factor, kurt = calculate_time_domain_features(signal)

        # Compute frequency-domain features
        psd, peak_freq, bandwidth, spectral_entropy = calculate_frequency_features(signal)

        # Compute wavelet features
        spectrai_centroid, spectral_bandwidth, spectral_entropy_feature, spectral_kutosis = calculate_stft_features(signal_data, sampling_rate, stft_window_size)

        # Store all features for the current channel
        part_features.extend([
            rms, peak, crest_factor, kurt,
            psd, dominant_freq, Total_energy, spectral_entropy,
            spectrai_centroid, spectral_bandwidth, spectral_entropy_feature, spectral_kutosis
        ])

    # Append Y_true value
    if part_idx < len(y_true):
        part_features.append(y_true[part_idx])
    else:
        print(f"Warning: No Y_true value for {part_no}, assigning NaN.")
        part_features.append(np.nan)

    # Store results
    all_results.append(part_features)
    
features_df = pd.DataFrame(all_results, columns=column_headers)
    
features_df.sort_values(by="PART NO. NUM", inplace=True, ascending=True)
features_df.drop(columns=["PART NO. NUM"], inplace=True)
features_df.reset_index(drop=True, inplace=True)

# Define output file path
output_file_path = "Vibration_Analysis_Combined_16_Parts.xlsx"

with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
    features_df.to_excel(writer, sheet_name="Combined_Results", index=False)

print(f"Results saved in {output_file_path} with all feature combinations listed in an additional sheet.")


# In[ ]:





# In[ ]:




