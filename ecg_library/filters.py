# Filters
# # Import necessary libraries

from scipy.signal import butter, filtfilt, medfilt

# Define the bandpass filter function (assuming it's not provided in the pasted code)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to filter the ECG signal
def filter_signal(combined_ecg_data, fs):
    lowcut = 0.5
    highcut = 40.0
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_ecg = filtfilt(b, a, combined_ecg_data)
    kernel_size = int(fs * 0.5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    baseline = medfilt(filtered_ecg, kernel_size=kernel_size)
    filtered_ecg -= baseline
    return filtered_ecg

