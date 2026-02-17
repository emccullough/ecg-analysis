# utils.py

import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Function to parse folder name for activity and start time
def parse_folder(subfolder):
    activity_match = re.search(r'^(.*?)_on_', subfolder)
    activity = activity_match.group(1) if activity_match else 'Unknown'
    timestamp_match = re.search(r'_on_(.*?)_by_', subfolder)
    timestamp = timestamp_match.group(1) if timestamp_match else 'Unknown'
    try:
        start_time = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
    except:
        start_time = None
    return activity, start_time

# Function to load and concatenate ECG data
def load_ecg_data(folder_path):
    file_pattern = os.path.join(folder_path, 'ecg_*.csv')
    file_paths = sorted(glob.glob(file_pattern), key=lambda f: int(re.search(r'ecg_(\d+)\.csv', f).group(1)))
    file_paths = [fp for fp in file_paths if 'ecg_0.csv' not in fp]
    combined_ecg_data = np.array([])
    for fp in file_paths:
        data = pd.read_csv(fp, header=None, comment='#')[0].values
        combined_ecg_data = np.concatenate((combined_ecg_data, data))
    fs = 125
    combined_time = np.arange(len(combined_ecg_data)) / fs
    return combined_ecg_data, combined_time, fs, file_paths

# Function to save rich processed beats to CSV
def save_rich_data(folder_path, subfolder, start_time, combined_time, filtered_ecg, all_beats, beat_types, all_rr_intervals, file_paths, fs):
    if not start_time:
        return
    all_p_points = np.array([b['p_position'] for b in all_beats])
    all_q_points = np.array([b['q_position'] for b in all_beats])
    all_s_points = np.array([b['s_position'] for b in all_beats])
    all_t_points = np.array([b['t_position'] for b in all_beats])
    all_r_peaks = np.array([b['r_position'] for b in all_beats])
    beat_data = []
    samples_per_file = 2500
    for i in range(len(all_r_peaks)):
        abs_r_time = start_time + timedelta(seconds=combined_time[all_r_peaks[i]])
        abs_p_time = start_time + timedelta(seconds=combined_time[all_p_points[i]]) if all_p_points[i] is not None else None
        abs_q_time = start_time + timedelta(seconds=combined_time[all_q_points[i]]) if all_q_points[i] is not None else None
        abs_s_time = start_time + timedelta(seconds=combined_time[all_s_points[i]]) if all_s_points[i] is not None else None
        abs_t_time = start_time + timedelta(seconds=combined_time[all_t_points[i]]) if all_t_points[i] is not None else None
        rr = all_rr_intervals[i-1] if i > 0 else None
        pr = (all_r_peaks[i] - all_p_points[i]) / fs if all_p_points[i] is not None else None
        qrs = (all_s_points[i] - all_q_points[i]) / fs if all_q_points[i] is not None and all_s_points[i] is not None else None
        qt = (all_t_points[i] - all_q_points[i]) / fs if all_q_points[i] is not None and all_t_points[i] is not None else None
        r_amp = int(filtered_ecg[all_r_peaks[i]])
        t_amp = int(filtered_ecg[all_t_points[i]]) if all_t_points[i] is not None else None
        file_index = int(all_r_peaks[i] // samples_per_file)
        orig_file = file_paths[file_index] if file_index < len(file_paths) else 'Unknown'
        p_index = all_p_points[i]
        q_index = all_q_points[i]
        r_index = all_r_peaks[i]
        s_index = all_s_points[i]
        t_index = all_t_points[i]
        beat_data.append({
            'path': folder_path,
            'filename': orig_file.split('/')[-1],
            'beat_type': beat_types[i],
            'abs_p_time': abs_p_time,
            'abs_q_time': abs_q_time,
            'abs_r_time': abs_r_time,
            'abs_s_time': abs_s_time,
            'abs_t_time': abs_t_time,
            'rr_interval': rr,
            'pr_interval': pr,
            'qrs_duration': qrs,
            'qt_interval': qt,
            'r_amplitude': r_amp,
            't_amplitude': t_amp,
            'p_index': p_index,
            'q_index': q_index,
            'r_index': r_index,
            's_index': s_index,
            't_index': t_index
        })
    df = pd.DataFrame(beat_data)
    csv_filename = os.path.join(folder_path, f"{subfolder}_rich_processed_beats.csv")
    df.to_csv(csv_filename, index=False)