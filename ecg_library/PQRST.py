# ECG aglorithms to detect components of a normal heart beat
# P wave, Q R & S, T wave

import numpy as np
from scipy.signal import find_peaks

# Function to detect QRS complexes

def detect_qrs(filtered_ecg, fs):
    positive_peaks, pos_properties = find_peaks(filtered_ecg, height=100, prominence=50, distance=10) # Added distance
    positive_heights = pos_properties['peak_heights']
    negative_peaks, neg_properties = find_peaks(-filtered_ecg, height=100, prominence=50, distance=10) # Added distance
    negative_heights = -neg_properties['peak_heights']
    max_rs_distance = int(0.048 * fs) # Modified from 0.04
    r_peaks_final = []
    s_peaks_final = []
    rs_pairs = []
    for r_idx, r_pos in enumerate(positive_peaks):
        r_amp = positive_heights[r_idx]
        if r_amp <= 50: # Different from Troubleshooting
            continue
        s_candidates = negative_peaks[(negative_peaks > r_pos) & (negative_peaks <= r_pos + max_rs_distance)]
        if len(s_candidates) > 0:
            s_amps = []
            for s_pos in s_candidates:
                s_idx = np.where(negative_peaks == s_pos)[0][0]
                s_amps.append(negative_heights[s_idx])
            deepest_s_idx = np.argmin(s_amps)
            s_pos = s_candidates[deepest_s_idx]
            s_amp = s_amps[deepest_s_idx]
            if s_amp < -100 and (r_amp - s_amp) > 300: # Different from Troubleshooting
                r_peaks_final.append(r_pos)
                s_peaks_final.append(s_pos)
                rs_pairs.append((r_pos, s_pos, r_amp, s_amp))
    r_peaks = np.array(r_peaks_final)
    s_peaks = np.array(s_peaks_final)
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan
    return r_peaks, s_peaks, rr_intervals, mean_rr

# Function to detect P, Q, T waves
def detect_pqt(filtered_ecg, fs, r_peaks, s_peaks):
    p_window = 0.24
    q_window = 0.04
    t_window = 0.4
    p_points = []
    q_points = []
    t_points = []
    for i, (r_idx, s_idx) in enumerate(zip(r_peaks, s_peaks)):
        # P wave — clamp search start to after previous T-wave to avoid T-P confusion
        p_start = int(r_idx - p_window * fs)
        p_end = r_idx
        if i > 0 and t_points[i-1] is not None:
            earliest_p = t_points[i-1] + int(0.04 * fs)  # 40ms buffer after prev T
            p_start = max(p_start, earliest_p)
        p_point = None
        if p_start >= 0 and p_end - p_start >= int(0.04 * fs):
            p_segment = filtered_ecg[p_start:p_end]
            p_peak_rel, p_props = find_peaks(p_segment, height=10, prominence=10)
            if len(p_peak_rel) > 0:
                max_idx = np.argmax(p_props['peak_heights'])
                p_point = p_start + p_peak_rel[max_idx]
        p_points.append(p_point)
        # Q wave
        q_start = int(r_idx - q_window * fs)
        q_end = r_idx
        q_point = None
        if q_start >= 0:
            q_segment = filtered_ecg[q_start:q_end]
            q_point = q_start + np.argmin(q_segment)
            if filtered_ecg[q_point] < 100:
                q_points.append(q_point)
            else:
                q_points.append(None)
        else:
            q_points.append(None)
        # if q_start >= 0:
        #     q_segment = filtered_ecg[q_start:q_end]
        #     q_point = q_start + np.argmin(q_segment)
        # q_points.append(q_point)
        # T wave
        t_start = s_idx + 10 # Moving start to the right to make sure we don't find T to close to S
        t_end = int(t_start + t_window * fs)
        t_point = None
        if t_end <= len(filtered_ecg) and q_point != None: # Needed to add None if there is no q_point
            t_segment = filtered_ecg[t_start:t_end]
            t_peak_rel, t_props = find_peaks(t_segment, height=20, prominence=20)
            if len(t_peak_rel) > 0:
                # t_point = t_start + t_peak_rel[0]
                # Changing approach and using max peak height in window instead of first peak
                max_idx = np.argmax(t_props['peak_heights'])
                t_point = t_start + t_peak_rel[max_idx]
        t_points.append(t_point)
    return p_points, q_points, t_points
