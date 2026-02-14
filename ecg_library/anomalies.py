# Anamolies

import numpy as np
from scipy.signal import find_peaks

# Function to analyze anomalies and classify beats
def analyze_anomalies(filtered_ecg, fs, combined_time, r_peaks, s_peaks, p_points, q_points, t_points):
    all_beats = []
    for i, r_pos in enumerate(r_peaks):
        all_beats.append({
            'r_position': r_pos,
            'type': 'normal',
            's_position': s_peaks[i],
            'p_position': p_points[i],
            'q_position': q_points[i],
            't_position': t_points[i],
            'energy': None,
            'fwhm': None
        })
    energy_threshold = 40 # Used to be 60
    pvc_threshold_amp = 400
    pvc_fwhm_threshold = 0.04
    vt_neg_threshold = -400
    vt_pos_threshold = 200
    vt_max_distance = int(0.4 * fs)
    min_peak_distance = int(0.04 * fs)
    tp_intervals = []
    tp_energies = []
    tp_anomalies = []
    anomaly_segments = []
    for i in range(len(t_points) - 1):
        if t_points[i] is None:
            continue
        interval_start = t_points[i]
        if i + 1 < len(p_points) and p_points[i + 1] is not None:
            interval_end = p_points[i + 1]
            interval_type = 'T-P'
        else:
            interval_end = r_peaks[i + 1] if i + 1 < len(r_peaks) else None
            interval_type = 'T-R'
        if interval_end is None or interval_start >= interval_end:
            continue
        segment = filtered_ecg[interval_start:interval_end]
        seg_len = len(segment)
        energy = np.sum(np.abs(segment)) / fs
        tp_intervals.append((interval_start, interval_end, interval_type))
        tp_energies.append(energy)
        is_anomaly = False
        anomaly_type = None
        anomaly_peaks = []
        if energy > energy_threshold:
            is_anomaly = True
            anomaly_type = 'High Energy'
            pos_peaks_rel, pos_props = find_peaks(segment, height=100, prominence=50, distance=min_peak_distance)
            pos_heights = pos_props['peak_heights']
            neg_peaks_rel, neg_props = find_peaks(-segment, height=100, prominence=50, distance=min_peak_distance)
            neg_heights = -neg_props['peak_heights']
            used_pos_peaks = set()
            # VT detection
            for n_idx, n_rel in enumerate(neg_peaks_rel):
                n_amp = neg_heights[n_idx]
                if n_amp >= vt_neg_threshold:
                    continue
                p_cands = pos_peaks_rel[(pos_peaks_rel > n_rel) & (pos_peaks_rel <= n_rel + vt_max_distance)]
                if len(p_cands) > 0:
                    p_rel = p_cands[0]
                    p_amp = pos_heights[np.where(pos_peaks_rel == p_rel)[0][0]]
                    if p_amp > vt_pos_threshold:
                        vt_pos = interval_start + n_rel
                        vt_start = max(0, interval_start + n_rel - int(0.05 * fs))
                        vt_end = min(len(filtered_ecg), interval_start + p_rel + int(0.05 * fs))
                        vt_energy = np.sum(np.abs(filtered_ecg[vt_start:vt_end])) / fs
                        if vt_pos not in [b['r_position'] for b in all_beats]:
                            all_beats.append({
                                'r_position': vt_pos,
                                'type': 'VT',
                                's_position': interval_start + p_rel,
                                'p_position': None,
                                'q_position': None,
                                't_position': None,
                                'energy': vt_energy,
                                'fwhm': None
                            })
                            anomaly_peaks.append({
                                'type': 'VT',
                                'time': combined_time[vt_pos],
                                'amplitude': n_amp,
                                'energy': vt_energy
                            })
                            used_pos_peaks.add(p_rel)
            # PVC detection
            for p_idx, p_rel in enumerate(pos_peaks_rel):
                if p_rel in used_pos_peaks:
                    continue
                p_amp = pos_heights[p_idx]
                if p_amp <= pvc_threshold_amp:
                    continue
                half = p_amp / 2
                left = p_rel - 1
                while left >= 0 and segment[left] >= half:
                    left -= 1
                if left >= 0:
                    left += (half - segment[left]) / (segment[left + 1] - segment[left])
                right = p_rel + 1
                while right < seg_len and segment[right] >= half:
                    right += 1
                if right < seg_len:
                    right -= (segment[right - 1] - half) / (segment[right - 1] - segment[right])
                fwhm = (right - left) / fs
                if fwhm > pvc_fwhm_threshold:
                    pvc_pos = interval_start + p_rel
                    left_idx = int(interval_start + left)
                    right_idx = int(interval_start + right)
                    pvc_energy = np.sum(np.abs(filtered_ecg[left_idx:right_idx])) / fs
                    if pvc_pos not in [b['r_position'] for b in all_beats]:
                        all_beats.append({
                            'r_position': pvc_pos,
                            'type': 'PVC',
                            's_position': None,
                            'p_position': None,
                            'q_position': None,
                            't_position': None,
                            'energy': pvc_energy,
                            'fwhm': fwhm
                        })
                        anomaly_peaks.append({
                            'type': 'PVC',
                            'time': combined_time[pvc_pos],
                            'amplitude': p_amp,
                            'fwhm': fwhm,
                            'energy': pvc_energy
                        })
        if is_anomaly:
            tp_anomalies.append(i)
            anomaly_segments.append({
                'start': interval_start,
                'end': interval_end,
                'type': anomaly_type,
                'energy': energy,
                'interval_type': interval_type,
                'peaks': anomaly_peaks
            })
    # Secondary PVC check: normal beats with missing P-wave + wide QRS or premature timing
    rr_from_r_peaks = np.diff(r_peaks) / fs
    mean_rr_val = np.mean(rr_from_r_peaks) if len(rr_from_r_peaks) > 0 else 1.0
    for beat in all_beats:
        if beat['type'] != 'normal' or beat['p_position'] is not None:
            continue
        q = beat['q_position']
        s = beat['s_position']
        wide_qrs = (q is not None and s is not None and (s - q) / fs > 0.10)
        premature = False
        r_pos = beat['r_position']
        prev_r = [b['r_position'] for b in all_beats if b['r_position'] < r_pos]
        if prev_r:
            prev_rr = (r_pos - max(prev_r)) / fs
            premature = prev_rr < 0.8 * mean_rr_val
        if wide_qrs or premature:
            beat['type'] = 'PVC'

    all_beats.sort(key=lambda x: x['r_position'])
    all_r_peaks = np.array([b['r_position'] for b in all_beats])
    beat_types = [b['type'] for b in all_beats]
    all_rr_intervals = np.diff(all_r_peaks) / fs
    all_mean_rr = np.mean(all_rr_intervals) if len(all_rr_intervals) > 0 else np.nan
    return all_beats, all_r_peaks, beat_types, all_rr_intervals, all_mean_rr, anomaly_segments, tp_anomalies