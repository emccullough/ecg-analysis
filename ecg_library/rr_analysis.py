"""
RR-interval anomaly analysis — v2 signal processing layer.

Reads the beat DataFrame produced by v1 (rich_processed_beats.csv) and adds
three columns that capture timing-based anomalies independent of morphology:

    rr_baseline    rolling median RR of the preceding `window` normal beats
    rr_flag        'premature' | 'compensatory' | 'missed_beat' | ''
    rr_burden_pct  % of premature beats in this segment (same value for all
                   beats in the same segment, used as a per-segment stat)

Baseline strategy
-----------------
Only beats classified as 'normal' by v1 contribute to the rolling median
baseline, and only when their RR interval does not exceed MAX_PAUSE_RR (1.5 s).
This keeps PVC/VT intervals and their following compensatory pauses from
inflating the expected-interval estimate — without any heuristic thresholds.

A normal beat whose RR is anomalously long (it follows a compensatory pause
that crossed a segment boundary, for example) is simply excluded from the
baseline window; its rr_baseline will be NaN and it won't be flagged.

Classification logic
--------------------
At each beat i, compute ratio = rr[i] / rr_baseline[i]:

  ratio < short_thresh  → 'premature'     beat arrived early (PVC/PAC pattern)
  ratio > long_thresh   → 'compensatory'  long pause AFTER a premature beat
                                          (classic PVC compensatory pause)
  ratio > missed_thresh → 'missed_beat'   long pause NOT after a premature beat
                                          (possible missed beat or artefact)
  otherwise             → ''              normal timing

Burden = premature beats / total beats with a valid RR in the segment × 100 %.
This is computed per segment (per ecg_N.csv file) so segments with no anomalies
get 0.0 and can be filtered out quickly.
"""

import os
import re

import numpy as np
import pandas as pd

# Imported lazily inside functions to avoid circular imports.
# Needed for template-based classification of missed-beat candidates.
_SCORE_BEAT    = None
_DEFAULT_WIN   = (20, 30)  # (before, after) — must match pvc_templates defaults


def _get_score_beat():
    global _SCORE_BEAT, _DEFAULT_WIN
    if _SCORE_BEAT is None:
        from ecg_library.pvc_templates import score_beat, DEFAULT_BEFORE, DEFAULT_AFTER
        _SCORE_BEAT  = score_beat
        _DEFAULT_WIN = (DEFAULT_BEFORE, DEFAULT_AFTER)
    return _SCORE_BEAT, _DEFAULT_WIN

def _s_dominant_post_check(
    sig, s_start, best_center, best_corr, best_label,
    score_fn, labels_list, library, morph_threshold, bef, aft,
    normal_amp, min_r_amp=30, max_rs_gap=8,
):
    """After the sliding template scan, check whether *best_center* is a T-wave.

    S-dominant beats (small positive R followed immediately by a large negative
    S-wave) have their T-waves at roughly +200–300 ms after the R.  Because the
    T-wave amplitude is often large compared to the tiny R, the sliding scan
    picks the T-wave as the best PVC match instead of the actual QRS.

    Detection: if a large negative S-trough is found in the zone
    ``[s_start, best_center - 3]``, this S-trough is the end of a QRS whose
    tiny R-peak lies 1–``max_rs_gap`` samples before it.  The candidate is
    relocated to that R-peak and re-scored; the type is set to ``'normal'``
    if the score is below *morph_threshold*.

    Returns (new_center, new_corr, new_label) — unchanged if the check
    does not find a prior S-trough.
    """
    pre_end = best_center - 3
    if pre_end <= s_start:
        return best_center, best_corr, best_label

    pre_sig = sig[s_start:pre_end]
    pre_min = float(np.min(pre_sig))

    # Threshold: "large negative" relative to normal amplitude
    neg_thresh = (max(-100.0, -abs(normal_amp) * 0.4)
                  if np.isfinite(normal_amp) else -100.0)

    if pre_min >= neg_thresh:
        return best_center, best_corr, best_label  # no large S-trough

    # Locate the S-trough
    s_local_offset = int(np.argmin(pre_sig))
    s_abs = s_start + s_local_offset

    # Guard: if the S-trough is within 12 samples of s_start it is almost
    # certainly residual S-wave / ST segment of the *previous detected* beat
    # bleeding into the scan zone — not a new S-dominant QRS.  Skip.
    if s_abs - s_start < 12:
        return best_center, best_corr, best_label

    # Search for a positive R within max_rs_gap samples before the S-trough
    r_start = max(0, s_abs - max_rs_gap)
    if r_start >= s_abs:
        return best_center, best_corr, best_label
    r_zone     = sig[r_start:s_abs]
    r_peak_pos = int(np.argmax(r_zone))
    r_peak_amp = float(r_zone[r_peak_pos])

    if r_peak_amp < min_r_amp:
        return best_center, best_corr, best_label  # no valid R found

    actual_r = r_start + r_peak_pos

    # Re-score at the actual R position
    lo2 = actual_r - bef
    hi2 = actual_r + aft + 1
    if lo2 < 0 or hi2 > len(sig):
        return best_center, best_corr, best_label

    win2       = sig[lo2:hi2].astype(float)
    best2      = -np.inf
    lbl2       = 'normal'
    for lbl in labels_list:
        c2 = score_fn(win2, library, label=lbl)
        if np.isfinite(c2) and c2 > best2:
            best2 = c2
            lbl2  = lbl

    new_label = lbl2 if best2 >= morph_threshold else 'normal'
    return actual_r, float(best2), new_label


# RR values above this are definite pauses — never a normal beat interval.
# 1.5 s ≈ 40 bpm; no healthy resting heart rate is slower than this.
MAX_PAUSE_RR = 1.5  # seconds

# Minimum samples from the previous detected R-peak before the missed-beat
# scan zone may start.  Clears the T-wave of the previous beat even at low
# heart rates (QT ≤ 400 ms at ≤ 75 bpm → 50 samples at 125 Hz).
MIN_T_SKIP = 50


def analyze_rr_anomalies(
    beats_df,
    window=7,
    short_thresh=0.80,
    long_thresh=1.35,
    missed_thresh=1.70,
):
    """Add RR-interval timing flags to a v1 beat DataFrame.

    Parameters
    ----------
    beats_df : pd.DataFrame
        Output of v1 processing — must contain columns ``rr_interval``,
        ``beat_type``, and ``filename`` (one row per beat, sorted in time
        order within each file).
    window : int
        Number of preceding *normal* beats used for the rolling median
        baseline.  Minimum 2 normal beats required before any flag is assigned.
    short_thresh : float
        Fraction of baseline below which an RR is 'premature' (default 0.80).
    long_thresh : float
        Fraction of baseline above which an RR after a premature beat is
        'compensatory' (default 1.35).
    missed_thresh : float
        Fraction of baseline above which an RR *not* after a premature beat is
        'missed_beat' (default 1.70, i.e. gap > 1.7× expected).

    Returns
    -------
    pd.DataFrame
        Copy of ``beats_df`` with three new columns added.
    """
    df = beats_df.copy()
    df['rr_interval'] = pd.to_numeric(df['rr_interval'], errors='coerce')
    df['rr_baseline'] = np.nan
    df['rr_flag'] = ''
    df['rr_burden_pct'] = np.nan

    # Each segment (ecg_N.csv) is independent — cross-segment RR gaps are
    # artefacts of how the recording is split, not real cardiac events.
    for fname, seg in df.groupby('filename', sort=False):
        idxs = seg.index.tolist()
        rr = seg['rr_interval'].values.astype(float)
        beat_types = seg['beat_type'].values

        baseline = np.full(len(rr), np.nan)
        flags = np.full(len(rr), '', dtype=object)

        # Build rolling baseline from *normal* beats only, capped at MAX_PAUSE_RR.
        # Using v1's beat_type classification means PVC/VT intervals and the
        # compensatory pause that follows them are automatically excluded —
        # no heuristic needed. The remaining RR cap handles the rare case where
        # a normal-classified beat has a cross-segment compensatory-pause-length
        # interval at the start of a file.
        normal_rr = np.where(
            (beat_types == 'normal') & np.isfinite(rr) & (rr <= MAX_PAUSE_RR),
            rr,
            np.nan,
        )

        for i in range(len(rr)):
            start = max(0, i - window)
            vals = normal_rr[start:i]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                baseline[i] = np.median(vals)

        # Classify each beat (normal *and* PVC/VT) against the baseline.
        # This lets v2 catch premature beats that v1's morphology detector missed
        # (e.g. interpolated PVCs with a near-normal QRS shape).
        prev_was_premature = False
        for i in range(len(rr)):
            if not np.isfinite(rr[i]) or not np.isfinite(baseline[i]):
                prev_was_premature = False
                continue

            ratio = rr[i] / baseline[i]

            if ratio < short_thresh:
                flags[i] = 'premature'
                prev_was_premature = True
            elif ratio > missed_thresh and not prev_was_premature:
                flags[i] = 'missed_beat'
                prev_was_premature = False
            elif ratio > long_thresh and prev_was_premature:
                flags[i] = 'compensatory'
                prev_was_premature = False
            else:
                prev_was_premature = False

        # Burden: premature beats as % of beats with a valid RR in this segment
        n_premature = int(np.sum(flags == 'premature'))
        n_valid = int(np.sum(np.isfinite(rr)))
        burden = round(n_premature / n_valid * 100, 2) if n_valid > 0 else 0.0

        for i, orig_idx in enumerate(idxs):
            df.at[orig_idx, 'rr_baseline'] = baseline[i]
            df.at[orig_idx, 'rr_flag'] = flags[i]
            df.at[orig_idx, 'rr_burden_pct'] = burden

    return df


def find_missed_beat_candidates(beats_df, session_dir, library=None,
                                morph_threshold=0.80, fs=125):
    """Search ECG gaps flagged as 'missed_beat' for a hidden QRS complex.

    For each beat whose ``rr_flag == 'missed_beat'``, the function loads the
    raw ECG for that segment, isolates the central zone of the gap (between
    the previous detected beat and the flagged beat), and looks for the
    largest-amplitude peak.

    When *library* is supplied the candidate peak is classified by Pearson
    correlation against the mean templates (``score_beat``).  The best-matching
    label is used when its correlation exceeds *morph_threshold*; otherwise the
    candidate is marked ``'weak'``.

    When *library* is ``None`` a polarity/amplitude heuristic is used as
    fallback (negative-dominant → ``'PVC'``; positive 0.35–1.5× normal →
    ``'normal'``; etc.)

    The search window is the central 60 % of the gap (20 % margin on each
    side), which avoids the T-wave and early repolarisation of the flanking
    detected beats.

    Parameters
    ----------
    beats_df : pd.DataFrame
        v2 beat DataFrame (output of ``analyze_rr_anomalies``).
    session_dir : str
        Path to the folder containing the ecg_N.csv segment files.
    library : dict or None
        Template library from ``load_template_library``.  If provided,
        correlation-based classification is used.
    morph_threshold : float
        Minimum Pearson correlation to assign a template label (default 0.80).
    fs : int
        Sampling frequency in Hz (default 125).

    Returns
    -------
    pd.DataFrame
        One row per missed_beat region, with columns:

        ``filename``        segment file
        ``missed_at_r``     r_index of the v2-flagged beat (local, 0-2499)
        ``prev_r``          r_index of the preceding beat
        ``gap_s``           gap duration in seconds
        ``baseline_s``      expected normal RR from v2 baseline
        ``candidate_r``     estimated r_index of the hidden beat
        ``candidate_amp``   filtered-signal amplitude at candidate_r
        ``normal_amp_med``  median R-amplitude of normal beats in the segment
        ``amp_ratio``       abs(candidate_amp) / normal_amp_med
        ``candidate_type``  template label (e.g. 'PVC', 'VT') or 'weak'
        ``morph_corr``      Pearson correlation used for classification (NaN if heuristic)
    """
    from ecg_library.filters import filter_signal

    df = beats_df.copy()
    df['rr_flag'] = df['rr_flag'].fillna('')

    SEG_LEN = fs * 20  # 2500 for 125 Hz / 20 s segments

    results = []

    for fname, seg in df.groupby('filename', sort=False):
        if 'missed_beat' not in seg['rr_flag'].values:
            continue

        ecg_path = os.path.join(session_dir, fname)
        if not os.path.exists(ecg_path):
            continue

        raw = pd.read_csv(ecg_path, header=None, comment='#')[0].values.astype(float)
        sig = filter_signal(raw, fs)

        # Median R-amplitude of normal beats — our amplitude reference.
        norm_amps = pd.to_numeric(
            seg.loc[seg['beat_type'] == 'normal', 'r_amplitude'], errors='coerce'
        ).dropna()
        normal_amp = float(norm_amps.median()) if len(norm_amps) else np.nan

        seg_rows = seg.reset_index(drop=True)

        for i, row in seg_rows.iterrows():
            if row['rr_flag'] != 'missed_beat':
                continue
            if i == 0:
                continue  # no preceding beat in this segment

            prev = seg_rows.iloc[i - 1]
            prev_r = int(prev['r_index']) % SEG_LEN
            curr_r = int(row['r_index']) % SEG_LEN
            baseline_s = float(row['rr_baseline'])

            if not np.isfinite(baseline_s) or curr_r <= prev_r:
                continue

            # Search zone: skip at least MIN_T_SKIP samples from prev_r to
            # clear its T-wave, but cap the left margin so the zone doesn't
            # collapse for small gaps (keeps ≥ 10 samples available).
            gap_samp    = curr_r - prev_r
            margin      = max(int(gap_samp * 0.20), 5)
            left_margin = max(margin, min(MIN_T_SKIP, gap_samp - margin - 10))
            s_start     = prev_r + left_margin
            s_end       = min(curr_r - margin, len(sig))

            if s_end <= s_start:
                continue

            # ── Slide the template window across the gap ──────────────────────
            # Centering on the dominant amplitude peak fails when the S-wave
            # is larger than the R-wave (common during high-intensity exercise):
            # the S-trough gets picked, its window matches the VT template, and
            # a normal beat is mis-classified.  Sliding across all positions and
            # picking the best shape match avoids this entirely.
            ctype      = 'weak'
            morph_corr = np.nan
            candidate_r   = (s_start + s_end) // 2   # fallback if no library
            candidate_amp = float(sig[candidate_r])

            if library is not None:
                score_fn, (bef, aft) = _get_score_beat()
                labels_list = list(library.get('mean_templates', {}).keys())
                best_center = None
                best_label  = None
                best_corr   = -np.inf

                for center in range(s_start, s_end):
                    lo = center - bef
                    hi = center + aft + 1
                    if lo < 0 or hi > len(sig):
                        continue
                    win = sig[lo:hi].astype(float)
                    for lbl in labels_list:
                        c = score_fn(win, library, label=lbl)
                        if np.isfinite(c) and c > best_corr:
                            best_corr  = c
                            best_label = lbl
                            best_center = center

                # S-dominant post-check: re-locate if best candidate is a T-wave
                if best_center is not None and best_corr >= morph_threshold:
                    best_center, best_corr, best_label = _s_dominant_post_check(
                        sig, s_start, best_center, best_corr, best_label,
                        score_fn, labels_list, library, morph_threshold,
                        bef, aft, normal_amp,
                    )

                if best_center is not None:
                    candidate_r   = best_center
                    candidate_amp = float(sig[best_center])
                    if best_label == 'normal' or best_corr >= morph_threshold:
                        ctype      = best_label
                        morph_corr = round(float(best_corr), 3)
            else:
                # Heuristic fallback (no library): dominant peak + polarity.
                window_arr = sig[s_start:s_end]
                max_pos    = float(np.max(window_arr))
                max_neg    = float(np.min(window_arr))
                if abs(max_neg) > abs(max_pos):
                    candidate_amp = max_neg
                    candidate_r   = int(np.argmin(window_arr)) + s_start
                    polarity      = 'negative'
                else:
                    candidate_amp = max_pos
                    candidate_r   = int(np.argmax(window_arr)) + s_start
                    polarity      = 'positive'

                if not np.isfinite(normal_amp) or normal_amp <= 0:
                    ctype = 'unknown'
                else:
                    ratio = abs(candidate_amp) / abs(normal_amp)
                    if ratio < 0.35:
                        ctype = 'weak'
                    elif polarity == 'negative':
                        ctype = 'PVC'
                    elif ratio > 1.5:
                        ctype = 'PVC'
                    else:
                        ctype = 'normal'

            results.append({
                'filename':       fname,
                'missed_at_r':    curr_r,
                'prev_r':         prev_r,
                'gap_s':          round(gap_samp / fs, 3),
                'baseline_s':     round(baseline_s, 3),
                'candidate_r':    candidate_r,
                'candidate_amp':  round(candidate_amp, 1),
                'normal_amp_med': round(normal_amp, 1) if np.isfinite(normal_amp) else np.nan,
                'amp_ratio':      round(abs(candidate_amp) / abs(normal_amp), 2)
                                  if np.isfinite(normal_amp) and normal_amp > 0 else np.nan,
                'candidate_type': ctype,
                'morph_corr':     morph_corr,
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


def find_cross_segment_missed_beats(beats_df, session_dir, library=None,
                                    morph_threshold=0.80, fs=125, missed_thresh=1.70):
    """Search for hidden QRS complexes in anomalously long cross-segment gaps.

    The first beat of every segment has no RR interval (NaN), so
    ``analyze_rr_anomalies`` cannot flag missed beats at segment boundaries.
    This function fills that gap by:

    1. Computing the cross-segment RR for each adjacent pair (N, N+1):
       ``cross_rr = (SEG_LEN - last_r_in_N + first_r_in_N+1) / fs``
    2. Comparing it to the rolling baseline from the end of segment N.
       If ``cross_rr > missed_thresh × baseline``, a QRS is likely hidden
       in the gap.
    3. Concatenating ``sig_N[last_r:]`` and ``sig_N1[:first_r]`` to form
       the full gap signal, then searching its central 60 % for the dominant
       peak — regardless of how the split falls between the two files.

    The "first_r ≈ 0" edge case (first beat of N+1 is at sample 0 or very
    early) is handled automatically: the head portion is empty or tiny, so
    the entire searchable gap lies in the tail of segment N.

    Parameters
    ----------
    beats_df : pd.DataFrame
        v2 beat DataFrame (output of ``analyze_rr_anomalies``).
    session_dir : str
        Path to the folder containing the ecg_N.csv segment files.
    fs : int
        Sampling frequency in Hz (default 125).
    missed_thresh : float
        Ratio above baseline that triggers a search (default 1.70).

    Returns
    -------
    pd.DataFrame
        Columns mirror ``find_missed_beat_candidates`` with two additions:

        ``boundary``       always ``'cross_segment'``
        ``prev_file``      the segment file where the gap *starts* (seg N)
        ``candidate_file`` the file the candidate beat falls in (N or N+1)

        ``filename`` is set to the N+1 file so the viewer can look it up
        when displaying that segment.
    """
    from ecg_library.filters import filter_signal

    df = beats_df.copy()
    df['rr_flag'] = df['rr_flag'].fillna('')

    SEG_LEN = fs * 20  # 2500 for 125 Hz / 20 s segments

    def _seg_num(f):
        m = re.search(r'\d+', f)
        return int(m.group()) if m else 0

    all_files = sorted(df['filename'].unique(), key=_seg_num)
    all_files = [f for f in all_files if f != 'ecg_0.csv']

    # Cache filtered signals — each file is loaded at most twice
    # (once as seg N, once as seg N+1).
    _sig_cache = {}

    def _load(fname):
        if fname not in _sig_cache:
            path = os.path.join(session_dir, fname)
            if not os.path.exists(path):
                _sig_cache[fname] = None
            else:
                raw = pd.read_csv(path, header=None, comment='#')[0].values.astype(float)
                _sig_cache[fname] = filter_signal(raw, fs)
        return _sig_cache[fname]

    results = []

    for i in range(len(all_files) - 1):
        fn_n  = all_files[i]
        fn_n1 = all_files[i + 1]
        seg_n  = df[df['filename'] == fn_n]
        seg_n1 = df[df['filename'] == fn_n1]

        if seg_n.empty or seg_n1.empty:
            continue

        last_beat  = seg_n.iloc[-1]
        first_beat = seg_n1.iloc[0]

        last_r  = int(last_beat['r_index']) % SEG_LEN
        first_r = int(first_beat['r_index']) % SEG_LEN

        cross_rr = (SEG_LEN - last_r + first_r) / fs

        # Baseline: prefer the last valid rr_baseline stored in seg N.
        baseline_s = float(last_beat['rr_baseline']) if pd.notna(last_beat['rr_baseline']) else np.nan
        if not np.isfinite(baseline_s):
            norm_n = seg_n[(seg_n['beat_type'] == 'normal') & seg_n['rr_baseline'].notna()]
            if norm_n.empty:
                continue
            baseline_s = float(norm_n['rr_baseline'].iloc[-1])

        if not np.isfinite(baseline_s) or baseline_s <= 0:
            continue
        if cross_rr <= missed_thresh * baseline_s:
            continue

        # Normal-beat amplitude reference (from seg N).
        norm_amps = pd.to_numeric(
            seg_n.loc[seg_n['beat_type'] == 'normal', 'r_amplitude'], errors='coerce'
        ).dropna()
        normal_amp = float(norm_amps.median()) if len(norm_amps) else np.nan

        sig_n  = _load(fn_n)
        sig_n1 = _load(fn_n1)
        if sig_n is None or sig_n1 is None:
            continue

        # Build the gap signal: tail of N  +  head of N+1.
        # first_r == 0 → head is empty; the full gap is in the tail of N.
        tail = sig_n[last_r:]
        head = sig_n1[:first_r] if first_r > 0 else np.array([], dtype=float)
        gap_signal = np.concatenate([tail, head])
        gap_len = len(gap_signal)

        if gap_len < 10:
            continue

        # Search zone: same T-wave clearance logic as intra-segment scanner.
        margin  = max(int(gap_len * 0.20), 5)
        left_margin = max(margin, min(MIN_T_SKIP, gap_len - margin - 10))
        s_start = left_margin
        s_end   = gap_len - margin
        if s_end <= s_start:
            continue

        # ── Slide the template window across the gap ──────────────────────────
        ctype      = 'weak'
        morph_corr = np.nan
        candidate_local = (s_start + s_end) // 2   # fallback
        candidate_amp   = float(gap_signal[candidate_local])

        if library is not None:
            score_fn, (bef, aft) = _get_score_beat()
            labels_list = list(library.get('mean_templates', {}).keys())
            best_center = None
            best_label  = None
            best_corr   = -np.inf

            for center in range(s_start, s_end):
                lo = center - bef
                hi = center + aft + 1
                if lo < 0 or hi > len(gap_signal):
                    continue
                win = gap_signal[lo:hi].astype(float)
                for lbl in labels_list:
                    c = score_fn(win, library, label=lbl)
                    if np.isfinite(c) and c > best_corr:
                        best_corr  = c
                        best_label = lbl
                        best_center = center

            # S-dominant post-check: re-locate if best candidate is a T-wave
            if best_center is not None and best_corr >= morph_threshold:
                best_center, best_corr, best_label = _s_dominant_post_check(
                    gap_signal, s_start, best_center, best_corr, best_label,
                    score_fn, labels_list, library, morph_threshold,
                    bef, aft, normal_amp,
                )

            if best_center is not None:
                candidate_local = best_center
                candidate_amp   = float(gap_signal[best_center])
                if best_label == 'normal' or best_corr >= morph_threshold:
                    ctype      = best_label
                    morph_corr = round(float(best_corr), 3)
        else:
            # Heuristic fallback (no library): dominant peak + polarity.
            window      = gap_signal[s_start:s_end]
            max_pos     = float(np.max(window))
            max_neg     = float(np.min(window))
            if abs(max_neg) > abs(max_pos):
                candidate_amp   = max_neg
                candidate_local = int(np.argmin(window)) + s_start
                polarity        = 'negative'
            else:
                candidate_amp   = max_pos
                candidate_local = int(np.argmax(window)) + s_start
                polarity        = 'positive'

            if not np.isfinite(normal_amp) or normal_amp <= 0:
                ctype = 'unknown'
            else:
                ratio = abs(candidate_amp) / abs(normal_amp)
                if ratio < 0.35:
                    ctype = 'weak'
                elif polarity == 'negative':
                    ctype = 'PVC'
                elif ratio > 1.5:
                    ctype = 'PVC'
                else:
                    ctype = 'normal'

        # Map candidate_local back to (file, r_index, signal array).
        tail_len = len(tail)
        if candidate_local < tail_len:
            cand_file = fn_n
            cand_r    = last_r + candidate_local
            cand_sig  = sig_n
        else:
            cand_file = fn_n1
            cand_r    = candidate_local - tail_len   # local index in N+1
            cand_sig  = sig_n1

        results.append({
            'filename':       fn_n1,       # N+1 — for viewer look-up
            'boundary':       'cross_segment',
            'prev_file':      fn_n,
            'missed_at_r':    first_r,     # first beat of N+1 (gap end)
            'prev_r':         last_r,      # last beat of N  (gap start)
            'gap_s':          round(cross_rr, 3),
            'baseline_s':     round(baseline_s, 3),
            'candidate_file': cand_file,
            'candidate_r':    cand_r,
            'candidate_amp':  round(candidate_amp, 1),
            'normal_amp_med': round(normal_amp, 1) if np.isfinite(normal_amp) else np.nan,
            'amp_ratio':      round(abs(candidate_amp) / abs(normal_amp), 2)
                              if np.isfinite(normal_amp) and normal_amp > 0 else np.nan,
            'candidate_type': ctype,
            'morph_corr':     morph_corr,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()
