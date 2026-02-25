"""
RR-interval anomaly analysis — v2 signal processing layer.

Reads the beat DataFrame produced by v1 (rich_processed_beats.csv) and adds
three columns that capture timing-based anomalies independent of morphology:

    rr_baseline    rolling median RR of the preceding `window` beats (seconds)
    rr_flag        'premature' | 'compensatory' | 'missed_beat' | ''
    rr_burden_pct  % of premature beats in this segment (same value for all
                   beats in the same segment, used as a per-segment stat)

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

import numpy as np
import pandas as pd


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
        Output of v1 processing — must contain columns ``rr_interval`` and
        ``filename`` (one row per beat, sorted in time order within each file).
    window : int
        Number of preceding beats used for the rolling median baseline.
        Minimum 2 beats required before any flag is assigned.
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

        baseline = np.full(len(rr), np.nan)
        flags = np.full(len(rr), '', dtype=object)

        # Rolling median over the window of preceding beats
        for i in range(len(rr)):
            start = max(0, i - window)
            vals = rr[start:i]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                baseline[i] = np.median(vals)

        # Classify each beat using the baseline
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
