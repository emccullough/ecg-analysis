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

import numpy as np
import pandas as pd

# RR values above this are definite pauses — never a normal beat interval.
# 1.5 s ≈ 40 bpm; no healthy resting heart rate is slower than this.
MAX_PAUSE_RR = 1.5  # seconds


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
