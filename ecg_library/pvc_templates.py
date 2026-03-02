"""
PVC / VT waveform template library.

Builds a cross-session template from confirmed labelled beats, normalises
each QRS window to unit-norm so that correlation is purely shape-based
(not amplitude-based).  The library is stored as a compressed NumPy archive
and can be updated incrementally as more curated sessions are added.

Typical workflow
----------------
1. Curate a session (fix v1 labels, review missed-beat candidates).
2. Call ``extract_waveforms()`` on the curated beats DataFrame to get
   normalised QRS windows for every confirmed PVC / VT beat.
3. Call ``build_template_library()`` to compute per-label mean templates
   and save everything to ``pvc_templates.npz``.
4. Later sessions: load the library with ``load_template_library()`` and
   use ``score_beat()`` to correlate any beat window against the stored
   mean templates.

Scoring
-------
Pearson correlation is used throughout (zero-mean + unit-std per window),
so a correlation of 1.0 means an exact shape match regardless of amplitude.
Typical thresholds:  corr > 0.80 = strong match, > 0.70 = moderate match.
"""

import os

import numpy as np
import pandas as pd


# Default window: 20 samples before the R-peak, 30 samples after.
# Total 51 samples = 408 ms at 125 Hz — enough to capture the QRS complex
# and the early ST segment without bleeding into the previous T-wave.
DEFAULT_BEFORE = 20
DEFAULT_AFTER  = 30


def _pearson_normalise(window: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation (Pearson norm)."""
    mu  = window.mean()
    std = window.std()
    if std < 1e-6:
        return np.zeros_like(window)
    return (window - mu) / std


def extract_waveforms(
    beats_df,
    session_dir,
    labels=('PVC', 'VT'),
    fs=125,
    before=DEFAULT_BEFORE,
    after=DEFAULT_AFTER,
):
    """Extract normalised QRS windows for every beat matching *labels*.

    Parameters
    ----------
    beats_df : pd.DataFrame
        v1 (or v2) beat DataFrame — must have ``filename``, ``r_index``,
        ``beat_type``.  Pass a pre-filtered DataFrame to restrict which
        beats are included (e.g. only the first N segments of a session,
        or only beats you have manually verified).
    session_dir : str
        Folder containing the ecg_N.csv raw segment files.
    labels : sequence of str
        Beat types to extract (default ``('PVC', 'VT')``).
    fs : int
        Sampling frequency in Hz.
    before, after : int
        Samples to include before and after the R-peak.

    Returns
    -------
    waveforms : np.ndarray, shape (N, before+after+1)
        Pearson-normalised QRS windows, one row per beat.
    meta : pd.DataFrame
        One row per extracted beat with columns
        ``label``, ``session``, ``filename``, ``r_index``, ``r_amplitude``.
    """
    from ecg_library.filters import filter_signal

    SEG_LEN  = fs * 20
    win_len  = before + after + 1
    target   = set(labels)

    waveforms = []
    meta_rows = []

    session_name = os.path.basename(session_dir)
    sig_cache    = {}

    for fname, seg in beats_df.groupby('filename', sort=False):
        candidates = seg[seg['beat_type'].isin(target)]
        if candidates.empty:
            continue

        if fname not in sig_cache:
            path = os.path.join(session_dir, fname)
            if not os.path.exists(path):
                continue
            raw = pd.read_csv(path, header=None, comment='#')[0].values.astype(float)
            sig_cache[fname] = filter_signal(raw, fs)

        sig = sig_cache[fname]

        for _, beat in candidates.iterrows():
            r_local = int(beat['r_index']) % SEG_LEN
            lo = r_local - before
            hi = r_local + after + 1
            if lo < 0 or hi > len(sig):
                continue  # skip beats too close to segment boundary

            window = sig[lo:hi].astype(float)
            norm   = _pearson_normalise(window)

            waveforms.append(norm)
            meta_rows.append({
                'label':       beat['beat_type'],
                'session':     session_name,
                'filename':    fname,
                'r_index':     int(beat['r_index']),
                'r_amplitude': float(beat['r_amplitude'])
                               if pd.notna(beat.get('r_amplitude')) else np.nan,
            })

    if not waveforms:
        return np.empty((0, win_len)), pd.DataFrame()

    return np.array(waveforms, dtype=np.float32), pd.DataFrame(meta_rows)


def build_template_library(waveforms, meta, save_path):
    """Compute per-label mean templates and save the full library to disk.

    Parameters
    ----------
    waveforms : np.ndarray, shape (N, win_len)
    meta : pd.DataFrame   (from ``extract_waveforms``)
    save_path : str       Path to save the ``.npz`` file.

    Saved arrays
    ------------
    ``waveforms``        All normalised windows  (N × win_len)
    ``labels``           Beat-type string per row
    ``sessions``         Session name per row
    ``filenames``        Segment file per row
    ``r_indices``        R-peak index per row
    ``r_amplitudes``     Original amplitude per row
    ``mean_labels``      Unique labels that have a mean template
    ``mean_<LABEL>``     Mean template for each label (e.g. ``mean_PVC``)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    save_dict = {
        'waveforms':   waveforms,
        'labels':      meta['label'].values.astype(str),
        'sessions':    meta['session'].values.astype(str),
        'filenames':   meta['filename'].values.astype(str),
        'r_indices':   meta['r_index'].values.astype(np.int32),
        'r_amplitudes': meta['r_amplitude'].values.astype(np.float32),
    }

    unique_labels = sorted(meta['label'].unique())
    save_dict['mean_labels'] = np.array(unique_labels, dtype=str)

    for lbl in unique_labels:
        mask = meta['label'].values == lbl
        mean_template = waveforms[mask].mean(axis=0)
        save_dict[f'mean_{lbl}'] = mean_template.astype(np.float32)

    np.savez_compressed(save_path, **save_dict)

    print(f'Template library saved → {save_path}')
    for lbl in unique_labels:
        n = (meta['label'] == lbl).sum()
        print(f'  {lbl}: {n} waveforms')


def load_template_library(path):
    """Load a saved template library.

    Returns
    -------
    dict with keys:
        ``waveforms``        np.ndarray (N × win_len)
        ``meta``             pd.DataFrame (label, session, filename, r_index, r_amplitude)
        ``mean_templates``   dict  {label: mean_waveform_array}
    """
    data = np.load(path, allow_pickle=True)

    meta = pd.DataFrame({
        'label':       data['labels'],
        'session':     data['sessions'],
        'filename':    data['filenames'],
        'r_index':     data['r_indices'],
        'r_amplitude': data['r_amplitudes'],
    })

    mean_templates = {}
    for lbl in data['mean_labels']:
        key = f'mean_{lbl}'
        if key in data:
            mean_templates[lbl] = data[key]

    return {
        'waveforms':      data['waveforms'],
        'meta':           meta,
        'mean_templates': mean_templates,
    }


def score_beat(window, library, label='PVC'):
    """Pearson correlation of *window* against the mean template for *label*.

    Parameters
    ----------
    window : np.ndarray   Raw (unnormalised) QRS window from the ECG signal.
    library : dict        Output of ``load_template_library``.
    label : str           Template class to compare against (default 'PVC').

    Returns
    -------
    float
        Pearson correlation in [-1, 1].  Returns NaN if the template is
        not in the library or the window is flat.
    """
    mean_templates = library.get('mean_templates', {})
    if label not in mean_templates:
        return np.nan

    norm_query    = _pearson_normalise(np.asarray(window, dtype=float))
    norm_template = _pearson_normalise(mean_templates[label].astype(float))

    return float(np.dot(norm_query, norm_template) / len(norm_query))


def find_morph_candidates(
    beats_df,
    session_dir,
    library,
    threshold=0.80,
    normal_only=True,
    fs=125,
    before=DEFAULT_BEFORE,
    after=DEFAULT_AFTER,
):
    """Score beats against the template library and return high-correlation candidates.

    Parameters
    ----------
    beats_df : pd.DataFrame
        v2 beats DataFrame with ``filename``, ``r_index``, ``beat_type``,
        ``r_amplitude``.
    session_dir : str
        Folder containing the raw ecg_N.csv files.
    library : dict
        Output of ``load_template_library``.
    threshold : float
        Minimum Pearson correlation to report (default 0.80).
    normal_only : bool
        If True (default), only score beats labelled 'normal' — i.e. look for
        PVC/VT beats that v1 called normal.
    fs, before, after : int
        Must match the window used to build the library.

    Returns
    -------
    pd.DataFrame with columns:
        ``filename``, ``r_index``, ``r_amplitude``, ``best_label``, ``best_corr``
    """
    from ecg_library.filters import filter_signal

    SEG_LEN = fs * 20
    labels  = list(library.get('mean_templates', {}).keys())
    if not labels:
        return pd.DataFrame()

    source    = beats_df[beats_df['beat_type'] == 'normal'] if normal_only else beats_df
    sig_cache = {}
    rows      = []

    for fname, seg in source.groupby('filename', sort=False):
        if fname not in sig_cache:
            path = os.path.join(session_dir, fname)
            if not os.path.exists(path):
                continue
            raw = pd.read_csv(path, header=None, comment='#')[0].values.astype(float)
            sig_cache[fname] = filter_signal(raw, fs)

        sig = sig_cache[fname]

        for _, beat in seg.iterrows():
            r_local = int(beat['r_index']) % SEG_LEN
            lo = r_local - before
            hi = r_local + after + 1
            if lo < 0 or hi > len(sig):
                continue

            window     = sig[lo:hi].astype(float)
            best_label = None
            best_corr  = -np.inf

            for lbl in labels:
                corr = score_beat(window, library, label=lbl)
                if np.isfinite(corr) and corr > best_corr:
                    best_corr  = corr
                    best_label = lbl

            if best_label is not None and best_corr >= threshold:
                rows.append({
                    'filename':    fname,
                    'r_index':     int(beat['r_index']),
                    'r_amplitude': float(beat['r_amplitude']),
                    'best_label':  best_label,
                    'best_corr':   round(float(best_corr), 3),
                })

    return pd.DataFrame(rows)


def rescore_vt_s_dominant(
    beats_df,
    session_dir,
    library=None,
    morph_threshold=0.80,
    min_r_amp=30,
    max_rs_gap=8,
    fs=125,
    before=DEFAULT_BEFORE,
    after=DEFAULT_AFTER,
):
    """Correct VT-labeled beats where v1 detected the S-wave trough instead of the R-peak.

    During high-intensity exercise the S-wave can exceed the R-wave in absolute
    amplitude (S-dominant QRS).  v1's peak-picker finds the large negative
    S-trough and classifies it as an inverted-QRS VT beat.  The true R-peak is
    a small positive spike that arrives a few samples *before* the S-trough.

    Only modifies beats where **all** of the following hold:

    * ``beat_type == 'VT'``  **and**  ``r_amplitude < 0``
      (v1 detected a negative peak — the S-trough)
    * A positive peak ``>= min_r_amp`` exists within ``max_rs_gap`` samples
      **before** the detected trough.  At 125 Hz the normal RS interval is
      2–8 samples (16–64 ms), so ``max_rs_gap=8`` is the physiological limit.

    Relabeling decision (at the corrected true R position):

    * PVC template correlation >= *morph_threshold* → ``'PVC'``
    * Otherwise                                     → ``'normal'``

    ``r_index`` and ``r_amplitude`` are updated to the true R-peak position.

    Parameters
    ----------
    beats_df : pd.DataFrame   v2 beat DataFrame.
    session_dir : str          Folder containing the ecg_N.csv files.
    library : dict or None     Template library from ``load_template_library``.
    morph_threshold : float    Min PVC correlation to relabel as PVC (default 0.80).
    min_r_amp : float          Min amplitude (ADU) to accept as a real R-peak.
    max_rs_gap : int           Max samples before the S-trough to search for R.
    fs, before, after : int    Must match the window used to build the library.

    Returns
    -------
    pd.DataFrame  Modified copy of *beats_df*.
    """
    from ecg_library.filters import filter_signal

    df = beats_df.copy()
    # Ensure r_amplitude is float so we can assign filtered-signal values
    df['r_amplitude'] = pd.to_numeric(df['r_amplitude'], errors='coerce').astype(float)
    r_amp_col = df['r_amplitude']
    vt_mask = (df['beat_type'] == 'VT') & (r_amp_col < 0)
    if not vt_mask.any():
        return df

    SEG_LEN   = fs * 20
    sig_cache = {}

    for fname, seg in df[vt_mask].groupby('filename', sort=False):
        path = os.path.join(session_dir, fname)
        if not os.path.exists(path):
            continue
        if fname not in sig_cache:
            raw = pd.read_csv(path, header=None, comment='#')[0].values.astype(float)
            sig_cache[fname] = filter_signal(raw, fs)
        sig = sig_cache[fname]

        for idx, beat in seg.iterrows():
            r_local = int(beat['r_index']) % SEG_LEN

            # Search for positive R in [r_local - max_rs_gap, r_local - 1]
            search_start = max(0, r_local - max_rs_gap)
            if search_start >= r_local:
                continue
            zone     = sig[search_start:r_local]
            best_pos = int(np.argmax(zone))
            best_amp = float(zone[best_pos])

            if best_amp < min_r_amp:
                continue  # no positive R found → leave as VT

            true_r_local = search_start + best_pos

            # Score against PVC template if library available
            new_label = 'normal'
            if library is not None:
                lo = true_r_local - before
                hi = true_r_local + after + 1
                if lo >= 0 and hi <= len(sig):
                    window = sig[lo:hi].astype(float)
                    corr = score_beat(window, library, label='PVC')
                    if np.isfinite(corr) and corr >= morph_threshold:
                        new_label = 'PVC'

            # Update r_index (works for both local and global storage)
            new_r_global = int(beat['r_index']) - r_local + true_r_local
            df.at[idx, 'beat_type']   = new_label
            df.at[idx, 'r_index']     = new_r_global
            df.at[idx, 'r_amplitude'] = round(best_amp, 1)

    return df


def library_summary(library):
    """Print a summary of a loaded template library."""
    meta = library['meta']
    print(f"Template library: {len(meta)} waveforms")
    print(f"  Sessions : {sorted(meta['session'].unique())}")
    for lbl, grp in meta.groupby('label'):
        sessions = grp['session'].unique()
        print(f"  {lbl:10s}: {len(grp):4d} waveforms from {len(sessions)} session(s)")
        print(f"             amp range: {grp['r_amplitude'].min():.0f} – "
              f"{grp['r_amplitude'].max():.0f}")
