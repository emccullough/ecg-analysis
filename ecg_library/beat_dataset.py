import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

from .filters import filter_signal

SAMPLES_PER_SEGMENT = 2500
FS = 125
DS_FACTOR = 4
DS_LENGTH = SAMPLES_PER_SEGMENT // DS_FACTOR  # 625

# Beat type → class ID
BEAT_TYPE_MAP = {
    'normal': 1,
    'Normal': 1,
    'PVC': 2,
    'VT': 0,
    'PAC': 0,
    'AV': 0,
    'Artifact': 0,
    'Other': 0,
    'Deleted': -1,   # beat marked as false-positive — excluded from training
}

CLASS_NAMES = {0: 'Other', 1: 'Normal', 2: 'PVC'}
NUM_CLASSES = 3


def _build_index(data_dir, annotations_path=None):
    """Scan all sessions and build a list of segment entries with beat labels.

    Returns:
        index: list of dicts with keys session, session_dir, segment_file,
               segment_idx, beats (list of (local_r_index, class_id))
        all_sessions: sorted list of unique session names
    """
    # Load human annotations for label overrides
    annotations = {}
    if annotations_path and os.path.exists(annotations_path):
        ann_df = pd.read_csv(annotations_path)
        for _, row in ann_df.iterrows():
            key = (row['session'], int(row['r_index']))
            annotations[key] = row['human_label']

    rich_files = glob.glob(os.path.join(data_dir, '**', '*_rich_processed_beats.csv'), recursive=True)

    index = []
    session_set = set()

    for rich_path in sorted(rich_files):
        session_dir = os.path.dirname(rich_path)
        session_name = os.path.basename(session_dir)

        try:
            df = pd.read_csv(rich_path)
        except Exception:
            continue

        if 'r_index' not in df.columns or 'beat_type' not in df.columns:
            continue

        # Get ecg_N.csv files (skip ecg_0.csv which is metadata)
        ecg_files = sorted(
            glob.glob(os.path.join(session_dir, 'ecg_*.csv')),
            key=lambda f: int(re.search(r'ecg_(\d+)\.csv', f).group(1))
        )
        ecg_files = [f for f in ecg_files if not f.endswith('ecg_0.csv')]

        if not ecg_files:
            continue

        # Group beats by segment — track both class IDs and raw types
        segment_beats = {}    # seg_idx -> [(local_idx, class_id), ...]
        segment_types = {}    # seg_idx -> set of raw beat_type strings
        for _, row in df.iterrows():
            r_idx = int(row['r_index'])
            beat_type = row['beat_type']

            # Override with human annotation if available
            ann_key = (session_name, r_idx)
            if ann_key in annotations:
                beat_type = annotations[ann_key]

            class_id = BEAT_TYPE_MAP.get(beat_type, 0)
            seg_idx = r_idx // SAMPLES_PER_SEGMENT
            local_idx = r_idx % SAMPLES_PER_SEGMENT

            if seg_idx not in segment_beats:
                segment_beats[seg_idx] = []
                segment_types[seg_idx] = set()
            segment_beats[seg_idx].append((local_idx, class_id))
            segment_types[seg_idx].add(beat_type)

        # Add "undetected" beats from annotations (algo missed, human found)
        for (sess, r_idx), human_label in annotations.items():
            if sess != session_name:
                continue
            if not ((df['r_index'] == r_idx).any()):
                class_id = BEAT_TYPE_MAP.get(human_label, 0)
                seg_idx = r_idx // SAMPLES_PER_SEGMENT
                local_idx = r_idx % SAMPLES_PER_SEGMENT
                if seg_idx not in segment_beats:
                    segment_beats[seg_idx] = []
                    segment_types[seg_idx] = set()
                segment_beats[seg_idx].append((local_idx, class_id))
                segment_types[seg_idx].add(human_label)

        n_segments = len(ecg_files)

        # Create index entries
        for seg_idx, beats in segment_beats.items():
            if seg_idx >= n_segments:
                continue
            ecg_path = ecg_files[seg_idx]
            # Exclude beats explicitly marked as deleted (false-positives)
            sorted_beats = sorted(
                [(r, c) for r, c in beats if c != -1],
                key=lambda x: x[0],
            )
            if not sorted_beats:
                continue
            class_ids = {c for _, c in sorted_beats}
            raw_types = segment_types.get(seg_idx, set())
            index.append({
                'session': session_name,
                'session_dir': session_dir,
                'segment_file': ecg_path,
                'segment_idx': seg_idx,
                'n_segments': n_segments,
                'beats': sorted_beats,
                'has_artifact': 'Artifact' in raw_types,
                'has_pvc': 2 in class_ids,
                'all_normal': class_ids == {1},
            })
            session_set.add(session_name)

    return index, sorted(session_set)


def _load_segment(segment_file):
    """Load a single ecg_N.csv file (2500 samples)."""
    data = pd.read_csv(segment_file, header=None, comment='#')[0].values.astype(np.float64)
    return data


def _make_targets(beats, sigma=4, beat_radius=8):
    """Create beat heatmap [2500] and class map [2500], then downsample to [625].

    Args:
        beats: list of (local_r_index, class_id)
        sigma: Gaussian sigma for beat heatmap (in samples at 2500 resolution)
        beat_radius: radius around R-peak for class map (in samples at 2500 resolution)

    Returns:
        beat_heatmap_ds: [625] float32 downsampled beat probability
        class_map_ds: [625] int64 downsampled class IDs
    """
    beat_heatmap = np.zeros(SAMPLES_PER_SEGMENT, dtype=np.float32)
    class_map = np.zeros(SAMPLES_PER_SEGMENT, dtype=np.int64)

    for r_idx, class_id in beats:
        if 0 <= r_idx < SAMPLES_PER_SEGMENT:
            beat_heatmap[r_idx] = 1.0
            start = max(0, r_idx - beat_radius)
            end = min(SAMPLES_PER_SEGMENT, r_idx + beat_radius + 1)
            class_map[start:end] = class_id

    # Gaussian smoothing
    if beat_heatmap.sum() > 0:
        beat_heatmap = gaussian_filter1d(beat_heatmap, sigma=sigma)
        beat_heatmap = beat_heatmap / beat_heatmap.max()

    # Downsample to [625]
    beat_heatmap_ds = beat_heatmap.reshape(DS_LENGTH, DS_FACTOR).max(axis=1)
    class_map_ds = class_map.reshape(DS_LENGTH, DS_FACTOR).max(axis=1)

    return beat_heatmap_ds, class_map_ds


class ECGBeatDataset(Dataset):
    """PyTorch dataset for ECG beat detection and classification.

    Each item is a 20-second ECG segment (2500 samples at 125 Hz) with
    beat heatmap and class map targets downsampled to 625 positions.
    """

    def __init__(self, index, augment=False, preload=True):
        """
        Args:
            index: pre-built list of segment dicts from _build_index
            augment: apply data augmentation (amplitude scaling, noise, time shift)
            preload: load + filter all signals into memory at init so each epoch
                     pays zero disk I/O or scipy cost (recommended on M-series Macs)
        """
        self.index = index
        self.augment = augment
        self._cache = None

        if preload and len(index) > 0:
            print(f"  Preloading {len(index)} segments into memory...", end=' ', flush=True)
            cache = []
            for entry in index:
                raw = _load_segment(entry['segment_file'])
                if len(raw) < SAMPLES_PER_SEGMENT:
                    raw = np.pad(raw, (0, SAMPLES_PER_SEGMENT - len(raw)))
                elif len(raw) > SAMPLES_PER_SEGMENT:
                    raw = raw[:SAMPLES_PER_SEGMENT]
                filtered = filter_signal(raw, FS)
                std = np.std(filtered)
                if std > 1e-6:
                    filtered = (filtered - np.mean(filtered)) / std
                cache.append(filtered.astype(np.float32))
            self._cache = cache
            print("done.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]

        if self._cache is not None:
            signal = self._cache[idx].copy()
        else:
            raw = _load_segment(entry['segment_file'])
            if len(raw) < SAMPLES_PER_SEGMENT:
                raw = np.pad(raw, (0, SAMPLES_PER_SEGMENT - len(raw)))
            elif len(raw) > SAMPLES_PER_SEGMENT:
                raw = raw[:SAMPLES_PER_SEGMENT]
            filtered = filter_signal(raw, FS)
            std = np.std(filtered)
            if std > 1e-6:
                filtered = (filtered - np.mean(filtered)) / std
            signal = filtered.astype(np.float32)

        beats = list(entry['beats'])

        # Augmentation
        if self.augment:
            # Amplitude scaling (0.8–1.2x)
            scale = np.random.uniform(0.8, 1.2)
            signal = signal * scale

            # Gaussian noise (sigma=0.02)
            noise = np.random.normal(0, 0.02, signal.shape).astype(np.float32)
            signal = signal + noise

            # Time shift (±10 samples)
            shift = np.random.randint(-10, 11)
            if shift != 0:
                signal = np.roll(signal, shift)
                beats = [
                    (max(0, min(SAMPLES_PER_SEGMENT - 1, r + shift)), c)
                    for r, c in beats
                ]

        # Create downsampled targets
        beat_heatmap_ds, class_map_ds = _make_targets(beats)

        return {
            'signal': torch.from_numpy(signal).unsqueeze(0),         # [1, 2500]
            'beat_heatmap': torch.from_numpy(beat_heatmap_ds),        # [625]
            'class_map': torch.from_numpy(class_map_ds),              # [625]
            'beats': beats,                                            # [(r_idx, class_id), ...]
            'session': entry['session'],
            'segment_idx': entry['segment_idx'],
        }


def collate_fn(batch):
    """Custom collate that handles variable-length beats lists."""
    return {
        'signal': torch.stack([b['signal'] for b in batch]),
        'beat_heatmap': torch.stack([b['beat_heatmap'] for b in batch]),
        'class_map': torch.stack([b['class_map'] for b in batch]),
        'beats': [b['beats'] for b in batch],
        'session': [b['session'] for b in batch],
        'segment_idx': [b['segment_idx'] for b in batch],
    }


def build_datasets(data_dir='Data', annotations_path='annotations.csv',
                   train_ratio=0.8, val_ratio=0.1, seed=42):
    """Build train/val/test datasets split by session.

    Returns:
        train_ds, val_ds, test_ds: ECGBeatDataset instances
        split_info: dict with session lists and counts
    """
    index, all_sessions = _build_index(data_dir, annotations_path)

    rng = np.random.RandomState(seed)
    sessions = list(all_sessions)
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_sessions = set(sessions[:n_train])
    val_sessions = set(sessions[n_train:n_train + n_val])
    test_sessions = set(sessions[n_train + n_val:])

    train_idx = [e for e in index if e['session'] in train_sessions]
    val_idx = [e for e in index if e['session'] in val_sessions]
    test_idx = [e for e in index if e['session'] in test_sessions]

    train_ds = ECGBeatDataset(train_idx, augment=True)
    val_ds = ECGBeatDataset(val_idx, augment=False)
    test_ds = ECGBeatDataset(test_idx, augment=False)

    split_info = {
        'train_sessions': sorted(train_sessions),
        'val_sessions': sorted(val_sessions),
        'test_sessions': sorted(test_sessions),
        'train_segments': len(train_idx),
        'val_segments': len(val_idx),
        'test_segments': len(test_idx),
        'total_sessions': n,
        'total_segments': len(index),
    }

    return train_ds, val_ds, test_ds, split_info
