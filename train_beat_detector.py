#!/usr/bin/env python3
"""Standalone training script for ECG beat detector."""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_library.beat_dataset import (
    ECGBeatDataset, collate_fn, _build_index, CLASS_NAMES, NUM_CLASSES,
    SAMPLES_PER_SEGMENT, DS_LENGTH,
)
from ecg_library.beat_detector import (
    CNNTransformerDetector, compute_loss, detect_beats, evaluate_detection,
)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load annotated sessions only
    ann_df = pd.read_csv('annotations.csv')
    annotated_sessions = sorted(ann_df['session'].unique().tolist())
    print(f"Annotated sessions ({len(annotated_sessions)}):")
    for sess in annotated_sessions:
        n_ann = len(ann_df[ann_df['session'] == sess])
        labels = ann_df[ann_df['session'] == sess]['human_label'].value_counts().to_dict()
        print(f"  {sess}: {n_ann} annotations — {labels}")

    # Build index filtered to annotated sessions
    index, _ = _build_index('Data', 'annotations.csv')
    curated_index = [e for e in index if e['session'] in annotated_sessions]
    print(f"\nTotal curated segments (before filtering): {len(curated_index)}")

    # --- Apply training filters ---
    # 1. Remove first 2 min (6 segments) and last 1 min (3 segments) of each session
    SKIP_START = 6   # 6 × 20s = 120s = 2 min
    SKIP_END = 3     # 3 × 20s = 60s = 1 min
    before = len(curated_index)
    curated_index = [
        e for e in curated_index
        if e['segment_idx'] >= SKIP_START
        and e['segment_idx'] < (e['n_segments'] - SKIP_END)
    ]
    print(f"  After trimming first {SKIP_START} / last {SKIP_END} segments: {len(curated_index)} ({before - len(curated_index)} removed)")

    # 2. Remove segments with artifacts
    before = len(curated_index)
    curated_index = [e for e in curated_index if not e['has_artifact']]
    print(f"  After removing artifact segments: {len(curated_index)} ({before - len(curated_index)} removed)")

    # 3. Remove segments with only normal beats (keep PVC, VT, Other, etc.)
    before = len(curated_index)
    curated_index = [e for e in curated_index if not e['all_normal']]
    print(f"  After removing normal-only segments: {len(curated_index)} ({before - len(curated_index)} removed)")

    print(f"\nFiltered curated segments: {len(curated_index)}")

    # Split: smallest session → val, rest → train
    seg_counts = Counter(e['session'] for e in curated_index)
    print("\nSegments per session:")
    for sess, cnt in seg_counts.most_common():
        print(f"  {sess}: {cnt}")

    sessions_by_size = [sess for sess, _ in seg_counts.most_common()]
    val_session = sessions_by_size[-1]
    train_sessions = [s for s in sessions_by_size if s != val_session]

    print(f"\nSplit:")
    print(f"  Train: {train_sessions}")
    print(f"  Val:   [{val_session}]")

    train_idx = [e for e in curated_index if e['session'] in train_sessions]
    val_idx = [e for e in curated_index if e['session'] == val_session]

    train_ds = ECGBeatDataset(train_idx, augment=True)
    val_ds = ECGBeatDataset(val_idx, augment=False)

    print(f"  Train: {len(train_sessions)} sessions, {len(train_idx)} segments")
    print(f"  Val:   1 session, {len(val_idx)} segments")

    # Count beats per class in training set
    train_counts = Counter()
    for entry in train_ds.index:
        for _, cls in entry['beats']:
            train_counts[cls] += 1
    print(f"\nTraining beats by class:")
    for cls_id, name in CLASS_NAMES.items():
        print(f"  {name} ({cls_id}): {train_counts.get(cls_id, 0):,}")

    # Class weights (inverse frequency) + focal loss to handle imbalance
    total_beats = sum(train_counts.values())
    base_weights = [
        total_beats / (NUM_CLASSES * max(train_counts.get(i, 1), 1))
        for i in range(NUM_CLASSES)
    ]
    class_weights = torch.tensor(base_weights, dtype=torch.float32).to(device)
    FOCAL_GAMMA = 2.0
    CLASS_LOSS_WEIGHT = 2.0  # upweight classification vs detection
    print(f"Class weights: {class_weights}")
    print(f"Focal gamma={FOCAL_GAMMA}, class_loss_weight={CLASS_LOSS_WEIGHT}")

    # Model
    model = CNNTransformerDetector(
        d_model=128, nhead=8, dim_ff=256, num_layers=4,
        num_classes=NUM_CLASSES, dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # DataLoaders — num_workers=0 to avoid multiprocessing issues
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    PATIENCE = 10

    # Oversample PVC-containing segments so model sees more PVC morphology
    from torch.utils.data import WeightedRandomSampler
    PVC_OVERSAMPLE = 5.0
    sample_weights = []
    n_pvc_segs = 0
    for entry in train_ds.index:
        has_pvc = any(c == 2 for _, c in entry['beats'])
        if has_pvc:
            sample_weights.append(PVC_OVERSAMPLE)
            n_pvc_segs += 1
        else:
            sample_weights.append(1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"PVC segments: {n_pvc_segs}/{len(train_ds)} ({100*n_pvc_segs/len(train_ds):.1f}%), oversampled {PVC_OVERSAMPLE}x")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_detect_f1': [], 'val_pvc_recall': [], 'lr': []}

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            signal = batch['signal'].to(device)
            beat_target = batch['beat_heatmap'].to(device)
            class_target = batch['class_map'].to(device)

            beat_prob, class_logits = model(signal)
            loss, loss_dict = compute_loss(
                beat_prob, class_logits, beat_target, class_target,
                pos_weight=10.0, class_weights=class_weights,
                focal_gamma=FOCAL_GAMMA, class_loss_weight=CLASS_LOSS_WEIGHT,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss_dict['total'])

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_losses = []
        all_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                signal = batch['signal'].to(device)
                beat_target = batch['beat_heatmap'].to(device)
                class_target = batch['class_map'].to(device)

                beat_prob, class_logits = model(signal)
                loss, loss_dict = compute_loss(
                    beat_prob, class_logits, beat_target, class_target,
                    pos_weight=10.0, class_weights=class_weights,
                )
                val_losses.append(loss_dict['total'])

                for i in range(signal.size(0)):
                    pred = detect_beats(beat_prob[i], class_logits[i])
                    true = batch['beats'][i]
                    metrics = evaluate_detection(pred, true)
                    all_metrics.append(metrics)

        # Aggregate
        total_tp = sum(m['tp'] for m in all_metrics)
        total_fp = sum(m['fp'] for m in all_metrics)
        total_fn = sum(m['fn'] for m in all_metrics)
        val_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        val_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0

        # PVC recall
        pvc_tp = sum(m['per_class'].get('PVC', {}).get('tp', 0) for m in all_metrics)
        pvc_fn = sum(m['per_class'].get('PVC', {}).get('fn', 0) for m in all_metrics)
        pvc_recall = pvc_tp / (pvc_tp + pvc_fn) if (pvc_tp + pvc_fn) > 0 else 0

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_detect_f1'].append(val_f1)
        history['val_pvc_recall'].append(pvc_recall)
        history['lr'].append(current_lr)

        # Combined metric: detection F1 + PVC recall (both matter)
        combined_score = 0.5 * val_f1 + 0.5 * pvc_recall

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Det F1: {val_f1:.4f} | PVC Rec: {pvc_recall:.4f} | "
              f"Score: {combined_score:.4f} | LR: {current_lr:.6f}",
              flush=True)

        if combined_score > best_val_f1:
            best_val_f1 = combined_score
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'pvc_recall': pvc_recall,
                'combined_score': combined_score,
                'history': history,
            }, 'ecg_beat_detector_best.pt')
            print(f"  -> New best! Saved checkpoint.", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  -> Early stopping after {PATIENCE} epochs without improvement.", flush=True)
                break

    print(f"\nBest combined score (0.5*F1 + 0.5*PVC_rec): {best_val_f1:.4f}")

    # --- Test evaluation (using val set since only 3 sessions) ---
    checkpoint = torch.load('ecg_beat_detector_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch'] + 1}")

    all_metrics = []
    with torch.no_grad():
        for batch in val_loader:
            signal = batch['signal'].to(device)
            beat_prob, class_logits = model(signal)
            for i in range(signal.size(0)):
                pred = detect_beats(beat_prob[i], class_logits[i])
                true = batch['beats'][i]
                metrics = evaluate_detection(pred, true)
                all_metrics.append(metrics)

    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    test_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    test_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0
    timing_maes = [m['timing_mae_ms'] for m in all_metrics if m['timing_mae_ms'] > 0]
    timing_mae = np.mean(timing_maes) if timing_maes else 0

    print(f"\n{'='*50}")
    print(f"TEST RESULTS ({len(all_metrics)} segments)")
    print(f"{'='*50}")
    print(f"Beat Detection:  P={test_prec:.4f}  R={test_rec:.4f}  F1={test_f1:.4f}")
    print(f"Timing MAE:      {timing_mae:.1f} ms")
    print()

    for cname in ['Normal', 'PVC', 'Other']:
        c_tp = sum(m['per_class'].get(cname, {}).get('tp', 0) for m in all_metrics)
        c_fp = sum(m['per_class'].get(cname, {}).get('fp', 0) for m in all_metrics)
        c_fn = sum(m['per_class'].get(cname, {}).get('fn', 0) for m in all_metrics)
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0
        print(f"  {cname:8s}: P={c_prec:.4f}  R={c_rec:.4f}  F1={c_f1:.4f}  (TP={c_tp} FP={c_fp} FN={c_fn})")

    # Save final
    split_info = {
        'train_sessions': train_sessions,
        'val_sessions': [val_session],
        'train_segments': len(train_idx),
        'val_segments': len(val_idx),
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'd_model': 128, 'nhead': 8, 'dim_ff': 256,
            'num_layers': 4, 'num_classes': NUM_CLASSES, 'dropout': 0.1,
        },
        'class_names': CLASS_NAMES,
        'class_weights': class_weights.cpu(),
        'split_info': split_info,
        'history': history,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
    }, 'ecg_beat_detector_final.pt')
    print(f'\nSaved ecg_beat_detector_final.pt')

if __name__ == '__main__':
    main()
