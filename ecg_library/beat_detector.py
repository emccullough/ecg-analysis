import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks


class CNNTransformerDetector(nn.Module):
    """CNN-Transformer model for ECG beat detection and classification.

    Architecture:
        CNN Feature Encoder (3 conv layers with stride-2 downsampling)
        → Transformer Context Encoder (4 layers)
        → Dual Detection Heads (beat probability + class logits)

    Input:  [batch, 1, 2500]  (20s filtered ECG at 125 Hz)
    Output: beat_prob [batch, 625], class_logits [batch, 625, 3]
    """

    def __init__(self, d_model=128, nhead=8, dim_ff=256, num_layers=4,
                 num_classes=3, dropout=0.1):
        super().__init__()

        # CNN Feature Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # Transformer Context Encoder
        self.pos_encoding = nn.Parameter(torch.randn(1, 625, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Detection Heads
        self.beat_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: [batch, 1, 2500] normalized ECG signal

        Returns:
            beat_prob: [batch, 625] beat probability (sigmoid)
            class_logits: [batch, 625, num_classes] class logits (raw)
        """
        # CNN encoder: [batch, 1, 2500] → [batch, 128, 625]
        h = self.conv1(x)     # [batch, 32, 2500]
        h = self.conv2(h)     # [batch, 64, 1250]
        h = self.conv3(h)     # [batch, 128, 625]

        # Reshape for transformer: [batch, 625, 128]
        h = h.permute(0, 2, 1)

        # Add positional encoding
        h = h + self.pos_encoding

        # Transformer: [batch, 625, 128]
        h = self.transformer(h)

        # Detection heads
        beat_prob = torch.sigmoid(self.beat_head(h).squeeze(-1))  # [batch, 625]
        class_logits = self.class_head(h)                          # [batch, 625, 3]

        return beat_prob, class_logits


def focal_loss(logits, targets, class_weights=None, gamma=2.0):
    """Focal loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples so the model focuses on hard cases (e.g. PVCs).
    """
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
    p_t = torch.exp(-ce)  # probability of correct class
    loss = ((1 - p_t) ** gamma) * ce
    return loss.mean()


def compute_loss(beat_prob, class_logits, beat_target, class_target,
                 pos_weight=10.0, class_weights=None, focal_gamma=2.0,
                 class_loss_weight=0.5):
    """Compute combined detection + classification loss.

    Args:
        beat_prob: [batch, 625] predicted beat probability
        class_logits: [batch, 625, num_classes] predicted class logits
        beat_target: [batch, 625] ground truth beat heatmap
        class_target: [batch, 625] ground truth class IDs (int64)
        pos_weight: weight for positive beat samples (beats are sparse)
        class_weights: [num_classes] tensor of class weights (inverse frequency)
        focal_gamma: focusing parameter for focal loss (0 = standard CE)
        class_loss_weight: multiplier for classification loss vs detection loss

    Returns:
        total_loss, loss_dict with individual components
    """
    # Beat detection loss: weighted BCE
    weight_map = torch.where(beat_target > 0.5, pos_weight, 1.0)
    loss_detect = F.binary_cross_entropy(beat_prob, beat_target, weight=weight_map)

    # Classification loss: focal loss only at beat positions
    beat_mask = beat_target > 0.5
    loss_class = torch.tensor(0.0, device=beat_prob.device)

    if beat_mask.any():
        masked_logits = class_logits[beat_mask]              # [N, num_classes]
        masked_targets = class_target[beat_mask]             # [N]
        loss_class = focal_loss(masked_logits, masked_targets,
                                class_weights=class_weights, gamma=focal_gamma)

    total_loss = loss_detect + class_loss_weight * loss_class

    return total_loss, {
        'total': total_loss.item(),
        'detect': loss_detect.item(),
        'class': loss_class.item(),
    }


def detect_beats(beat_prob, class_logits, threshold=0.5, min_distance=25):
    """Post-process model output to extract beat list.

    Args:
        beat_prob: [625] beat probability (numpy or tensor)
        class_logits: [625, num_classes] class logits (numpy or tensor)
        threshold: minimum beat probability
        min_distance: minimum distance between beats (in samples at 625 resolution)

    Returns:
        list of (sample_idx_2500, class_name, confidence) tuples
    """
    if isinstance(beat_prob, torch.Tensor):
        beat_prob = beat_prob.detach().cpu().numpy()
    if isinstance(class_logits, torch.Tensor):
        class_logits = class_logits.detach().cpu().numpy()

    # Upsample 625 → 2500 via linear interpolation
    x_ds = np.arange(625)
    x_full = np.linspace(0, 624, 2500)
    beat_prob_full = np.interp(x_full, x_ds, beat_prob)

    # Upsample class logits: [625, C] → [2500, C]
    num_classes = class_logits.shape[1]
    class_logits_full = np.zeros((2500, num_classes))
    for c in range(num_classes):
        class_logits_full[:, c] = np.interp(x_full, x_ds, class_logits[:, c])

    # Find peaks in upsampled beat probability
    min_dist_full = min_distance * 4  # convert from DS to full resolution
    peaks, properties = find_peaks(beat_prob_full, height=threshold, distance=min_dist_full)

    class_names = {0: 'Other', 1: 'Normal', 2: 'PVC'}
    results = []
    for peak in peaks:
        confidence = beat_prob_full[peak]
        class_id = int(np.argmax(class_logits_full[peak]))
        class_name = class_names.get(class_id, 'Other')
        results.append((int(peak), class_name, float(confidence)))

    return results


def evaluate_detection(pred_beats, true_beats, tolerance=5):
    """Evaluate beat detection accuracy.

    Args:
        pred_beats: list of (sample_idx, class_name, confidence) from detect_beats
        true_beats: list of (r_idx, class_id) ground truth
        tolerance: matching tolerance in samples at 2500 resolution

    Returns:
        dict with TP, FP, FN, precision, recall, F1, per-class metrics,
        and timing MAE
    """
    class_names = {0: 'Other', 1: 'Normal', 2: 'PVC'}

    pred_positions = np.array([p[0] for p in pred_beats]) if pred_beats else np.array([])
    pred_classes = {p[0]: p[1] for p in pred_beats}
    true_positions = np.array([t[0] for t in true_beats]) if true_beats else np.array([])
    true_classes = {t[0]: class_names.get(t[1], 'Other') for t in true_beats}

    matched_pred = set()
    matched_true = set()
    timing_errors = []

    # Match predictions to ground truth
    for i, t_pos in enumerate(true_positions):
        if len(pred_positions) == 0:
            break
        dists = np.abs(pred_positions - t_pos)
        best_j = np.argmin(dists)
        if dists[best_j] <= tolerance and best_j not in matched_pred:
            matched_pred.add(best_j)
            matched_true.add(i)
            timing_errors.append(abs(int(pred_positions[best_j]) - int(t_pos)))

    tp = len(matched_true)
    fp = len(pred_positions) - tp
    fn = len(true_positions) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    timing_mae = np.mean(timing_errors) / 125 * 1000 if timing_errors else 0.0  # in ms

    # Per-class metrics
    per_class = {}
    for cid, cname in class_names.items():
        c_true = [i for i, (pos, cls) in enumerate(true_beats) if cls == cid]
        c_pred = [j for j, (pos, cls, _) in enumerate(pred_beats) if cls == cname]

        c_tp = 0
        for i in c_true:
            t_pos = true_beats[i][0]
            for j in c_pred:
                p_pos = pred_beats[j][0]
                if abs(p_pos - t_pos) <= tolerance:
                    c_tp += 1
                    break

        c_fp = len(c_pred) - c_tp
        c_fn = len(c_true) - c_tp
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        per_class[cname] = {'precision': c_prec, 'recall': c_rec, 'f1': c_f1,
                            'tp': c_tp, 'fp': c_fp, 'fn': c_fn}

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'timing_mae_ms': timing_mae,
        'per_class': per_class,
    }
