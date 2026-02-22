# Filters

from scipy.signal import butter, sosfiltfilt, medfilt

# Precompute filter coefficients once at import time (125 Hz device, fixed cutoffs).
# Using SOS (second-order sections) + sosfiltfilt is more numerically stable
# than the ba-form filtfilt and avoids recomputing coefficients on every call.
_FS = 125
_SOS = butter(5, [0.5 / (_FS / 2), 40.0 / (_FS / 2)], btype='band', output='sos')
_BASELINE_KERNEL = 63  # int(_FS * 0.5) = 62, rounded up to nearest odd


def filter_signal(signal, fs):
    filtered = sosfiltfilt(_SOS, signal)
    baseline = medfilt(filtered, kernel_size=_BASELINE_KERNEL)
    filtered -= baseline
    return filtered

