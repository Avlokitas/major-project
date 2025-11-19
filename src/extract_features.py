import numpy as np
from scipy.signal import welch

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

def bandpower(signal, fs=250):
    f, Pxx = welch(signal, fs, nperseg=256)
    powers = []
    for low, high in bands.values():
        idx = np.logical_and(f >= low, f <= high)
        powers.append(np.trapz(Pxx[idx], f[idx]))
    return powers

def extract_window_features(window):
    feats = []
    for ch in window:         # 6 channels
        feats.extend(bandpower(ch))
    return feats              # 40 features
