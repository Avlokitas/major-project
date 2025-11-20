import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.signal import welch
import joblib
import sys

# -------------------------------------------------------------------------
# CONFIG — MUST MATCH build_dataset.py
# -------------------------------------------------------------------------
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "Fz", "Pz"]
SAMPLE_RATE = 250
WINDOW_SEC = 4
STEP_SEC = 1

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

EMOTION_MAP = {
    0: "happy",
    1: "stressed",
    2: "calm",
    3: "focused"
}

# -------------------------------------------------------------------------
# LOAD CSV
# -------------------------------------------------------------------------
def load_eeg_csv(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    # Find header
    header_i = 0
    for i, line in enumerate(lines[:40]):
        if line.startswith("pkt_num"):
            header_i = i
            break

    df = pd.read_csv(path, header=header_i)
    return df[CHANNELS].to_numpy().T


# -------------------------------------------------------------------------
# BANDPOWER
# -------------------------------------------------------------------------
def bandpower(x, fs, low, high):
    nperseg = min(256, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return np.trapz(Pxx[mask], f[mask])


# -------------------------------------------------------------------------
# FEATURE EXTRACTION  (42 FEATURES = 30 BANDPOWER + 12 RATIOS)
# -------------------------------------------------------------------------
def extract_features(arr):
    win = int(WINDOW_SEC * SAMPLE_RATE)
    step = int(STEP_SEC * SAMPLE_RATE)

    feats = []
    band_list = list(BANDS.values())
    n_bands = len(band_list)

    for start in range(0, arr.shape[1] - win + 1, step):
        w = arr[:, start:start + win]

        bp_all = []

        # 30 bandpowers (6 channels × 5 bands)
        for ch in range(len(CHANNELS)):
            for (low, high) in band_list:
                p = bandpower(w[ch], SAMPLE_RATE, low, high)
                bp_all.append(p)

        # 12 ratio features (alpha/beta and theta/alpha)
        ratio_features = []
        for ch in range(len(CHANNELS)):
            base = ch * n_bands
            delta = bp_all[base + 0]
            theta = bp_all[base + 1]
            alpha = bp_all[base + 2]
            beta = bp_all[base + 3]
            gamma = bp_all[base + 4]

            # alpha / beta
            ab = (alpha / beta) if beta > 0 else 0.0
            # theta / alpha
            ta = (theta / alpha) if alpha > 0 else 0.0

            ratio_features.append(ab)
            ratio_features.append(ta)

        final_feat = bp_all + ratio_features   # 42 features
        feats.append(final_feat)

    return np.array(feats)


# -------------------------------------------------------------------------
# MAIN PREDICTION PIPELINE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = sys.argv[1]
    print("Reading CSV:", csv_path)

    arr = load_eeg_csv(csv_path)

    feats = extract_features(arr)
    print("Extracted feature shape:", feats.shape)  # MUST be (N_windows, 42)

    # Load scaler
    scaler = joblib.load("models/scaler.save")
    feats = scaler.transform(feats)

    # Load trained DNN
    model = load_model("models/final_model.keras")

    # Predict per window
    preds = model.predict(feats)
    avg = np.mean(preds, axis=0)

    result = np.argmax(avg)

    print("\nPredicted Emotion:", EMOTION_MAP[result])
    print("Probabilities:", avg)
