import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, resample
from joblib import load

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "/Users/avlokita/Desktop/MAJOR PROJECT/neuphony_random_forest_4class.joblib"

FS_NEUPHONY = 250.0
FS_FEATURES = 128.0

LABEL_MAP = {
    "happy":   0,
    "stressed":1,
    "focused": 2,
    "calm":    3,
}
EMOTION_NAMES = {v: k for k, v in LABEL_MAP.items()}


# ==========================
# SIGNAL & FEATURES (same as training)
# ==========================

def bandpass(sig, low=4, high=45, fs=FS_FEATURES, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, sig)


def resample_to_fs(eeg_6xN, orig_fs=FS_NEUPHONY, target_fs=FS_FEATURES):
    N = eeg_6xN.shape[1]
    new_N = int(N * target_fs / orig_fs)
    eeg_res = resample(eeg_6xN, new_N, axis=1)
    return eeg_res


def psd_features(trial_6ch):
    bands = [(4, 8), (8, 13), (13, 30), (30, 45)]
    feats = []

    for ch in range(trial_6ch.shape[0]):
        sig = trial_6ch[ch]
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std > 1e-8:
            sig = sig / std

        sig_filt = bandpass(sig, fs=FS_FEATURES)
        f, Pxx = welch(sig_filt, fs=FS_FEATURES, nperseg=256)

        for lo, hi in bands:
            mask = (f >= lo) & (f <= hi)
            feats.append(Pxx[mask].mean())

    return np.array(feats, dtype=float)


def load_neuphony_csv(path):
    df = pd.read_csv(path, skiprows=10)
    needed_cols = ["Fp1", "Fp2", "F3", "F4", "Fz", "Pz"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Column {c} not found in {path}")
    eeg = df[needed_cols].to_numpy().T  # (6, N)
    return eeg


# ==========================
# MAIN: SINGLE-FILE TEST
# ==========================

if __name__ == "__main__":
    # 1) Load model
    clf = load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    # 2) Ask user for a single CSV path
    csv_path = input("Enter full path to Neuphony CSV file: ").strip()

    if not os.path.isfile(csv_path):
        print(f"❌ File not found: {csv_path}")
        exit(1)

    print(f"\nUsing file: {csv_path}")

    # 3) Load EEG data from this file
    eeg_raw = load_neuphony_csv(csv_path)
    eeg_res = resample_to_fs(eeg_raw)

    # 4) Extract features
    feat = psd_features(eeg_res).reshape(1, -1)  # (1, 24)

    # 5) Predict emotion
    pred = clf.predict(feat)[0]
    emotion = EMOTION_NAMES.get(pred, "unknown")

    print(f"\nPrediction for {os.path.basename(csv_path)}:")
    print(f"  Predicted class id: {pred}")
    print(f"  Predicted emotion:  {emotion}")
