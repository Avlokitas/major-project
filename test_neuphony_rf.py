import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load

# ==========================
# CONFIG
# ==========================

# Same dir as training:
DATA_DIR = "/Users/avlokita/Desktop/MAJOR PROJECT/neuphony_data"

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


def infer_label_from_filename(filename):
    fname = os.path.basename(filename).lower()
    for key in LABEL_MAP.keys():
        if fname.startswith(key):   # happy_*, stressed_*, etc.
            return LABEL_MAP[key]
    raise ValueError(f"Cannot infer label from filename: {filename}")


# ==========================
# BUILD TEST SET
# ==========================

def build_test_set(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    print("Found CSV files for testing:", len(files))

    X_list = []
    y_true = []
    file_names = []

    for fpath in files:
        try:
            label = infer_label_from_filename(fpath)
        except ValueError as e:
            print("Skipping:", fpath, "|", e)
            continue

        print("Preparing:", os.path.basename(fpath), "-> true label", EMOTION_NAMES[label])
        eeg_raw = load_neuphony_csv(fpath)
        eeg_res = resample_to_fs(eeg_raw)

        feat = psd_features(eeg_res)
        X_list.append(feat)
        y_true.append(label)
        file_names.append(fpath)

    X = np.vstack(X_list)
    y_true = np.array(y_true)

    print("\nTest X shape:", X.shape)
    print("Test y shape:", y_true.shape)
    print("Class counts in test set:", Counter(y_true))

    return X, y_true, file_names


# ==========================
# MAIN TESTING
# ==========================

if __name__ == "__main__":
    # 1) Load model
    clf = load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    # 2) Build test set from ALL CSVs in folder
    X_test, y_true, files = build_test_set(DATA_DIR)

    # 3) Predict
    y_pred = clf.predict(X_test)

    # 4) Per-file results
    print("\n===== PER-FILE PREDICTIONS =====")
    for fpath, yt, yp in zip(files, y_true, y_pred):
        print(f"{os.path.basename(fpath):20s}  true: {EMOTION_NAMES[yt]:8s}  pred: {EMOTION_NAMES[yp]:8s}")

    # 5) Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print("\n===== OVERALL TEST RESULTS =====")
    print("Accuracy on all files:", acc)

    # 6) Classification report (handle missing classes safely)
    present_labels = sorted(list(set(y_true)))
    present_names = [EMOTION_NAMES[i] for i in present_labels]

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names
    ))

    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=present_labels))
