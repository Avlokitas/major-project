import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

# ==========================
# CONFIG
# ==========================

# Folder containing all your Neuphony CSVs
#   MAJOR Project/neuphony_data/
#       calm_001.csv
#       calm_002.csv
#       stressed_001.csv
#       focused_001.csv
#       happy_001.csv
DATA_DIR = "/Users/avlokita/Desktop/MAJOR Project/neuphony_data"

FS_NEUPHONY = 250.0   # from metadata
FS_FEATURES = 128.0   # we resample here for consistency

LABEL_MAP = {
    "happy":   0,
    "stressed":1,
    "focused": 2,
    "calm":    3,
}
EMOTION_NAMES = {v: k for k, v in LABEL_MAP.items()}


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
        if fname.startswith(key):
            return LABEL_MAP[key]
    raise ValueError(f"Cannot infer label from filename: {filename}")


def build_neuphony_dataset(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    print("Found CSV files:", len(files))

    X_list = []
    y_list = []

    for fpath in files:
        try:
            label = infer_label_from_filename(fpath)
        except ValueError as e:
            print("Skipping:", fpath, "|", e)
            continue

        print("Loading:", os.path.basename(fpath), "-> label", EMOTION_NAMES[label])
        eeg_raw = load_neuphony_csv(fpath)
        eeg_res = resample_to_fs(eeg_raw)

        feat = psd_features(eeg_res)
        X_list.append(feat)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list)
    print("X shape (trials, features):", X.shape)
    print("y shape (trials,):", y.shape)
    print("Class counts:", Counter(y))
    return X, y


def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample"
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n===== RANDOM FOREST RESULTS =====")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[EMOTION_NAMES[i] for i in sorted(LABEL_MAP.values())]
    ))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return rf


if __name__ == "__main__":
    X, y = build_neuphony_dataset(DATA_DIR)
    rf_model = train_random_forest(X, y)
    dump(rf_model, "neuphony_random_forest_4class.joblib")
    print("\nSaved model as neuphony_random_forest_4class.joblib")
