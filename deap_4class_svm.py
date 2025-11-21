import glob
import os
import pickle
import numpy as np














































from scipy.signal import welch, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from joblib import dump

# ==============================
# 1. CONFIG
# ==============================

# CHANGE this if your .dat files are somewhere else
DATA_PATH = "/Users/avlokita/Desktop/MAJOR 1/data_preprocessed_python/s*.dat"

# 6 EEG channels (DEAP indices):
# Fp1 (0), Fp2 (16), F3 (2), F4 (19), Fz (18), Pz (15)
CHANNEL_IDX = [0, 16, 2, 19, 18, 15]

FS = 128  # DEAP sampling rate

# label mapping:
# 0 = happy
# 1 = stressed
# 2 = focused
# 3 = calm
EMOTION_NAMES = {
    0: "happy",
    1: "stressed",
    2: "focused",
    3: "calm",
}


# ==============================
# 2. LOAD DEAP & KEEP 6 CHANNELS
# ==============================

def load_deap_6ch(data_path_pattern):
    file_list = sorted(glob.glob(data_path_pattern))
    print("Found .dat files:", len(file_list))
    if len(file_list) == 0:
        raise FileNotFoundError("No .dat files found. Check DATA_PATH.")

    X_all = []
    y_all = []

    for fpath in file_list:
        print("Loading:", os.path.basename(fpath))
        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="latin1")

        trials = d["data"]      # (40, 40, 8064)
        labels = d["labels"]    # (40, 4) -> [valence, arousal, dominance, liking]

        eeg_32 = trials[:, :32, :]         # (40, 32, 8064)
        eeg_6  = eeg_32[:, CHANNEL_IDX, :] # (40, 6, 8064)

        X_all.append(eeg_6)
        y_all.append(labels)

    X = np.concatenate(X_all, axis=0)  # (N_trials, 6, 8064)
    y = np.concatenate(y_all, axis=0)  # (N_trials, 4)

    print("X shape (trials, 6ch, samples):", X.shape)
    print("y shape (trials, 4 labels):", y.shape)
    return X, y


# ==============================
# 3. MAP DEAP → 4 EMOTIONS
# ==============================

def map_deap_to_4classes(y):
    """
    y: (N, 4) array -> [valence, arousal, dominance, liking]
    returns: labels_4 (N,) with values {0,1,2,3} or -1 for 'ignore'
    """
    valence   = y[:, 0]
    arousal   = y[:, 1]
    dominance = y[:, 2]

    labels_4 = []

    for v, a, d in zip(valence, arousal, dominance):
        if v >= 6 and a >= 6:
            labels_4.append(0)  # happy
        elif v <= 4 and a >= 6:
            labels_4.append(1)  # stressed
        elif v >= 6 and a <= 4:
            labels_4.append(3)  # calm
        elif 4 <= v <= 6 and 5 <= a <= 7:
            labels_4.append(2)  # focused
        else:
            labels_4.append(-1) # ignore / doesn't fit nicely

    return np.array(labels_4, dtype=int)


# ==============================
# 4. FEATURE EXTRACTION (PSD) + NORMALIZATION
# ==============================

def bandpass(sig, low=4, high=45, fs=FS, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def psd_features(trial_6ch):
    """
    trial_6ch: (6, N) at 128 Hz
    returns 24-dim feature vector (6 channels * 4 bands)
    """
    bands = [(4, 8), (8, 13), (13, 30), (30, 45)]  # theta, alpha, beta, gamma
    feats = []

    for ch in range(trial_6ch.shape[0]):
        sig = trial_6ch[ch]

        # per-channel normalization (mean 0, std 1)
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std > 1e-8:
            sig = sig / std

        sig = bandpass(sig)
        f, Pxx = welch(sig, fs=FS, nperseg=256)

        for lo, hi in bands:
            mask = (f >= lo) & (f <= hi)
            feats.append(Pxx[mask].mean())

    return np.array(feats, dtype=float)

def build_feature_matrix(X):
    feat_list = []
    for i, trial in enumerate(X):
        if (i + 1) % 100 == 0:
            print(f"Extracting features: trial {i+1}/{len(X)}")
        feat_list.append(psd_features(trial))
    return np.vstack(feat_list)


# ==============================
# 5. TRAIN SVM (4-CLASS)
# ==============================

def train_svm_4class(X_feat, y_4):
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_4, test_size=0.2, random_state=42, stratify=y_4
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True  # for predict_proba later
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n===== RESULTS (4-class SVM) =====")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[EMOTION_NAMES[k] for k in sorted(set(y_4))]
    ))

    return clf


# ==============================
# 6. MAIN
# ==============================

if __name__ == "__main__":
    # 1) Load DEAP 6-channel
    X, y = load_deap_6ch(DATA_PATH)

    # 2) Map to 4-class labels
    labels_4 = map_deap_to_4classes(y)
    print("\nRaw 4-class label distribution (including -1):")
    print(Counter(labels_4))

    # 3) Filter out ignored trials (-1)
    mask = labels_4 != -1
    X_use = X[mask]
    y_use = labels_4[mask]

    print("\nAfter filtering:")
    print("X_use shape:", X_use.shape)
    print("y_use shape:", y_use.shape)
    print("Class counts:", Counter(y_use))
    print("Class names:", {k: EMOTION_NAMES[k] for k in sorted(Counter(y_use).keys())})

    # 4) Extract PSD features
    print("\nExtracting PSD features...")
    X_feat = build_feature_matrix(X_use)
    print("X_feat shape:", X_feat.shape)  # (N_trials_kept, 24)

    # 5) Train SVM
    clf = train_svm_4class(X_feat, y_use)

    # 6) Save model
    dump(clf, "deap_svm_4class.joblib")
    print("\nModel saved as deap_svm_4class.joblib")
    print("Done.")
