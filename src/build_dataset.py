# import os
# import glob
# import numpy as np
# import pandas as pd
# from scipy.signal import welch
# from sklearn.preprocessing import StandardScaler

# # -------------------------------------------------------------------------
# # CONFIG
# # -------------------------------------------------------------------------
# CHANNELS = ["Fp1","Fp2","F3","F4","Fz","Pz"]
# SAMPLE_RATE = 250
# WINDOW_SEC = 4
# STEP_SEC = 1

# BANDS = {
#     "delta": (0.5, 4),
#     "theta": (4, 8),
#     "alpha": (8, 13),
#     "beta":  (13, 30),
#     "gamma": (30, 45)
# }

# LABEL_MAP = {
#     "happy": 0,
#     "stressed": 1,
#     "calm": 2,
#     "focused": 3

# }

# # -------------------------------------------------------------------------
# # HELPERS
# # -------------------------------------------------------------------------
# def load_neuphony_csv(path):
#     with open(path, "r", errors="ignore") as f:
#         lines = f.readlines()

#     header_i = None
#     for i, line in enumerate(lines[:50]):
#         if line.startswith("pkt_num"):
#             header_i = i
#             break

#     if header_i is None:
#         raise ValueError("Header not found")

#     df = pd.read_csv(path, header=header_i)

#     # Ensure channels exist
#     for ch in CHANNELS:
#         if ch not in df.columns:
#             raise ValueError(f"Missing channel {ch}")

#     arr = df[CHANNELS].to_numpy().T
#     return arr


# def bandpower(x, fs, low, high):
#     f, Pxx = welch(x, fs=fs, nperseg=256)
#     mask = (f >= low) & (f <= high)
#     return np.trapz(Pxx[mask], f[mask])


# def infer_label(filename):
#     name = filename.lower()
#     for key, val in LABEL_MAP.items():
#         if key in name:
#             return val
#     raise ValueError("No label found for file:", filename)


# # -------------------------------------------------------------------------
# # FEATURE EXTRACTION
# # -------------------------------------------------------------------------
# def extract_features(arr):
#     win = int(WINDOW_SEC * SAMPLE_RATE)
#     step = int(STEP_SEC * SAMPLE_RATE)

#     feats = []

#     for start in range(0, arr.shape[1] - win, step):
#         w = arr[:, start:start+win]      # shape (6, 500)

#         bp_all = []
#         for ch in range(6):
#             for (low, high) in BANDS.values():
#                 bp_all.append(bandpower(w[ch], SAMPLE_RATE, low, high))

#         feats.append(bp_all)

#     return np.array(feats)  # shape = (#windows, 30)


# # -------------------------------------------------------------------------
# # MAIN: BUILD DATASET
# # -------------------------------------------------------------------------
# def build_dataset():
#     RAW_DIR = "../data/raw"
#     SAVE_DIR = "../data"

#     X_all = []
#     y_all = []

#     files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

#     for file in files:
#         try:
#             arr = load_neuphony_csv(file)
#         except Exception as e:
#             print("Error loading", file, ":", e)
#             continue

#         X = extract_features(arr)
#         label = infer_label(os.path.basename(file))
#         y = np.full((X.shape[0],), label)

#         X_all.append(X)
#         y_all.append(y)

#         print(f"{os.path.basename(file)} -> windows={X.shape[0]} label={label}")

#     X_all = np.vstack(X_all)
#     y_all = np.concatenate(y_all)

#     print("\nFINAL DATASET:")
#     print("X shape:", X_all.shape)
#     print("y shape:", y_all.shape)
#     print("Label counts:", np.unique(y_all, return_counts=True))

#     np.save(os.path.join(SAVE_DIR, "features.npy"), X_all)
#     np.save(os.path.join(SAVE_DIR, "labels.npy"), y_all)


# if __name__ == "__main__":
#     build_dataset()


import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------------------------------------------------
# CONFIG
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



LABEL_MAP = {
    "happy": 0,
    "stressed": 1,
    "calm": 2,
    "focused": 3

}

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def load_neuphony_csv(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    header_i = None
    for i, line in enumerate(lines[:50]):
        if line.startswith("pkt_num"):
            header_i = i
            break

    if header_i is None:
        raise ValueError("Header not found")

    df = pd.read_csv(path, header=header_i)

    # Ensure channels exist
    for ch in CHANNELS:
        if ch not in df.columns:
            raise ValueError(f"Missing channel {ch}")

    arr = df[CHANNELS].to_numpy().T
    return arr


def bandpower(x, fs, low, high):
    # use nperseg no larger than signal length
    nperseg = min(256, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return np.trapz(Pxx[mask], f[mask])


def infer_label(filename):
    name = filename.lower()
    for key, val in LABEL_MAP.items():
        if key in name:
            return val
    raise ValueError(("No label found for file:", filename))


# -------------------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------------------
def extract_features(arr):
    """
    Input: arr shape (channels, samples) where channels == 6
    Output: features per window (30 bandpower + 12 ratios) => 42 features
    """
    win = int(WINDOW_SEC * SAMPLE_RATE)
    step = int(STEP_SEC * SAMPLE_RATE)

    feats = []

    n_bands = len(BANDS)
    band_list = list(BANDS.values())

    for start in range(0, arr.shape[1] - win + 1, step):
        w = arr[:, start:start+win]      # shape (6, win_samples)

        bp_all = []
        # compute bandpowers per channel
        for ch in range(len(CHANNELS)):
            ch_powers = []
            for (low, high) in band_list:
                p = bandpower(w[ch], SAMPLE_RATE, low, high)
                ch_powers.append(p)
                bp_all.append(p)

        # now add the two ratio features per channel:
        # alpha/beta and theta/alpha (avoid div by zero)
        ratio_features = []
        for ch_idx in range(len(CHANNELS)):
            base = ch_idx * n_bands
            delta = bp_all[base + 0]
            theta = bp_all[base + 1]
            alpha = bp_all[base + 2]
            beta = bp_all[base + 3]
            gamma = bp_all[base + 4]

            # alpha / beta
            if beta <= 0:
                ab = 0.0
            else:
                ab = alpha / beta

            # theta / alpha
            if alpha <= 0:
                ta = 0.0
            else:
                ta = theta / alpha

            ratio_features.append(ab)
            ratio_features.append(ta)

        # final feature vector = [30 bandpowers, 12 ratios] -> 42
        feature_vector = bp_all + ratio_features
        feats.append(feature_vector)

    return np.array(feats)


# -------------------------------------------------------------------------
# MAIN: BUILD DATASET
# -------------------------------------------------------------------------

def build_dataset():
    RAW_DIR = os.path.join("..", "data", "raw")
    SAVE_DIR = os.path.join("..", "data")
    MODEL_DIR = os.path.join("..", "models")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_all = []
    y_all = []

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    if len(files) == 0:
        print("No raw CSV files found in:", RAW_DIR)
        return

    for file in files:
        try:
            arr = load_neuphony_csv(file)
        except Exception as e:
            print("Error loading", file, ":", e)
            continue

        X = extract_features(arr)
        if X.size == 0:
            print("No windows extracted for", file)
            continue

        try:
            label = infer_label(os.path.basename(file))
        except Exception as e:
            print("Skipping file (no label):", file, e)
            continue

        y = np.full((X.shape[0],), label)

        X_all.append(X)
        y_all.append(y)

        print(f"{os.path.basename(file)} -> windows={X.shape[0]} label={label}")

    if len(X_all) == 0:
        print("No data collected. Exiting.")
        return

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print("\nFINAL DATASET:")
    print("X shape:", X_all.shape)   # expected (N_windows, 42)
    print("y shape:", y_all.shape)
    print("Label counts:", np.unique(y_all, return_counts=True))

    # scale features and save scaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_all)
    # joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))
    # print("Saved scaler to:", os.path.join(MODEL_DIR, "scaler.save"))

    # np.save(os.path.join(SAVE_DIR, "features.npy"), X_scaled)
    # np.save(os.path.join(SAVE_DIR, "labels.npy"), y_all)

    # fit scaler on raw features but DO NOT APPLY here
    scaler = StandardScaler()
    scaler.fit(X_all)  # Only fit — do not transform
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))
    print("Saved scaler to:", os.path.join(MODEL_DIR, "scaler.save"))

    # save ONLY RAW features
    np.save(os.path.join(SAVE_DIR, "features.npy"), X_all)
    np.save(os.path.join(SAVE_DIR, "labels.npy"), y_all)


    print("Saved features.npy and labels.npy in:", SAVE_DIR)


if __name__ == "__main__":
    build_dataset()
