import os
import pickle
import numpy as np
from scipy.signal import welch, butter, filtfilt
from joblib import load

# ---------------------------
# 1. LOAD TRAINED DEAP MODEL
# ---------------------------
MODEL_PATH = "deap_svm_4class.joblib"
clf = load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# ---------------------------
# 2. SETTINGS
# ---------------------------
FS_DEAP = 128        # DEAP sampling rate

EMOTION_NAMES = {
    0: "happy",
    1: "stressed",
    2: "focused",
    3: "calm",
}

# DEAP CHANNEL IDX FOR 6 CHANNELS
CHANNEL_IDX = [0, 16, 2, 19, 18, 15]  
# Fp1, Fp2, F3, F4, Fz, Pz


# ---------------------------
# 3. FILTER + FEATURE FUNCS (same as training)
# ---------------------------

def bandpass(sig, low=4, high=45, fs=FS_DEAP, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, sig)

def psd_features(trial_6ch):
    """
    trial_6ch: (6, N) at 128 Hz
    returns 24-dim feature vector (6 channels * 4 bands)
    """
    bands = [(4,8), (8,13), (13,30), (30,45)]
    feats = []

    for ch in range(6):
        sig = trial_6ch[ch]

        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std > 1e-8:
            sig = sig / std

        sig = bandpass(sig)
        f, Pxx = welch(sig, fs=FS_DEAP, nperseg=256)

        for lo, hi in bands:
            mask = (f >= lo) & (f <= hi)
            feats.append(Pxx[mask].mean())

    return np.array(feats, dtype=float)



# ---------------------------
# 4. LOAD ONE DEAP .DAT TRIAL
# ---------------------------

def load_deap_trial(dat_path, trial_index=0):
    """
    Load a single DEAP trial and return its 6-channel EEG.
    """
    with open(dat_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    data = d["data"]              # shape: (40, 40, 8064)
    eeg_32 = data[trial_index, :32, :]      # first 32 EEG channels
    eeg_6  = eeg_32[CHANNEL_IDX, :]         # pick required 6 channels

    print(f"Loaded DEAP trial {trial_index} from {dat_path} with shape {eeg_6.shape}")
    return eeg_6



# ---------------------------
# 5. PREDICT EMOTION
# ---------------------------

def predict_emotion_deap(dat_path, trial_idx=0):

    eeg_6 = load_deap_trial(dat_path, trial_idx)   # (6, 8064)

    feat = psd_features(eeg_6).reshape(1, -1)  # (1, 24)

    pred = clf.predict(feat)[0]
    probs = clf.predict_proba(feat)[0]

    emotion = EMOTION_NAMES.get(pred, "unknown")

    print(f"\nPrediction for DEAP file: {dat_path}, trial {trial_idx}")
    print(f"Emotion: {emotion}  (class {pred})")
    #print("Probabilities [happy, stressed, focused, calm]:")
    #print(probs)

    return pred, emotion, probs



# ---------------------------
# 6. INTERACTIVE USAGE
# ---------------------------

if __name__ == "__main__":
    # Ask user for DEAP .dat path
    dat_path = input("Enter full path to DEAP .dat file: ").strip()

    if not dat_path:
        # fallback default if user just presses Enter
        dat_path = "/Users/avlokita/Desktop/MAJOR 1/data_preprocessed_python/s01.dat"

    if not os.path.isfile(dat_path):
        print(f"❌ File not found: {dat_path}")
        exit(1)

    # Ask user for trial index
    trial_str = input("Enter trial index (0–39) [default 0]: ").strip()
    if trial_str == "":
        trial_idx = 0
    else:
        try:
            trial_idx = int(trial_str)
        except ValueError:
            print("❌ Invalid trial index, must be an integer.")
            exit(1)

    if not (0 <= trial_idx <= 39):
        print("❌ Trial index must be between 0 and 39.")
        exit(1)

    # Run prediction
    predict_emotion_deap(dat_path, trial_idx=trial_idx)
