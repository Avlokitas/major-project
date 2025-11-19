import os
import numpy as np
from preprocess import load_neuphony_csv
from extract_features import extract_window_features

DATA_DIR =  "../data/raw/"


WINDOW_SIZE = 250 * 2   # 2 sec
STEP = 250 * 1          # 50% overlap

def sliding_windows(data, ws, step):
    ch, N = data.shape
    windows = []
    for start in range(0, N - ws, step):
        win = data[:, start:start + ws]
        windows.append(win)
    return windows

# MODIFY LABELS HERE 
label_map = {
    "happy": 0,
    "stressed": 1,
    "calm": 2,
    "focused": 3
}

def build_dataset():
    X, y = [], []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            path = DATA_DIR + file

            # get label from filename (e.g., relaxed_01.csv)
            emotion = file.split("_")[0]
            label = label_map[emotion]

            data = load_neuphony_csv(path)
            windows = sliding_windows(data, WINDOW_SIZE, STEP)

            for w in windows:
                feats = extract_window_features(w)
                X.append(feats)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    np.save("../data/features.npy", X)
    np.save("../data/labels.npy", y)

    print("Dataset saved!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    build_dataset()
