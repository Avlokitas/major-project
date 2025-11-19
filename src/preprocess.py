import numpy as np
import pandas as pd

def load_neuphony_csv(path):
    df = pd.read_csv(path, skiprows=10)   # skip metadata
    channels = ["Fp1", "Fp2", "F3", "F4", "Fz", "Pz"]
    data = df[channels].values.T          # shape (6, samples)
    return data
