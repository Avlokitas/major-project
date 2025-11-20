import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
X = np.load("../data/features.npy")     # (N, 30)
y = np.load("../data/labels.npy")       # (N,)

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# Split into train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



# Load scaler created during dataset building
scaler = joblib.load("../models/scaler.save")
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


# -------------------------
# BUILD THE MODEL (DNN)
# -------------------------
model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(4, activation="softmax")  # 4 emotions
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5)
]

# -------------------------
# TRAIN THE MODEL
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save("../models/final_model.keras")
print("\nModel saved at ../models/final_model.keras")
