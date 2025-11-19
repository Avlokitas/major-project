import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
)

# Load dataset
X = np.load("../data/features.npy")   # shape (N, 30)
y = np.load("../data/labels.npy")    # shape (N,)

# ---- Fix reshape ----
X = X.reshape((X.shape[0], 6, 5, 1))

# ---- FIXED: we force 4 classes ----
num_classes = 4

# CNN model
model = Sequential([
    InputLayer(input_shape=(6, 5, 1)),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')    # FIXED OUTPUT
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

model.fit(X, y, epochs=35, batch_size=32, validation_split=0.2)

model.save("../models/cnn_model.keras")
print("Model saved!")
