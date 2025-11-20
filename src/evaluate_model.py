import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Load data
X = np.load("../data/features.npy")
y = np.load("../data/labels.npy")

# Split same way as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load scaler
scaler = joblib.load("../models/scaler.save")
X_test = scaler.transform(X_test)

# Load trained model
model = load_model("../models/final_model.keras")

# Predict
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Accuracy
print("\n🎯 TEST ACCURACY:", accuracy_score(y_test, y_pred))

# Confusion matrix
print("\n📌 CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\n📌 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))
