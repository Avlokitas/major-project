import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load raw features
X = np.load("../data/features.npy")   # RAW features (42 features)
y = np.load("../data/labels.npy")

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# Load scaler (already fitted in build_dataset.py)
scaler = joblib.load("../models/scaler.save")

# Apply scaling
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Build SVM model
svm = SVC(
    kernel="rbf",
    C=5,
    gamma="scale",
    probability=True
)

# Train
print("\nTraining SVM...")
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Accuracy
print("\n🎯 Test Accuracy:", accuracy_score(y_test, y_pred))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm, "../models/svm_model.pkl")
print("\nSVM model saved at ../models/svm_model.pkl")
