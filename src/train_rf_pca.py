import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

print("Loading PCA dataset...")

X = np.load("DATASETS/processed/X_pca.npy")
y = np.load("DATASETS/processed/y_hard_chunk_0.npy")

print("Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=50,
    n_jobs=-1,
    random_state=42
)

print("Training...")

rf.fit(X_train, y_train)

preds = rf.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("\nPCA Accuracy:", acc)
print("PCA F1:", f1)

joblib.dump(rf, "models/rf_pca.pkl")

print("Model saved successfully")