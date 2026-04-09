import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

print("Loading dataset...")

X = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")
y = np.load("DATASETS/processed/y_hard_chunk_0.npy")

print("Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=800,        # ↑ more trees
    max_depth=50,            # ↑ deeper learning
    min_samples_leaf=1,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)

print("Training final RF...")

rf.fit(X_train, y_train)

preds = rf.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("\nFINAL Accuracy:", acc)
print("FINAL F1:", f1)

joblib.dump(rf, "models/rf_final.pkl")

print("Final model saved")