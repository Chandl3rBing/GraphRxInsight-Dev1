import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

print("Loading dataset...")

X = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")
y = np.load("DATASETS/processed/y_hard_chunk_0.npy")

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Loading models...")

rf = joblib.load("models/rf_final.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

print("Predicting...")

rf_preds = rf.predict_proba(X_test)[:,1]
xgb_preds = xgb_model.predict_proba(X_test)[:,1]

# Weighted average (tune this)
final_preds = (0.7 * rf_preds + 0.3 * xgb_preds)

final_preds = (final_preds > 0.5).astype(int)

acc = accuracy_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

print("\nENSEMBLE Accuracy:", acc)
print("ENSEMBLE F1:", f1)