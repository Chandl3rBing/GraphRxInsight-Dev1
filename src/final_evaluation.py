import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("\n==============================")
print(" FINAL MODEL EVALUATION")
print("==============================\n")


#################################################
# LOAD DATASET
#################################################

print("Loading Evaluation Dataset...\n")

X = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")
y = np.load("DATASETS/processed/y_hard_chunk_0.npy")

print("Dataset Shape:", X.shape)


#################################################
# LOAD RANDOM FOREST MODEL
#################################################

print("\nLoading Random Forest Model...")

try:
    rf_model = joblib.load("models/rf_optimized.pkl")
    print("Loaded models/rf_optimized.pkl")
except FileNotFoundError:
    rf_model = joblib.load("models/rf_model.pkl")
    print("Loaded fallback models/rf_model.pkl")


#################################################
# PREDICTIONS
#################################################

print("\nRunning Predictions...\n")

preds = rf_model.predict(X)


#################################################
# RESULTS
#################################################

acc = accuracy_score(y, preds)
f1 = f1_score(y, preds)


print("\n==============================")
print(" FINAL RESULTS")
print("==============================\n")


print("Accuracy:", acc)
print("F1 Score:", f1)

print("\nEvaluation Complete\n")