import numpy as np
import joblib
from sklearn.metrics import accuracy_score,f1_score


print("Evaluating XGBoost Best Model")


model = joblib.load(
"models/xgb_best.pkl"
)


# Use existing chunk
X = np.load(
"DATASETS/processed/X_chunk_5_scaled.npy"
)

y = np.load(
"DATASETS/processed/y_chunk_5.npy"
)


pred = model.predict(X)


print("\nRESULTS")

print("Accuracy:",accuracy_score(y,pred))
print("F1 Score:",f1_score(y,pred))