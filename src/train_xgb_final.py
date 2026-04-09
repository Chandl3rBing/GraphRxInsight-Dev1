import numpy as np
import xgboost as xgb
import joblib


print("Loading dataset...")

X = np.load(
"DATASETS/processed/X_hard_chunk_0_scaled.npy"
)

y = np.load(
"DATASETS/processed/y_hard_chunk_0.npy"
)

print("Shape:",X.shape)


model = xgb.XGBClassifier(

n_estimators=300,
max_depth=6,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
tree_method="hist"

)


print("Training XGBoost...")

model.fit(X,y)


print("Saving model...")

joblib.dump(
model,
"models/xgb_model.pkl"
)

print("XGB Training Finished")