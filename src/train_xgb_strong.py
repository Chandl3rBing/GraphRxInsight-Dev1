import numpy as np
import xgboost as xgb
import joblib

print("Loading dataset...")

X = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")
y = np.load("DATASETS/processed/y_hard_chunk_0.npy")

print("Shape:",X.shape)

model = xgb.XGBClassifier(

n_estimators=600,
max_depth=8,

learning_rate=0.03,

subsample=0.9,
colsample_bytree=0.9,

gamma=0.1,
min_child_weight=3,

tree_method="hist"

)

print("Training Strong XGB...")

model.fit(X,y)

joblib.dump(
model,
"models/xgb_strong.pkl"
)

print("Strong XGB Saved")