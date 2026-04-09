import numpy as np
import joblib
import xgboost as xgb

print("Loading dataset")

X=np.load("DATASETS/processed/X_chunk_3.npy")
y=np.load("DATASETS/processed/y_chunk_3.npy")

print("Shape:",X.shape)


print("\nTraining XGBoost")

model=xgb.XGBClassifier(

n_estimators=300,
max_depth=8,
learning_rate=0.1,
n_jobs=-1,
tree_method="hist"
)

model.fit(X,y)


joblib.dump(model,"models/xgb_model.pkl")

print("\nXGBoost Saved")