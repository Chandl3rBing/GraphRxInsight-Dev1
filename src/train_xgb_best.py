import numpy as np
import xgboost as xgb
import joblib

print("Training Optimized XGBoost Model")


# Load chunks
print("Loading chunks...")

X1 = np.load("DATASETS/processed/X_chunk_3_scaled.npy")
y1 = np.load("DATASETS/processed/y_chunk_3.npy")

X2 = np.load("DATASETS/processed/X_chunk_4_scaled.npy")
y2 = np.load("DATASETS/processed/y_chunk_4.npy")

X3 = np.load("DATASETS/processed/X_chunk_5_scaled.npy")
y3 = np.load("DATASETS/processed/y_chunk_5.npy")


X = np.vstack([X1,X2,X3])
y = np.concatenate([y1,y2,y3])


print("Dataset Shape:",X.shape)


model = xgb.XGBClassifier(

n_estimators=400,
max_depth=8,

learning_rate=0.05,

subsample=0.8,
colsample_bytree=0.8,

tree_method="hist",

eval_metric="logloss",

n_jobs=-1

)


print("\nTraining Started\n")

model.fit(X,y)


print("\nSaving Model")

joblib.dump(
model,
"models/xgb_best.pkl"
)

print("\nTraining Finished")