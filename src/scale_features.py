import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading chunks...")

X1=np.load("DATASETS/processed/X_chunk_3.npy")
X2=np.load("DATASETS/processed/X_chunk_4.npy")
X3=np.load("DATASETS/processed/X_chunk_5.npy")

X_all=np.vstack([X1,X2,X3])

print("Total shape:",X_all.shape)


print("\nFitting scaler...")

scaler=StandardScaler()

scaler.fit(X_all)


joblib.dump(scaler,"models/scaler.pkl")

print("Scaler saved")


for i in [3,4,5]:

    print("Scaling chunk:",i)

    X=np.load(f"DATASETS/processed/X_chunk_{i}.npy")

    X_scaled=scaler.transform(X)

    np.save(f"DATASETS/processed/X_chunk_{i}_scaled.npy",X_scaled)


print("\nScaling finished")