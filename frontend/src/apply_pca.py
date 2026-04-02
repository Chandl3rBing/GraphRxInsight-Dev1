import numpy as np
import joblib
from sklearn.decomposition import PCA

print("Loading dataset...")

X = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")

print("Original shape:", X.shape)

pca = PCA(n_components=800)

print("Applying PCA...")

X_pca = pca.fit_transform(X)

print("New shape:", X_pca.shape)

np.save("DATASETS/processed/X_pca.npy", X_pca)

joblib.dump(pca, "models/pca.pkl")

print("PCA saved successfully")