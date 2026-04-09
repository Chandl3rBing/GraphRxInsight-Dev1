import numpy as np

print("Loading datasets...")

# Hard dataset
X1 = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")
y1 = np.load("DATASETS/processed/y_hard_chunk_0.npy")

# Random dataset (use dynamic one)
X2 = np.load("DATASETS/processed/X_scaled.npy")
y2 = np.load("DATASETS/processed/y_dynamic.npy")

print("Hard:", X1.shape)
print("Random:", X2.shape)

# Balance both
size = min(len(X1), len(X2))

X_mix = np.vstack([X1[:size], X2[:size]])
y_mix = np.hstack([y1[:size], y2[:size]])

print("Mixed:", X_mix.shape)

np.save("DATASETS/processed/X_mixed.npy", X_mix)
np.save("DATASETS/processed/y_mixed.npy", y_mix)

print("Saved mixed dataset successfully")