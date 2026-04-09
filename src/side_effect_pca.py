import pandas as pd
from sklearn.decomposition import PCA

print("Loading side effect features...")

df = pd.read_csv(
    "DATASETS/processed/side_effect_features.csv",
    index_col=0
)

print("Original shape:", df.shape)

# Reduce to 200 dimensions
pca = PCA(n_components=200)

reduced = pca.fit_transform(df)

print("Reduced shape:", reduced.shape)

pd.DataFrame(reduced).to_csv(
    "DATASETS/processed/side_effect_pca.csv",
    index=False
)

print("Saved PCA side effects")