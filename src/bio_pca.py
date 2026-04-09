import pandas as pd
from sklearn.decomposition import PCA

print("Loading biological features...")

bio = pd.read_csv(
    "DATASETS/processed/bio_features.csv"
)

print("Original shape:", bio.shape)


# Remove DrugBank ID column
if bio.columns[0].startswith("DB") or "drug" in bio.columns[0].lower():
    bio_ids = bio.iloc[:, 0]
    bio = bio.iloc[:, 1:]
else:
    bio_ids = None


print("Numeric shape:", bio.shape)


# PCA Reduction
pca = PCA(n_components=300)

bio_pca = pca.fit_transform(bio.values)


print("Reduced shape:", bio_pca.shape)


bio_pca_df = pd.DataFrame(bio_pca)

# Add Drug IDs back if available
if bio_ids is not None:
    bio_pca_df.insert(0, "drugbank_id", bio_ids)


bio_pca_df.to_csv(
    "DATASETS/processed/bio_pca_features.csv",
    index=False
)

print("Saved PCA biological features")