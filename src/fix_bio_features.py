import pandas as pd

print("Loading bio features...")

bio = pd.read_csv(
    "DATASETS/processed/bio_features.csv",
    index_col=0
)

print("Before:", bio.shape)

bio = bio.groupby(bio.index).max()

print("After:", bio.shape)

bio.to_csv(
    "DATASETS/processed/bio_features_clean.csv"
)

print("Saved cleaned bio features")