import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


def apply_pca_to_atc(
    input_file="DATASETS/processed/atc_features.csv",
    output_file="DATASETS/processed/atc_pca_features.csv",
    n_components=50
):
    df = pd.read_csv(input_file)

    drug_ids = df["drug_id"].values
    X = df.drop(columns=["drug_id"]).values.astype(np.float32)

    print("Original ATC shape:", X.shape)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("PCA reduced shape:", X_pca.shape)
    print("Explained variance ratio sum:", pca.explained_variance_ratio_.sum())

    pca_df = pd.DataFrame(X_pca, columns=[f"ATC_PCA_{i+1}" for i in range(n_components)])
    pca_df.insert(0, "drug_id", drug_ids)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pca_df.to_csv(output_file, index=False)

    print("✅ PCA ATC features saved at:", output_file)


if __name__ == "__main__":
    apply_pca_to_atc()