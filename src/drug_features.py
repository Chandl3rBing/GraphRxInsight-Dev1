import pandas as pd
import numpy as np
import os


def build_drug_features(drugbank_file="DATASETS/raw/drugbank_clean.csv",
                        output_file="DATASETS/processed/drug_features.csv"):

    df = pd.read_csv(drugbank_file, low_memory=False)

    # Select columns for features
    feature_cols = ["average-mass", "monoisotopic-mass"]

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    drug_features = df[["drugbank-id"] + feature_cols].copy()
    drug_features.columns = ["drug_id", "avg_mass", "mono_mass"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    drug_features.to_csv(output_file, index=False)

    print("✅ Drug features created successfully!")
    print("Total drugs:", len(drug_features))
    print("Saved at:", output_file)


if __name__ == "__main__":
    build_drug_features()