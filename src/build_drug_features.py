import pandas as pd
import os


def build_drug_features(
    drugbank_file="DATASETS/raw/drugbank_clean.csv",
    output_file="DATASETS/processed/drug_features.csv"
):
    print("📂 Loading DrugBank dataset:", drugbank_file)

    df = pd.read_csv(drugbank_file, low_memory=False)

    # Required numeric feature columns
    feature_cols = ["average-mass", "monoisotopic-mass"]

    # Convert to numeric safely
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with mean
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # Extract drug_id + numeric features
    drug_features = df[["drugbank-id"] + feature_cols].copy()
    drug_features.columns = ["drug_id", "avg_mass", "mono_mass"]

    # Save file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    drug_features.to_csv(output_file, index=False)

    print("✅ Drug features created successfully!")
    print("Total drugs:", len(drug_features))
    print("Saved at:", output_file)


if __name__ == "__main__":
    build_drug_features()