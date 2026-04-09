import pandas as pd
import numpy as np
import os


def build_atc_features(drugbank_file="DATASETS/raw/drugbank_clean.csv",
                       output_file="DATASETS/processed/atc_features.csv"):

    df = pd.read_csv(drugbank_file, low_memory=False)

    # Keep only drugbank-id and atc-codes
    df = df[["drugbank-id", "atc-codes"]].copy()
    df.columns = ["drug_id", "atc_codes"]

    # Fill NaN
    df["atc_codes"] = df["atc_codes"].fillna("")

    # Extract all unique ATC codes
    all_codes = set()
    for codes in df["atc_codes"]:
        for c in str(codes).split(";"):
            c = c.strip()
            if c != "":
                all_codes.add(c)

    all_codes = sorted(list(all_codes))
    print("Total unique ATC codes:", len(all_codes))

    code_to_idx = {code: i for i, code in enumerate(all_codes)}

    # Build multi-hot matrix
    atc_matrix = np.zeros((len(df), len(all_codes)), dtype=np.int8)

    for i, codes in enumerate(df["atc_codes"]):
        for c in str(codes).split(";"):
            c = c.strip()
            if c in code_to_idx:
                atc_matrix[i, code_to_idx[c]] = 1

    # Save features
    atc_df = pd.DataFrame(atc_matrix, columns=[f"ATC_{c}" for c in all_codes])
    atc_df.insert(0, "drug_id", df["drug_id"])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    atc_df.to_csv(output_file, index=False)

    print("✅ ATC feature file created!")
    print("Saved at:", output_file)


if __name__ == "__main__":
    build_atc_features()