import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_morgan_fp(smiles, n_bits=1024, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)

    for i in range(n_bits):
        arr[i] = fp[i]

    return arr


def build_chemical_features(
    lipinski_file="DATASETS/external/DB_compounds_lipinski.csv",
    output_file="DATASETS/processed/chemical_features.csv",
    n_bits=1024
):
    print("📂 Loading Lipinski + SMILES dataset:", lipinski_file)
    df = pd.read_csv(lipinski_file)

    # Rename ID column to drug_id
    df = df.rename(columns={"ID": "drug_id"})

    required_cols = ["drug_id", "SMILES", "molecular_weight", "logp", "n_hba", "n_hbd", "ro5_fulfilled"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column in dataset: {col}")

    df = df[required_cols].copy()

    print("🔄 Generating Morgan fingerprints... (this may take some minutes)")

    fp_list = []
    valid_rows = []

    for i, row in df.iterrows():
        smiles = str(row["SMILES"])

        fp = smiles_to_morgan_fp(smiles, n_bits=n_bits)

        if fp is None:
            continue

        fp_list.append(fp)
        valid_rows.append(row)

        if (len(fp_list) % 200) == 0:
            print(f"✅ Processed {len(fp_list)} molecules...")

    valid_df = pd.DataFrame(valid_rows)

    fp_array = np.array(fp_list)

    # Combine lipinski numeric + fingerprint
    lipinski_numeric = valid_df[["molecular_weight", "logp", "n_hba", "n_hbd", "ro5_fulfilled"]].values

    final_features = np.concatenate([lipinski_numeric, fp_array], axis=1)

    # Create column names
    fp_cols = [f"fp_{i}" for i in range(n_bits)]
    final_cols = ["molecular_weight", "logp", "n_hba", "n_hbd", "ro5_fulfilled"] + fp_cols

    output_df = pd.DataFrame(final_features, columns=final_cols)
    output_df.insert(0, "drug_id", valid_df["drug_id"].values)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)

    print("✅ Chemical features created successfully!")
    print("Total drugs with fingerprints:", len(output_df))
    print("Feature dimension:", output_df.shape[1] - 1)
    print("Saved at:", output_file)


if __name__ == "__main__":
    build_chemical_features()