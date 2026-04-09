import pandas as pd
import re
import os
from glob import glob


def extract_ddi_pairs(raw_folder="DATASETS/raw", output_file="DATASETS/processed/ddi_pairs.csv"):
    all_pairs = []

    # search only files that contain "drugbank" in filename
    files = glob(os.path.join(raw_folder, "*drugbank*.csv")) + glob(os.path.join(raw_folder, "*drugbank*.tsv"))

    if len(files) == 0:
        print("❌ No DrugBank dataset found inside:", raw_folder)
        return

    for file in files:
        print(f"\n📂 Using DrugBank file: {file}")

        if file.endswith(".tsv"):
            df = pd.read_csv(file, sep="\t", low_memory=False)
        else:
            df = pd.read_csv(file, low_memory=False)

        for _, row in df.iterrows():
            drug1 = row.get("drugbank-id", None)
            interactions = row.get("drug-interactions", "")

            if drug1 is None or pd.isna(interactions):
                continue

            found_ids = re.findall(r"DB\d{5}", str(interactions))

            for drug2 in found_ids:
                if drug1 != drug2:
                    all_pairs.append([drug1, drug2, 1])

    ddi_df = pd.DataFrame(all_pairs, columns=["drug1_id", "drug2_id", "label"])
    ddi_df = ddi_df.drop_duplicates()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ddi_df.to_csv(output_file, index=False)

    print("\n✅ DDI extraction completed!")
    print("Total DDI pairs:", len(ddi_df))
    print("Saved at:", output_file)


if __name__ == "__main__":
    extract_ddi_pairs()