import pandas as pd
import numpy as np

print("Loading SIDER dataset...")

sider = pd.read_csv(
    "DATASETS/raw/meddra_all_se.tsv",
    sep="\t",
    header=None
)

# Columns:
# 0 = STITCH flat
# 1 = STITCH stereo
# 2 = UMLS id
# 3 = MedDRA type
# 4 = MedDRA id
# 5 = Side effect name

drug_col = 1
side_effect_col = 5

sider = sider[[drug_col, side_effect_col]]
sider.columns = ["drug_id","side_effect"]

print("Unique drugs:", sider["drug_id"].nunique())
print("Unique side effects:", sider["side_effect"].nunique())

print("Building feature matrix...")

features = pd.crosstab(
    sider["drug_id"],
    sider["side_effect"]
)

print("Shape:", features.shape)

features.to_csv(
    "DATASETS/processed/side_effect_features.csv"
)

print("Saved side effect features")