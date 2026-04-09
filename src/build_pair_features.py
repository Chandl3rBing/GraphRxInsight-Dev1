import pandas as pd
import numpy as np

print("Loading datasets...")

# Unified drug features
features = pd.read_csv(
    "DATASETS/processed/unified_drug_features.csv",
    index_col=0
)

print("Drug Features:", features.shape)

# Training dataset
ddi = pd.read_csv(
    "DATASETS/processed/final_ddi_dataset.csv"
)

print("DDI Dataset:", ddi.shape)

X_list = []
y_list = []

drug1_col = ddi.columns[0]
drug2_col = ddi.columns[1]
label_col = ddi.columns[2]

TOTAL_SAMPLES = len(ddi)

print("Building pair features...")

for i, row in ddi.iterrows():

    d1 = row[drug1_col]
    d2 = row[drug2_col]

    if d1 not in features.index:
        continue

    if d2 not in features.index:
        continue

    f1 = features.loc[d1].values.flatten()
    f2 = features.loc[d2].values.flatten()

    pair_feat = np.concatenate([f1, f2])

    X_list.append(pair_feat)
    y_list.append(row[label_col])

    if i % 100000 == 0:
        print("Processed:", i, "/", TOTAL_SAMPLES)


X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print("Final Feature Matrix:", X.shape)
print("Labels:", y.shape)

np.save("DATASETS/processed/X_pair.npy", X)
np.save("DATASETS/processed/y_pair.npy", y)

print("Saved pair features successfully.")
