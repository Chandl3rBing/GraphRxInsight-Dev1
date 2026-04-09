import pandas as pd
import numpy as np

print("Loading DrugBank dataset...")

df = pd.read_csv(
    "DATASETS/raw/drugbank_clean.csv",
    low_memory=False
)

print("Total drugs:", len(df))


# Columns we use
BIO_COLUMNS = [
    "targets",
    "enzymes",
    "pathways"
]


df = df[["drugbank-id"] + BIO_COLUMNS]

df = df.fillna("")


print("Extracting biological terms...")


all_terms = set()

for col in BIO_COLUMNS:

    for row in df[col]:

        words = str(row).lower().split()

        for w in words:

            if len(w) > 3:
                all_terms.add(w)


all_terms = sorted(list(all_terms))


print("Total biological terms:", len(all_terms))


term_index = {t:i for i,t in enumerate(all_terms)}


FEATURE_DIM = len(all_terms)


print("Building feature matrix...")


bio_matrix = np.zeros((len(df), FEATURE_DIM))


for i,row in df.iterrows():

    text = ""

    for col in BIO_COLUMNS:
        text += " " + str(row[col]).lower()

    words = text.split()

    for w in words:

        if w in term_index:
            bio_matrix[i, term_index[w]] = 1


bio_df = pd.DataFrame(
    bio_matrix,
    index=df["drugbank-id"]
)


bio_df.to_csv(
    "DATASETS/processed/bio_features.csv"
)


print("\nBiological features saved")

print("Shape:", bio_df.shape)