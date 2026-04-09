import pandas as pd
import numpy as np
import torch

print("Loading datasets...")


# Drug IDs
drugbank = pd.read_csv(
"DATASETS/raw/drugbank_clean.csv"
)

drug_ids = drugbank["drugbank-id"].values


# Embeddings
embeddings = torch.load(
"DATASETS/processed/drug_embeddings.pt"
).numpy()


# ATC PCA
atc = pd.read_csv(
"DATASETS/processed/atc_pca_features.csv"
)

atc = atc.iloc[:,1:].values


# Chemical Features
chem = pd.read_csv(
"DATASETS/processed/chemical_features.csv"
)

chem_ids = chem.iloc[:,0].astype(str).values
chem_feat = chem.iloc[:,1:].values


# Biological PCA (Correct File Name)
bio = pd.read_csv(
"DATASETS/processed/bio_features_pca.csv"
)

bio = bio.iloc[:,1:].values


print("Embeddings:",embeddings.shape)
print("ATC:",atc.shape)
print("Chemical:",chem_feat.shape)
print("Biological:",bio.shape)


feature_dim = (
embeddings.shape[1]
+ atc.shape[1]
+ chem_feat.shape[1]
+ bio.shape[1]
)


print("\nTotal features per drug:",feature_dim)


features = np.zeros(
(len(drug_ids),feature_dim),
dtype=np.float32
)


chem_map = {chem_ids[i]:i for i in range(len(chem_ids))}


for i,drug in enumerate(drug_ids):

    f=[]

    f.extend(embeddings[i])
    f.extend(atc[i])
    f.extend(bio[i])


    if drug in chem_map:

        f.extend(
        chem_feat[chem_map[drug]]
        )

    else:

        f.extend(
        np.zeros(chem_feat.shape[1])
        )


    features[i]=f



df=pd.DataFrame(features)

df.insert(0,"drug_id",drug_ids)


df.to_csv(
"DATASETS/processed/unified_drug_features.csv",
index=False
)


print("\nUnified Features Shape:",features.shape)

print("Saved unified features successfully.")