import pandas as pd
import numpy as np

print("Building Hard Negative Samples")


ddi = pd.read_csv(
"DATASETS/processed/final_ddi_dataset.csv"
)


pos = ddi[ddi["label"]==1]

neg = ddi[ddi["label"]==0]


print("Positives:",len(pos))
print("Negatives:",len(neg))


drug_counts = pd.concat([
pos["drug1_id"],
pos["drug2_id"]
]).value_counts()


top_drugs = drug_counts.head(1000).index


hard_neg = neg[
neg["drug1_id"].isin(top_drugs) &
neg["drug2_id"].isin(top_drugs)
]


print("Hard negatives found:",len(hard_neg))


# Balance dataset safely
target_size = min(len(pos),len(hard_neg))


pos = pos.sample(target_size,random_state=42)

hard_neg = hard_neg.sample(
target_size,
random_state=42
)


dataset = pd.concat([pos,hard_neg])


dataset.to_csv(
"DATASETS/processed/hard_dataset.csv",
index=False
)


print("\nSaved hard dataset")
print("Final Size:",len(dataset))