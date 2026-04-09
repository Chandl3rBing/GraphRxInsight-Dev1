import pandas as pd
import numpy as np

print("Loading unified features...")

features = pd.read_csv(
"DATASETS/processed/unified_drug_features.csv"
)

drug_ids = features.iloc[:,0].astype(str).values

drug_features = features.iloc[:,1:].values.astype(np.float32)

print("Drug features shape:",drug_features.shape)


drug_to_idx = {}

for i,d in enumerate(drug_ids):
    drug_to_idx[d]=i



print("Loading DDI dataset...")

ddi = pd.read_csv(
"DATASETS/processed/final_ddi_dataset.csv"
)

print("DDI shape:",ddi.shape)


MAX_SAMPLES = 100000


X_list=[]
y_list=[]

count=0


for _,row in ddi.iterrows():

    d1=str(row["drug1_id"]).strip()
    d2=str(row["drug2_id"]).strip()

    label=int(row["label"])


    if d1 not in drug_to_idx:
        continue

    if d2 not in drug_to_idx:
        continue


    f1=drug_features[
        drug_to_idx[d1]
    ]

    f2=drug_features[
        drug_to_idx[d2]
    ]


    pair=np.concatenate((f1,f2))


    X_list.append(pair)
    y_list.append(label)


    count+=1


    if count%10000==0:

        print("Processed:",count)


    if count>=MAX_SAMPLES:

        break



print("Saving dataset...")


X=np.array(X_list,dtype=np.float32)
y=np.array(y_list,dtype=np.int32)


np.save(
"DATASETS/processed/X_chunk_0.npy",
X
)

np.save(
"DATASETS/processed/y_chunk_0.npy",
y
)


print("Saved.")

print("X shape:",X.shape)
print("y shape:",y.shape)