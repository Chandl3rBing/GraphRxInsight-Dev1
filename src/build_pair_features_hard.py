import pandas as pd
import numpy as np

print("Building Hard Pair Features")


# Load features
df = pd.read_csv(
"DATASETS/processed/unified_drug_features.csv"
)

# Remove DrugBank ID column if present
if df.columns[0].lower().startswith("db") or "drug" in df.columns[0].lower():

    print("Removing ID column")

    df = df.iloc[:,1:]


features = df.values.astype(np.float32)


print("Feature matrix:",features.shape)


ddi = pd.read_csv(
"DATASETS/processed/hard_dataset.csv"
)


X_list=[]
y_list=[]


for i,row in ddi.iterrows():

    if i%10000==0:
        print("Processed:",i)


    try:

        d1=row["drug1_id"]
        d2=row["drug2_id"]


        d1=int(str(d1).replace("DB",""))
        d2=int(str(d2).replace("DB",""))


        f1=features[d1]
        f2=features[d2]


        pair=np.concatenate([f1,f2])


        X_list.append(pair)

        y_list.append(row["label"])


    except:
        continue



X=np.array(X_list,dtype=np.float32)
y=np.array(y_list,dtype=np.float32)


np.save(
"DATASETS/processed/X_hard.npy",
X
)

np.save(
"DATASETS/processed/y_hard.npy",
y
)


print("\nSaved Hard Dataset")

print("Shape:",X.shape)