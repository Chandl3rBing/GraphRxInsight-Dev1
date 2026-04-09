import pandas as pd
import numpy as np

print("Building Hard Pair Features (SAFE)")


# Load unified features
df = pd.read_csv(
"DATASETS/processed/unified_drug_features.csv"
)


# Remove DrugBank ID column safely
first_value=str(df.iloc[0,0])

if first_value.startswith("DB"):

    print("Removing ID column")

    df=df.iloc[:,1:]


features=df.values.astype(np.float32)

print("Feature shape:",features.shape)


ddi=pd.read_csv(
"DATASETS/processed/hard_dataset.csv"
)


chunk_size=50000

X_chunk=[]
y_chunk=[]

chunk_id=0


for i,row in ddi.iterrows():

    if i%10000==0:
        print("Processed:",i)


    try:

        d1=int(str(row["drug1_id"]).replace("DB",""))
        d2=int(str(row["drug2_id"]).replace("DB",""))


        f1=features[d1]
        f2=features[d2]


        pair=np.concatenate([f1,f2])


        X_chunk.append(pair)
        y_chunk.append(row["label"])


        if len(X_chunk)==chunk_size:

            print("Saving chunk:",chunk_id)

            X_array=np.array(X_chunk,dtype=np.float32)
            y_array=np.array(y_chunk,dtype=np.float32)


            np.save(
            f"DATASETS/processed/X_hard_chunk_{chunk_id}.npy",
            X_array
            )

            np.save(
            f"DATASETS/processed/y_hard_chunk_{chunk_id}.npy",
            y_array
            )


            X_chunk=[]
            y_chunk=[]

            chunk_id+=1


    except:
        continue



print("\nFinished safely")