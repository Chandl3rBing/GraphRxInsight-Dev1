import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

print("Training Hard Model")


# ---------- LOAD FEATURES ----------

df = pd.read_csv(
"DATASETS/processed/unified_drug_features.csv"
)

drug_ids = df.iloc[:,0].values
features = df.iloc[:,1:].values.astype(np.float32)

print("Features:",features.shape)


# Create mapping DBID -> index

drug_map = {}

for i,d in enumerate(drug_ids):
    drug_map[d] = i


print("Drug Map Size:",len(drug_map))


# ---------- LOAD DATASET ----------

ddi = pd.read_csv(
"DATASETS/processed/hard_dataset.csv"
)


X_list=[]
y_list=[]


for i,row in ddi.head(100000).iterrows():

    if i%10000==0:
        print("Processed:",i)

    d1=row["drug1_id"]
    d2=row["drug2_id"]

    if d1 not in drug_map:
        continue

    if d2 not in drug_map:
        continue


    f1=features[drug_map[d1]]
    f2=features[drug_map[d2]]

    pair=np.concatenate([f1,f2])

    X_list.append(pair)
    y_list.append(row["label"])



X=np.array(X_list,dtype=np.float32)
y=np.array(y_list,dtype=np.float32)

print("Dataset Shape:",X.shape)


input_dim=X.shape[1]


# ---------- MODEL ----------

class Model(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=nn.Linear(input_dim,512)
        self.fc2=nn.Linear(512,256)
        self.out=nn.Linear(256,1)


    def forward(self,x):

        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.sigmoid(self.out(x))

        return x


model=Model(input_dim)


optimizer=optim.Adam(
model.parameters(),
lr=0.001
)

loss_fn=nn.BCELoss()


X=torch.tensor(X)
y=torch.tensor(y).view(-1,1)


# ---------- TRAIN ----------

for epoch in range(10):

    optimizer.zero_grad()

    out=model(X)

    loss=loss_fn(out,y)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch+1,"Loss:",loss.item())


torch.save(
model.state_dict(),
"models/hard_nn_model.pth"
)

print("Model Saved")