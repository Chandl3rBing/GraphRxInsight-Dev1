import pandas as pd
import numpy as np
import torch


print("Loading Final AI Model")


# Load features
df = pd.read_csv(
"DATASETS/processed/unified_drug_features.csv"
)


# Remove ID column if exists
if str(df.iloc[0,0]).startswith("DB"):
    df = df.iloc[:,1:]


features = df.values.astype(np.float32)


input_dim = features.shape[1]*2


class FinalModel(torch.nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=torch.nn.Linear(input_dim,1024)
        self.bn1=torch.nn.BatchNorm1d(1024)

        self.fc2=torch.nn.Linear(1024,512)
        self.bn2=torch.nn.BatchNorm1d(512)

        self.fc3=torch.nn.Linear(512,256)

        self.out=torch.nn.Linear(256,1)


    def forward(self,x):

        x=torch.relu(self.bn1(self.fc1(x)))
        x=torch.relu(self.bn2(self.fc2(x)))
        x=torch.relu(self.fc3(x))

        x=torch.sigmoid(self.out(x))

        return x



model=FinalModel(input_dim)


model.load_state_dict(
torch.load(
"models/hard_nn_model.pth",
map_location="cpu"
)
)


model.eval()


def predict(drug1,drug2):

    d1=int(drug1.replace("DB",""))
    d2=int(drug2.replace("DB",""))


    f1=features[d1]
    f2=features[d2]


    pair=np.concatenate([f1,f2])


    X=torch.tensor(pair,dtype=torch.float32).view(1,-1)


    prob=model(X).item()


    print("\nPrediction Probability:",prob)


    if prob>0.7:
        print("Risk: HIGH")

    elif prob>0.4:
        print("Risk: MODERATE")

    else:
        print("Risk: LOW")



# Example test
predict("DB00001","DB00002")