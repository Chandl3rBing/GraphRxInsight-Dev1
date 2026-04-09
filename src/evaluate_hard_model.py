import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

print("Evaluating Hard Model")


# Load only existing chunk
X = np.load(
"DATASETS/processed/X_hard_chunk_0.npy"
)

y = np.load(
"DATASETS/processed/y_hard_chunk_0.npy"
)


print("Dataset Shape:", X.shape)


input_dim = X.shape[1]


class HardModel(torch.nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=torch.nn.Linear(input_dim,512)
        self.bn1=torch.nn.BatchNorm1d(512)

        self.fc2=torch.nn.Linear(512,256)
        self.bn2=torch.nn.BatchNorm1d(256)

        self.fc3=torch.nn.Linear(256,128)

        self.out=torch.nn.Linear(128,1)


    def forward(self,x):

        x=torch.relu(self.bn1(self.fc1(x)))
        x=torch.relu(self.bn2(self.fc2(x)))
        x=torch.relu(self.fc3(x))

        x=torch.sigmoid(self.out(x))

        return x



print("Loading Model")


model=HardModel(input_dim)


model.load_state_dict(
torch.load(
"models/hard_nn_model.pth",
map_location="cpu"
)
)


model.eval()


X=torch.tensor(X,dtype=torch.float32)


with torch.no_grad():

    pred=model(X).numpy()



pred=(pred>0.5).astype(int).flatten()


print("\nRESULTS")

print("Accuracy:",accuracy_score(y,pred))

print("F1 Score:",f1_score(y,pred))
      
    