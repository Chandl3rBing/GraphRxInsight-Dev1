import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


print("Loading dataset...")

X = np.load(
"DATASETS/processed/X_hard_chunk_0_scaled.npy"
)

y = np.load(
"DATASETS/processed/y_hard_chunk_0.npy"
)


print("Shape:", X.shape)


X = torch.tensor(X, dtype=torch.float32)
y_true = y


input_dim = X.shape[1]


class FinalDDIModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim,512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256,128)
        self.bn3 = nn.BatchNorm1d(128)

        self.out = nn.Linear(128,1)

        self.relu = nn.ReLU()


    def forward(self,x):

        x = self.relu(self.bn1(self.fc1(x)))

        x = self.relu(self.bn2(self.fc2(x)))

        x = self.relu(self.bn3(self.fc3(x)))

        x = torch.sigmoid(self.out(x))

        return x



print("Loading model...")

model = FinalDDIModel(input_dim)


model.load_state_dict(

torch.load(
"models/final_hard_model.pth",
map_location="cpu"
)

)


model.eval()


print("Predicting...")


with torch.no_grad():

    preds = model(X).numpy()


preds = (preds > 0.5).astype(int)


accuracy = accuracy_score(
y_true,
preds
)

f1 = f1_score(
y_true,
preds
)


print("\nRESULTS")

print("Accuracy:",accuracy)

print("F1 Score:",f1)