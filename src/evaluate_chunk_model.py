import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

print("Evaluating PCA model...")

INPUT_DIM = 2884   # 1442 × 2


class DDIModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 128)

        self.out = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = self.relu(self.fc4(x))

        x = self.sigmoid(self.out(x))

        return x


model = DDIModel()


model.load_state_dict(
    torch.load(
        "models/high_accuracy_model_pca.pth",
        map_location="cpu"
    )
)

model.eval()


print("Loading validation chunk...")


X = np.load("DATASETS/processed/X_chunk_2.npy")
y = np.load("DATASETS/processed/y_chunk_2.npy")


X = torch.tensor(X, dtype=torch.float32)


with torch.no_grad():

    preds = model(X).numpy()


preds_binary = (preds > 0.5).astype(int)


acc = accuracy_score(y, preds_binary)

f1 = f1_score(y, preds_binary)


print("\nRESULTS")

print("Accuracy:", acc)

print("F1 Score:", f1)