import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("Loading dataset...")

X = np.load("DATASETS/processed/X_chunk_0.npy")
y = np.load("DATASETS/processed/y_chunk_0.npy")

print("Shape:", X.shape)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


input_dim = X.shape[1]


class DDIModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


model = DDIModel(input_dim)


criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)


print("Training started...")


for epoch in range(10):

    outputs = model(X)

    loss = criterion(outputs,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("Epoch",epoch+1,"Loss:",loss.item())


torch.save(model.state_dict(),
           "models/small_model_100k.pth")


print("Training Finished")