import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading dataset")

X=np.load("DATASETS/processed/X_chunk_3.npy")
y=np.load("DATASETS/processed/y_chunk_3.npy")

print("Shape:",X.shape)


X=torch.tensor(X,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32).view(-1,1)


class DDIModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=nn.Linear(input_dim,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,1)

    def forward(self,x):

        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))

        return x


input_dim=X.shape[1]

model=DDIModel(input_dim)

criterion=nn.BCELoss()

optimizer=optim.Adam(model.parameters(),lr=0.001)


print("\nTraining Neural Model")


for epoch in range(10):

    optimizer.zero_grad()

    outputs=model(X)

    loss=criterion(outputs,y)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch,"Loss:",loss.item())


torch.save(
model.state_dict(),
"models/nn_3284_model.pth"
)

print("\nNeural Model Saved")