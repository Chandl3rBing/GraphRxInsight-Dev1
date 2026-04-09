import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading chunks...")

X1=np.load("DATASETS/processed/X_chunk_3.npy")
y1=np.load("DATASETS/processed/y_chunk_3.npy")

X2=np.load("DATASETS/processed/X_chunk_4.npy")
y2=np.load("DATASETS/processed/y_chunk_4.npy")

X3=np.load("DATASETS/processed/X_chunk_5.npy")
y3=np.load("DATASETS/processed/y_chunk_5.npy")


X=np.vstack([X1,X2,X3])
y=np.concatenate([y1,y2,y3])

print("Dataset shape:",X.shape)


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


model=DDIModel(X.shape[1])

criterion=nn.BCELoss()

optimizer=optim.Adam(model.parameters(),lr=0.001)


print("\nTraining Neural Network")


for epoch in range(20):

    optimizer.zero_grad()

    outputs=model(X)

    loss=criterion(outputs,y)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch,"Loss:",loss.item())


torch.save(

model.state_dict(),

"models/nn_large_model.pth"

)

print("\nNeural Model Saved")