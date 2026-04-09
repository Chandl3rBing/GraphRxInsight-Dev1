import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading scaled chunks...")

X1=np.load("DATASETS/processed/X_chunk_3_scaled.npy")
y1=np.load("DATASETS/processed/y_chunk_3.npy")

X2=np.load("DATASETS/processed/X_chunk_4_scaled.npy")
y2=np.load("DATASETS/processed/y_chunk_4.npy")

X3=np.load("DATASETS/processed/X_chunk_5_scaled.npy")
y3=np.load("DATASETS/processed/y_chunk_5.npy")

X=np.vstack([X1,X2,X3])
y=np.hstack([y1,y2,y3])

print("Dataset shape:",X.shape)


X=torch.tensor(X,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32).view(-1,1)


input_dim=X.shape[1]


class DDIModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=nn.Linear(input_dim,1024)
        self.bn1=nn.BatchNorm1d(1024)

        self.fc2=nn.Linear(1024,512)
        self.bn2=nn.BatchNorm1d(512)

        self.fc3=nn.Linear(512,256)
        self.bn3=nn.BatchNorm1d(256)

        self.out=nn.Linear(256,1)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.3)


    def forward(self,x):

        x=self.relu(self.bn1(self.fc1(x)))
        x=self.dropout(x)

        x=self.relu(self.bn2(self.fc2(x)))
        x=self.dropout(x)

        x=self.relu(self.bn3(self.fc3(x)))
        x=self.dropout(x)

        x=torch.sigmoid(self.out(x))

        return x


model=DDIModel(input_dim)

criterion=nn.BCELoss()

optimizer=optim.Adam(
    model.parameters(),
    lr=0.001
)


print("\nTraining started\n")


epochs=10

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs=model(X)

    loss=criterion(outputs,y)

    loss.backward()

    optimizer.step()


    print("Epoch",epoch+1,"Loss:",loss.item())


torch.save(
    model.state_dict(),
    "models/nn_scaled_model.pth"
)


print("\nTraining Finished")