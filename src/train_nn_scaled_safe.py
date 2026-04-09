import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


print("Training Neural Model (Memory Safe)")


# Get input dimension
X_test=np.load("DATASETS/processed/X_chunk_3_scaled.npy")
input_dim=X_test.shape[1]


class DDIModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=nn.Linear(input_dim,512)
        self.bn1=nn.BatchNorm1d(512)

        self.fc2=nn.Linear(512,256)
        self.bn2=nn.BatchNorm1d(256)

        self.fc3=nn.Linear(256,128)
        self.bn3=nn.BatchNorm1d(128)

        self.out=nn.Linear(128,1)

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


epochs=10


for epoch in range(epochs):

    print("\nEpoch:",epoch+1)

    for i in [3,4,5]:

        print("Loading chunk:",i)

        X=np.load(f"DATASETS/processed/X_chunk_{i}_scaled.npy")
        y=np.load(f"DATASETS/processed/y_chunk_{i}.npy")

        X=torch.tensor(X,dtype=torch.float32)
        y=torch.tensor(y,dtype=torch.float32).view(-1,1)


        optimizer.zero_grad()

        outputs=model(X)

        loss=criterion(outputs,y)

        loss.backward()

        optimizer.step()


        print("Chunk",i,"Loss:",loss.item())


torch.save(
model.state_dict(),
"models/nn_scaled_safe.pth"
)

print("\nTraining Finished")