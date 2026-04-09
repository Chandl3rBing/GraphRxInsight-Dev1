import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Starting training...")


# -------------------------
# Model Settings
# -------------------------

INPUT_DIM = 2884   # 1442 per drug × 2

EPOCHS = 3
LEARNING_RATE = 0.001


# -------------------------
# Model Definition
# -------------------------

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


# -------------------------
# Initialize Model
# -------------------------

model = DDIModel()

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training Loop
# -------------------------

for epoch in range(EPOCHS):

    print("\nEpoch:", epoch + 1)

    for chunk_id in range(3):

        print("Loading chunk:", chunk_id)

        X = np.load(
            f"DATASETS/processed/X_chunk_{chunk_id}.npy"
        )

        y = np.load(
            f"DATASETS/processed/y_chunk_{chunk_id}.npy"
        )


        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        ).view(-1,1)


        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()


        print("Chunk Loss:", loss.item())


print("\nTraining Finished")


# -------------------------
# Save Model
# -------------------------

torch.save(
    model.state_dict(),
    "models/high_accuracy_model_pca.pth"
)

print("Model saved as high_accuracy_model_pca.pth")