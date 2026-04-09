import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


STATIC_MODEL_PATH = "models/final_hard_model.pth"
DYNAMIC_MODEL_PATH = "models/dynamic_model.pth"
X_PATH = "DATASETS/processed/X_dynamic.npy"
Y_PATH = "DATASETS/processed/y_dynamic.npy"
MIN_SAMPLES = 8


class AdaptiveDDIModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return torch.sigmoid(self.out(x))


print("Loading dynamic dataset...")

X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32)

print("Dataset shape:", X.shape)

if len(X) < MIN_SAMPLES:
    raise SystemExit(f"Need at least {MIN_SAMPLES} samples before retraining the dynamic model.")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

print("Loading base model...")

base_state_dict = torch.load(STATIC_MODEL_PATH, map_location="cpu")
model = AdaptiveDDIModel(X.shape[1])
model.load_state_dict(base_state_dict)

if X.shape[1] == base_state_dict["fc1.weight"].shape[1]:
    try:
        dynamic_state_dict = torch.load(DYNAMIC_MODEL_PATH, map_location="cpu")
        if tuple(dynamic_state_dict["fc1.weight"].shape) == (512, X.shape[1]):
            model.load_state_dict(dynamic_state_dict)
            print("Using existing dynamic checkpoint as the warm start.")
    except Exception:
        print("Dynamic checkpoint missing or incompatible; starting from the static base model.")

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.BCELoss()

print("Training dynamic model...")

model.train()

for epoch in range(6):
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = loss_fn(preds, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/6 Loss: {loss.item():.6f}")

model.eval()
torch.save(model.state_dict(), DYNAMIC_MODEL_PATH)

print("Dynamic model saved to", DYNAMIC_MODEL_PATH)
