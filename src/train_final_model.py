import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


FEATURE_PATH = "DATASETS/processed/unified_drug_features.csv"
HARD_DATASET_PATH = "DATASETS/processed/hard_dataset.csv"
MODEL_PATH = "models/final_hard_model.pth"
SCALER_STATS_PATH = "models/final_hard_scaler_stats.npz"

TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "60000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))


print("Loading unified drug features...")

features_df = pd.read_csv(FEATURE_PATH)
features_df["drug_id"] = features_df["drug_id"].astype(str).str.strip()

drug_feature_map = {
    row["drug_id"]: row.iloc[1:].values.astype(np.float32)
    for _, row in features_df.iterrows()
}

per_drug_dim = len(next(iter(drug_feature_map.values())))
print("Drug feature count:", len(drug_feature_map))
print("Per-drug feature dim:", per_drug_dim)

print("Loading hard dataset...")

ddi_df = pd.read_csv(HARD_DATASET_PATH)
ddi_df["drug1_id"] = ddi_df["drug1_id"].astype(str).str.strip()
ddi_df["drug2_id"] = ddi_df["drug2_id"].astype(str).str.strip()

target_df = ddi_df.head(TRAIN_SAMPLES)
print("Requested samples:", TRAIN_SAMPLES)
print("Using hard dataset slice:", target_df.shape)

X_list = []
y_list = []
skipped = 0

for idx, row in target_df.iterrows():
    d1 = row["drug1_id"]
    d2 = row["drug2_id"]

    f1 = drug_feature_map.get(d1)
    f2 = drug_feature_map.get(d2)

    if f1 is None or f2 is None:
        skipped += 1
        continue

    pair = np.concatenate([f1, f2]).astype(np.float32)
    X_list.append(pair)
    y_list.append(float(row["label"]))

    if (idx + 1) % 10000 == 0:
        print("Processed rows:", idx + 1)

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)

print("Usable training samples:", X.shape[0])
print("Skipped rows:", skipped)
print("Pair feature dim:", X.shape[1])

print("Fitting scaler stats on the training slice...")

mean = X.mean(axis=0).astype(np.float32)
scale = X.std(axis=0).astype(np.float32)
scale[scale == 0] = 1.0

X_scaled = ((X - mean) / scale).astype(np.float32)
np.savez(SCALER_STATS_PATH, mean=mean, scale=scale)

print("Scaler stats saved to", SCALER_STATS_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class FinalDDIModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.out(x))
        return x


model = FinalDDIModel(X_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training started...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    avg_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {avg_loss:.6f}")

print("Saving model...")

torch.save(model.state_dict(), MODEL_PATH)

print("Training finished")
print("Saved model to", MODEL_PATH)
