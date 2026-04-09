import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader


class DDIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DDIClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


def train_ddi_classifier(
    embeddings_path="DATASETS/processed/drug_embeddings.pt",
    feature_file="DATASETS/processed/drug_features.csv",
    ddi_dataset_file="DATASETS/processed/final_ddi_dataset.csv",
    epochs=10,
    batch_size=256,
    lr=0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embeddings
    embeddings = torch.load(embeddings_path)
    embeddings = embeddings.numpy()

    # Load drug ID mapping (same order used in drug_features.csv)
    features_df = pd.read_csv(feature_file)
    drug_ids = features_df["drug_id"].tolist()
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

    # Load DDI dataset
    ddi_df = pd.read_csv(ddi_dataset_file)

    X = []
    y = []

    for _, row in ddi_df.iterrows():
        d1 = row["drug1_id"]
        d2 = row["drug2_id"]
        label = row["label"]

        if d1 not in drug_to_idx or d2 not in drug_to_idx:
            continue

        emb1 = embeddings[drug_to_idx[d1]]
        emb2 = embeddings[drug_to_idx[d2]]

        pair_feat = np.concatenate([emb1, emb2])
        X.append(pair_feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("✅ Final samples used for training:", len(X))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(DDIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(DDIDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = DDIClassifier(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        preds = []
        true = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)

                pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(pred_labels)
                true.extend(batch_y.numpy())

        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds)
        prec = precision_score(true, preds)
        rec = recall_score(true, preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/ddi_classifier.pth")
    print("\n✅ DDI Classifier saved at models/ddi_classifier.pth")


if __name__ == "__main__":
    train_ddi_classifier()