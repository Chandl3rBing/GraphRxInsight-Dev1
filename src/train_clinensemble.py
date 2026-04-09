import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# -----------------------------
# CLINENSEMBLE MODEL
# -----------------------------
class CLINENSEMBLE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
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


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_clinensemble(
    dataset_file="DATASETS/processed/final_ddi_dataset.csv",
    atc_file="DATASETS/processed/atc_pca_features.csv",
    embeddings_file="DATASETS/processed/drug_embeddings.pt",
    drug_features_file="DATASETS/processed/drug_features.csv",
    save_path="models/clinensemble_pca_model.pth",
    epochs=10,
    batch_size=2048,
    lr=0.001,
    sample_size=600000
):
    print("\n====================================================")
    print("🚀 Training CLINENSEMBLE Model (GAT + ATC PCA)")
    print("====================================================\n")

    # Load dataset
    print("📂 Loading DDI dataset:", dataset_file)
    ddi_df = pd.read_csv(dataset_file)

    if sample_size is not None and sample_size < len(ddi_df):
        ddi_df = ddi_df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"⚡ Using sample size: {sample_size}")

    # Load embeddings
    print("📂 Loading embeddings:", embeddings_file)
    embeddings = torch.load(embeddings_file).numpy()
    embedding_dim = embeddings.shape[1]
    print("✅ Embedding dim:", embedding_dim)

    # Load drug_features mapping
    print("📂 Loading drug features mapping:", drug_features_file)
    drug_feat_df = pd.read_csv(drug_features_file)
    drug_ids = drug_feat_df["drug_id"].tolist()
    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}

    # Load ATC PCA
    print("📂 Loading ATC PCA features:", atc_file)
    atc_df = pd.read_csv(atc_file).set_index("drug_id")

    # ✅ FIX: Convert all PCA columns to numeric and replace NaN
    atc_df = atc_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    atc_dim = atc_df.shape[1]
    print("✅ ATC PCA dim:", atc_dim)

    # Compute final input dim
    drug_feature_dim = embedding_dim + atc_dim
    input_dim = drug_feature_dim * 2

    print("\n===================================")
    print("Final Drug Feature Dim:", drug_feature_dim)
    print("Final Pair Input Dim:", input_dim)
    print("===================================\n")

    # -----------------------------
    # Build training data
    # -----------------------------
    X_list = []
    y_list = []

    missing_count = 0
    skipped_dim_mismatch = 0

    expected_dim = input_dim

    print("🔄 Building feature pairs...")

    for _, row in tqdm(ddi_df.iterrows(), total=len(ddi_df)):
        d1 = row["drug1_id"]
        d2 = row["drug2_id"]
        label = row["label"]

        if d1 not in drug_to_idx or d2 not in drug_to_idx:
            missing_count += 1
            continue

        if d1 not in atc_df.index or d2 not in atc_df.index:
            missing_count += 1
            continue

        emb1 = embeddings[drug_to_idx[d1]].astype(np.float32).flatten()
        emb2 = embeddings[drug_to_idx[d2]].astype(np.float32).flatten()

        atc1 = atc_df.loc[d1].values.astype(np.float32).flatten()
        atc2 = atc_df.loc[d2].values.astype(np.float32).flatten()

        feat1 = np.concatenate([emb1, atc1])
        feat2 = np.concatenate([emb2, atc2])

        pair_feat = np.concatenate([feat1, feat2])

        # ✅ Ensure fixed input length
        if pair_feat.shape[0] != expected_dim:
            skipped_dim_mismatch += 1
            continue

        X_list.append(pair_feat)
        y_list.append(label)

    print(f"⚠ Skipped missing pairs: {missing_count}")
    print(f"⚠ Skipped dimension mismatch pairs: {skipped_dim_mismatch}")

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print("✅ Total usable training samples:", X.shape[0])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    # Model init
    model = CLINENSEMBLE(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\n✅ Model initialized.")
    print("Training started...\n")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(X_train.size(0))
        X_train = X_train[perm]
        y_train = y_train[perm]

        total_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            preds = torch.argmax(logits, dim=1).numpy()
            true = y_test.numpy()

            acc = accuracy_score(true, preds)
            f1 = f1_score(true, preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print("\n====================================================")
    print("✅ Training Completed Successfully!")
    print("Model Saved At:", save_path)
    print("====================================================\n")


if __name__ == "__main__":
    train_clinensemble()