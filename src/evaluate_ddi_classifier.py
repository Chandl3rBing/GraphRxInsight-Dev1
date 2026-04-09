import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt


class DDIClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


def evaluate_model(
    embeddings_path="DATASETS/processed/drug_embeddings.pt",
    feature_file="DATASETS/processed/drug_features.csv",
    ddi_dataset_file="DATASETS/processed/final_ddi_dataset.csv",
    model_path="models/ddi_classifier.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = torch.load(embeddings_path).numpy()
    features_df = pd.read_csv(feature_file)

    drug_ids = features_df["drug_id"].tolist()
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

    ddi_df = pd.read_csv(ddi_dataset_file)

    X = []
    y = []

    for _, row in ddi_df.iterrows():
        d1, d2, label = row["drug1_id"], row["drug2_id"], row["label"]

        if d1 not in drug_to_idx or d2 not in drug_to_idx:
            continue

        emb1 = embeddings[drug_to_idx[d1]]
        emb2 = embeddings[drug_to_idx[d2]]

        X.append(np.concatenate([emb1, emb2]))
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DDIClassifier(input_dim=X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("\n✅ Confusion Matrix:\n", cm)
    print("\n✅ ROC-AUC Score:", auc)
    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, preds))

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Interaction", "Interaction"])
    plt.yticks([0, 1], ["No Interaction", "Interaction"])
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    print("✅ Confusion matrix saved at results/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()