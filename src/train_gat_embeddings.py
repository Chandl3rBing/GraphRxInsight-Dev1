import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os
import random


class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=0.3)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


def train_gat(
    graph_path="DATASETS/processed/drug_graph.pt",
    output_embedding_path="DATASETS/processed/drug_embeddings.pt",
    epochs=20,
    lr=0.005,
    edge_batch_size=5000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PyTorch fix
    data = torch.load(graph_path, weights_only=False)
    data = data.to(device)

    model = GATEncoder(
        input_dim=data.num_features,
        hidden_dim=32,
        out_dim=64,
        heads=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_index = data.edge_index
    num_edges = edge_index.size(1)

    print("✅ Loaded Graph")
    print("Nodes:", data.num_nodes)
    print("Edges:", num_edges)
    print("Training started...\n")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass on full graph (needed for GAT)
        z = model(data.x, edge_index)

        # Random mini-batch edges
        idx = torch.randint(0, num_edges, (edge_batch_size,), device=device)

        src = edge_index[0, idx]
        dst = edge_index[1, idx]

        # Positive loss
        pos_score = (z[src] * z[dst]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

        # Negative sampling
        neg_dst = torch.randint(0, data.num_nodes, (edge_batch_size,), device=device)
        neg_score = (z[src] * z[neg_dst]).sum(dim=1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(output_embedding_path), exist_ok=True)
    torch.save(z.detach().cpu(), output_embedding_path)

    print("\n✅ GAT Embeddings saved at:", output_embedding_path)


if __name__ == "__main__":
    train_gat()