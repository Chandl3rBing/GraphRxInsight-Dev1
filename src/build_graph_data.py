import pandas as pd
import torch
from torch_geometric.data import Data
import os


def build_graph_data(feature_file="DATASETS/processed/drug_features.csv",
                     ddi_file="DATASETS/processed/ddi_pairs.csv",
                     output_file="DATASETS/processed/drug_graph.pt"):

    features_df = pd.read_csv(feature_file)
    ddi_df = pd.read_csv(ddi_file)

    # Map drug IDs to node index
    drug_ids = features_df["drug_id"].tolist()
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

    # Create node feature matrix
    x = torch.tensor(features_df[["avg_mass", "mono_mass"]].values, dtype=torch.float)

    # Build edges
    edge_list = []
    for _, row in ddi_df.iterrows():
        d1 = row["drug1_id"]
        d2 = row["drug2_id"]

        if d1 in drug_to_idx and d2 in drug_to_idx:
            edge_list.append([drug_to_idx[d1], drug_to_idx[d2]])
            edge_list.append([drug_to_idx[d2], drug_to_idx[d1]])  # undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    graph_data = Data(x=x, edge_index=edge_index)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(graph_data, output_file)

    print("✅ Graph Data created successfully!")
    print("Nodes:", graph_data.num_nodes)
    print("Edges:", graph_data.num_edges)
    print("Saved at:", output_file)


if __name__ == "__main__":
    build_graph_data()