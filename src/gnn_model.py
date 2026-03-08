"""
gnn_model.py - Siamese Graph Attention Network (GAT) for structural similarity,
plus helper to convert NetworkX graphs to PyTorch Geometric Data objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

class SiameseGNN(nn.Module):
    """Siamese Graph Attention Network (GAT) for structural similarity."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 output_dim: int = 128, heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim // heads, heads=heads,
                            dropout=0.2, concat=True)
        self.gat2 = GATConv(hidden_dim, output_dim, heads=1,
                            dropout=0.2, concat=False)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward_once(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = (
            data.batch
            if data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        x = self.gat1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.layer_norm2(x)
        x = F.elu(x)
        return global_mean_pool(x, batch)

    def forward(self, data1: Data, data2: Data) -> torch.Tensor:
        emb1 = self.forward_once(data1)
        emb2 = self.forward_once(data2)
        return F.cosine_similarity(emb1, emb2)

def nx_to_pyg_data(nx_graph, node_embeddings_dict: dict) -> Data:
    """Convert a NetworkX DiGraph + node-embedding dict to a PyG ``Data`` object."""
    nodes = list(nx_graph.nodes())
    num_nodes = len(nodes)

    if num_nodes == 0:
        return Data(
            x=torch.zeros((1, 768)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    node_map = {node: i for i, node in enumerate(nodes)}

    x = torch.zeros((num_nodes, 768))
    for node, idx in node_map.items():
        emb = node_embeddings_dict.get(node)
        if emb is not None:
            x[idx] = emb.detach().cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb)

    edges = list(nx_graph.edges())
    if edges:
        edge_list = [[node_map[u], node_map[v]] for u, v in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
