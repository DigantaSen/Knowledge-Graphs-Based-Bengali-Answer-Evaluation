"""
siamese_gnn.py
==============
Siamese GAT-GNN for graph similarity and PyG data conversion utility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from src.graph.embedder import EMB_DIM
from src.graph.kg_constructor import EDGE_TYPES, NODE_TYPES

# Node features: text embedding | one-hot node type | negated flag
NODE_FEAT_DIM = EMB_DIM + len(NODE_TYPES) + 1
# Edge features: weight | one-hot edge type | reverse-direction flag
EDGE_FEAT_DIM = 1 + len(EDGE_TYPES) + 1


class SiameseGNN(nn.Module):
    """
    Twin-branch Graph Attention Network (GAT) for computing a cosine
    similarity score between two knowledge-graph PyG Data objects.

    Edge features (weight, relation type, direction) feed the attention via
    ``edge_dim``, so the network can tell an agent edge from an object edge
    and see which way it points.
    """

    def __init__(self, input_dim: int = NODE_FEAT_DIM, hidden_dim: int = 256,
                 output_dim: int = 128, heads: int = 4, edge_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        self.edge_dim    = edge_dim
        self.gat1        = GATConv(input_dim, hidden_dim // heads, heads=heads,
                                   dropout=0.2, concat=True, edge_dim=edge_dim,
                                   fill_value=0.0)
        self.gat2        = GATConv(hidden_dim, output_dim, heads=1,
                                   dropout=0.2, concat=False, edge_dim=edge_dim,
                                   fill_value=0.0)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward_once(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1], self.edge_dim),
                                    dtype=torch.float32, device=x.device)

        batch = (data.batch if getattr(data, 'batch', None) is not None
                 else torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.layer_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = self.layer_norm2(x)
        x = F.elu(x)
        return global_mean_pool(x, batch)

    def forward(self, data1: Data, data2: Data) -> torch.Tensor:
        emb1 = self.forward_once(data1)
        emb2 = self.forward_once(data2)
        return F.cosine_similarity(emb1, emb2)


def _one_hot(index: int, size: int) -> list:
    v = [0.0] * size
    if 0 <= index < size:
        v[index] = 1.0
    return v


def nx_to_pyg_data(nx_graph, node_embeddings_dict: dict) -> Data:
    """
    Convert a NetworkX DiGraph (from KnowledgeGraphConstructor) with
    precomputed node embeddings to a PyTorch Geometric Data object.

    Parameters
    ----------
    nx_graph : nx.DiGraph
    node_embeddings_dict : dict
        Mapping from node id to {'emb': Tensor[768], 'node_type': str}
        (or a bare Tensor[768]).

    Returns
    -------
    torch_geometric.data.Data  with:
        x           : float32 tensor [N, NODE_FEAT_DIM]
        edge_index  : long tensor    [2, 2E]   (every edge plus its reverse)
        edge_attr   : float32 tensor [2E, EDGE_FEAT_DIM]
    """
    nodes = list(nx_graph.nodes())
    if not nodes:
        return Data(x=torch.zeros((1, NODE_FEAT_DIM)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float32))

    node_mapping = {node: i for i, node in enumerate(nodes)}
    x = torch.zeros((len(nodes), NODE_FEAT_DIM))
    for node, idx in node_mapping.items():
        attrs = nx_graph.nodes[node]
        entry = node_embeddings_dict.get(node)
        emb = entry.get('emb') if isinstance(entry, dict) else entry
        if emb is not None:
            x[idx, :EMB_DIM] = emb.detach().cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb)
        ntype = attrs.get('node_type', 'entity')
        x[idx, EMB_DIM:EMB_DIM + len(NODE_TYPES)] = torch.tensor(
            _one_hot(NODE_TYPES.index(ntype) if ntype in NODE_TYPES else 0, len(NODE_TYPES)))
        x[idx, -1] = float(bool(attrs.get('negated', False)))

    src, dst, feats = [], [], []
    for u, v, d in nx_graph.edges(data=True):
        etype = d.get('triple_type', '')
        base = [float(d.get('weight', 1.0))] + _one_hot(
            EDGE_TYPES.index(etype) if etype in EDGE_TYPES else -1, len(EDGE_TYPES))
        src += [node_mapping[u], node_mapping[v]]
        dst += [node_mapping[v], node_mapping[u]]
        feats += [base + [0.0], base + [1.0]]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
