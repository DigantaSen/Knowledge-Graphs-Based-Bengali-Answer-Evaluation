"""
kg_constructor.py
=================
Knowledge Graph construction from typed triples using NetworkX.

  - Accepts (head_id, relation, tail_id, triple_type) triples from
    TripleExtractor; predicate nodes ('PRED::<text>::<n>') become 'predicate'
    nodes, argument nodes are typed by the relation that points at them.
  - Each node stores its display / embedding text ('label'), its 'node_type'
    and whether it is 'negated' (predicates such as 'ওঠে না').
  - Each edge stores its 'triple_type' and a 'weight' (core agent/object
    edges 1.0, attribute edges lower).
  - NODE_TYPES / EDGE_TYPES define the one-hot features the GNN receives.
"""

import networkx as nx

from src.nlp.triple_extractor import PRED_PREFIX, node_label
from src.utils.helpers import has_negation

NODE_TYPES = ('entity', 'predicate', 'number', 'time', 'location', 'quality')
EDGE_TYPES = ('agent', 'object', 'location', 'time', 'count', 'quality', 'possessive')

# triple_type → (node type of the tail, edge weight)
_TYPE_META = {
    'agent':      ('predicate', 1.0),
    'object':     ('entity',    1.0),
    'location':   ('location',  0.8),
    'time':       ('time',      0.8),
    'count':      ('number',    0.8),
    'quality':    ('quality',   0.6),
    'possessive': ('entity',    0.7),
}


class KnowledgeGraphConstructor:
    """Build a typed, weighted directed NetworkX graph from triples."""

    def __init__(self):
        pass

    @staticmethod
    def _add_node(G, node_id: str, node_type: str):
        if node_id in G:
            return
        label = node_label(node_id)
        is_pred = node_id.startswith(PRED_PREFIX)
        G.add_node(node_id, label=label,
                   node_type='predicate' if is_pred else node_type,
                   negated=is_pred and has_negation(label))

    def build_graph(self, triples: list) -> nx.DiGraph:
        """
        Construct a typed DiGraph from a list of 4-tuples
        (head_id, relation, tail_id, triple_type).

        Returns
        -------
        nx.DiGraph with node attributes 'label', 'node_type', 'negated' and
        edge attributes 'relation', 'triple_type', 'weight'.
        """
        G = nx.DiGraph()
        for head, rel, tail, ttype in triples:
            self._add_node(G, head, 'entity')
            if ttype == 'none' or not tail:
                continue
            tail_type, weight = _TYPE_META.get(ttype, ('entity', 0.5))
            self._add_node(G, tail, tail_type)
            G.add_edge(head, tail, relation=rel, weight=weight, triple_type=ttype)
        return G
