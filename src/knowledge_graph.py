"""
knowledge_graph.py - Builds and visualises directed knowledge graphs from (S, R, O) triples.
"""

import networkx as nx
import matplotlib.pyplot as plt


class KnowledgeGraphConstructor:
    """Constructs directed NetworkX graphs from (subject, relation, object) triples."""

    def build_graph(self, triples: list) -> nx.DiGraph:
        """Return a DiGraph where nodes are entities and edges are labelled relations."""
        G = nx.DiGraph()
        for subj, rel, obj in triples:
            if subj not in G:
                G.add_node(subj, type="entity")
            if obj not in G:
                G.add_node(obj, type="entity")
            G.add_edge(subj, obj, relation=rel)
        return G

    def plot_graph(self, G: nx.DiGraph, title: str = "Knowledge Graph",
                   bengali_fp=None) -> None:
        """Render a simple spring-layout plot of *G*."""
        font_kwargs = {}
        if bengali_fp is not None:
            font_kwargs["font_family"] = bengali_fp.get_name()

        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=12, **font_kwargs)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")
        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=10, **font_kwargs)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
