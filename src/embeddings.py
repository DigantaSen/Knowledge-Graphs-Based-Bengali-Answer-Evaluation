"""
embeddings.py - Dual-encoder embedder: LaBSE (sentence-level) + BanglaBERT (node-level).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from src.config import SENTENCE_MODEL_NAME, NODE_MODEL_NAME


class BanglaBERTEmbedder:
    """Dual-encoder: LaBSE for sentence-level similarity, BanglaBERT for node-level embeddings."""

    def __init__(
        self,
        sentence_model_name: str = SENTENCE_MODEL_NAME,
        node_model_name: str = NODE_MODEL_NAME,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Sentence-BERT model: {sentence_model_name}...")
        self.sentence_model = SentenceTransformer(sentence_model_name, device=str(self.device))
        print(f"  LaBSE loaded on {self.device}.")

        print(f"Loading BanglaBERT for node embeddings: {node_model_name}...")
        self.node_tokenizer = AutoTokenizer.from_pretrained(node_model_name)
        self.node_model = AutoModel.from_pretrained(node_model_name)
        self.node_model.to(self.device)
        self.node_model.eval()
        print(f"  BanglaBERT loaded on {self.device}.")

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Sentence-level cosine similarity using LaBSE."""
        embeddings = self.sentence_model.encode(
            [text1, text2], convert_to_tensor=True
        )
        sim = F.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return sim.item()

    # ── word / node level ─────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> torch.Tensor:
        """768-dim BanglaBERT mean-pool embedding for a single word / phrase."""
        inputs = self.node_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.node_model(**inputs)
        token_embs = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embs.size()).float()
        summed = torch.sum(token_embs * mask, 1)
        count = torch.clamp(mask.sum(1), min=1e-9)
        return (summed / count).squeeze(0).cpu()

    def word_cosine_similarity(self, word1: str, word2: str) -> float:
        """Word-level cosine similarity using BanglaBERT (for soft entity matching)."""
        emb1 = self.get_embedding(word1).unsqueeze(0)
        emb2 = self.get_embedding(word2).unsqueeze(0)
        return F.cosine_similarity(emb1, emb2).item()

    def generate_node_embeddings(self, graph) -> dict:
        """Return a {node: tensor} dict with BanglaBERT embeddings for all KG nodes."""
        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = (
                self.get_embedding(node) if node != "[NONE]" else torch.zeros(768)
            )
        return embeddings
