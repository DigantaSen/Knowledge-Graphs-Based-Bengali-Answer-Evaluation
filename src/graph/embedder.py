"""
embedder.py
===========
LaBSE embedder for sentences, words and graph nodes, plus BanglaBERT
sentence embeddings for the BanglaBERT baseline.

LaBSE is used for words and nodes because BanglaBERT (an ELECTRA
discriminator) gives every pair of Bengali words a cosine similarity above
0.82, so its vectors cannot tell entities apart.  LaBSE separates them
(different entities ≈ 0.3–0.6, synonyms ≈ 0.85–0.98).

Offline-first loading: tries the local HuggingFace cache first, then
falls back to downloading if the cache misses.  Set the environment
variable HF_HUB_OFFLINE=1 to force cache-only mode system-wide.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from src.config import BANGLA_BERT_MODEL, LABSE_MODEL

try:
    # BanglaBERT's model card requires this normaliser before tokenisation
    from normalizer import normalize as _banglabert_normalize
except ImportError:
    _banglabert_normalize = None

EMB_DIM = 768


def _load_sentence_transformer(model_name: str, device: str) -> Optional[SentenceTransformer]:
    """
    Load a SentenceTransformer model.
    Tries local cache first (local_files_only=True) to avoid network calls.
    Falls back to a download attempt if the cache misses.
    """
    # 1) Try from local cache (fast, no network)
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
        print(f'  [OK] {model_name} loaded from cache.')
        return model
    except Exception:
        pass

    # 2) Try downloading (needs internet)
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f'  [OK] {model_name} downloaded and loaded.')
        return model
    except Exception as e:
        print(f'  [FAIL] {model_name}: {e}')
        return None


def _load_hf_model(model_name: str, device: torch.device) -> Tuple[Optional[object], Optional[object]]:
    """
    Load a HuggingFace tokenizer + model.
    Tries local cache first, then falls back to download.
    """
    for local_only in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=local_only)
            model = AutoModel.from_pretrained(
                model_name, local_files_only=local_only)
            model.to(device)
            model.eval()
            source = 'cache' if local_only else 'download'
            print(f'  [OK] {model_name} loaded from {source}.')
            return tokenizer, model
        except Exception as e:
            if local_only:
                continue          # try the download fallback
            print(f'  [FAIL] {model_name}: {e}')
            return None, None
    return None, None


class BanglaBERTEmbedder:
    """
    * LaBSE: sentence similarity, word similarity and graph-node embeddings.
    * BanglaBERT: mean-pooled sentence embeddings (BanglaBERT baseline only).

    All embeddings are cached per text, so repeated nodes and sentences
    (very common in these datasets) are encoded once.
    """

    def __init__(self,
                 sentence_model_name: str = LABSE_MODEL,
                 node_model_name:     str = BANGLA_BERT_MODEL):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._labse_cache = {}
        self._bb_cache = {}

        # ── LaBSE ──
        print(f'Loading LaBSE: {sentence_model_name}...')
        self.sentence_model = _load_sentence_transformer(
            sentence_model_name, str(self.device))

        # ── BanglaBERT ──
        print(f'Loading BanglaBERT: {node_model_name}...')
        self.node_tokenizer, self.node_model = _load_hf_model(
            node_model_name, self.device)
        if _banglabert_normalize is None:
            print('  [WARN] csebuetnlp normalizer not installed; BanglaBERT input is not '
                  'normalised (pip install git+https://github.com/csebuetnlp/normalizer).')

    # ─── LaBSE ────────────────────────────────────────────────────────────────

    def encode_sentences(self, texts: list) -> np.ndarray:
        """Return unit-norm LaBSE embeddings [len(texts), 768] (cached)."""
        texts = [str(t) for t in texts]
        missing = list(dict.fromkeys(t for t in texts if t not in self._labse_cache))
        if missing:
            if self.sentence_model is None:
                embs = np.zeros((len(missing), EMB_DIM), dtype=np.float32)
            else:
                embs = self.sentence_model.encode(missing, convert_to_numpy=True,
                                                  normalize_embeddings=True,
                                                  batch_size=64, show_progress_bar=False)
            self._labse_cache.update(zip(missing, embs))
        if not texts:
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        return np.stack([self._labse_cache[t] for t in texts])

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Return LaBSE cosine similarity between *text1* and *text2*."""
        if self.sentence_model is None:
            return 0.0
        e = self.encode_sentences([text1, text2])
        return float(e[0] @ e[1])

    def word_cosine_similarity(self, word1: str, word2: str) -> float:
        """Return LaBSE cosine similarity between two words / short phrases."""
        return self.cosine_similarity(word1, word2)

    def generate_node_embeddings(self, graph) -> dict:
        """Return {node_id: {'emb': tensor[768], 'node_type': str}} using LaBSE on node labels."""
        nodes = list(graph.nodes())
        labels = [graph.nodes[n].get('label', n) for n in nodes]
        embs = self.encode_sentences(labels)
        return {n: {'emb': torch.from_numpy(np.array(e, dtype=np.float32)),
                    'node_type': graph.nodes[n].get('node_type', 'entity')}
                for n, e in zip(nodes, embs)}

    # ─── BanglaBERT (baseline) ────────────────────────────────────────────────

    def get_embedding(self, text: str) -> torch.Tensor:
        """Return a mean-pooled BanglaBERT embedding for *text* (shape: [768])."""
        if self.node_model is None:
            return torch.zeros(EMB_DIM)
        if text in self._bb_cache:
            return self._bb_cache[text]
        model_text = _banglabert_normalize(text) if _banglabert_normalize else text
        inputs = self.node_tokenizer(model_text, return_tensors='pt', truncation=True,
                                     padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.node_model(**inputs)
        token_embs  = outputs.last_hidden_state
        attn_mask   = inputs['attention_mask']
        mask_exp    = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
        sum_emb     = torch.sum(token_embs * mask_exp, 1)
        sum_mask    = torch.clamp(mask_exp.sum(1), min=1e-9)
        emb = (sum_emb / sum_mask).squeeze(0).cpu()
        self._bb_cache[text] = emb
        return emb

    def banglabert_similarity(self, text1: str, text2: str) -> float:
        e1, e2 = self.get_embedding(text1), self.get_embedding(text2)
        return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
