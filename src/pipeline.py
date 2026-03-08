"""
pipeline.py - Feature extraction pipeline, GNN training helpers, and metric utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

from src.config import SYNONYM_THRESHOLD
from src.gnn_model import nx_to_pyg_data

# ── entity mismatch ────────────────────────────────────────────────────────────

def compute_entity_mismatch(
    ref_parsed, stu_parsed,
    ref_text: str, stu_text: str,
    embedder_obj=None,
    ner_pipeline_obj=None,
) -> float:
    """Compute a soft entity-mismatch penalty in [0.0, 1.0].

    Improvements over a plain set-difference:
      1. Uses NER to identify named entities.
      2. Uses BanglaBERT cosine similarity to treat near-synonyms as matches.
      3. Only penalises genuine entity substitutions, not word-order changes.
    """
    ref_words = set(ref_text.replace("।", "").replace(",", "").split())
    stu_words = set(stu_text.replace("।", "").replace(",", "").split())
    if ref_words == stu_words:
        return 0.0

    # Step 1: NER-based entity sets
    ref_entities, stu_entities = set(), set()
    if ner_pipeline_obj is not None:
        try:
            for ent in ner_pipeline_obj(ref_text):
                ref_entities.add(ent["word"].replace("##", ""))
            for ent in ner_pipeline_obj(stu_text):
                stu_entities.add(ent["word"].replace("##", ""))
        except Exception:
            pass

    # Step 2: POS-based content words
    content_pos = {"NOUN", "PROPN", "NUM", "ADJ"}
    ref_content, stu_content = set(), set()

    for sent in ref_parsed:
        for word in sent:
            if word.get("upos", "X") in content_pos and word["deprel"] != "punct":
                ref_content.add(word["lemma"] if word["lemma"] else word["text"])

    for sent in stu_parsed:
        for word in sent:
            if word.get("upos", "X") in content_pos and word["deprel"] != "punct":
                stu_content.add(word["lemma"] if word["lemma"] else word["text"])

    ref_content |= ref_entities
    stu_content |= stu_entities

    if not ref_content:
        return max(0.0, 1.0 - len(ref_words & stu_words) / len(ref_words)) if ref_words else 0.0

    # Step 3: Soft matching
    missing = ref_content - stu_content
    added = stu_content - ref_content
    truly_missing = set()

    for ref_word in missing:
        found_synonym = False
        if embedder_obj is not None and added:
            for stu_word in added:
                try:
                    if embedder_obj.word_cosine_similarity(ref_word, stu_word) >= SYNONYM_THRESHOLD:
                        found_synonym = True
                        break
                except Exception:
                    pass
        if not found_synonym:
            truly_missing.add(ref_word)

    missing_ratio = len(truly_missing) / len(ref_content)
    substitution_penalty = 0.2 if (truly_missing and added) else 0.0
    return min(1.0, missing_ratio + substitution_penalty)

# ── feature extraction ─────────────────────────────────────────────────────────

def create_graphs_and_features(
    df,
    text_processor,
    dep_parser,
    triple_extractor,
    kg_constructor,
    embedder,
    validator,
    coref_resolver=None,
    ner_pipeline_obj=None,
):
    """Process dataframe rows → extract all features needed by the scorer.

    Returns
    -------
    baseline_sims, pyg_ref_data, pyg_stu_data, karak_penalties, entity_mismatches
    """
    print(f"Processing {len(df)} samples through the NLP pipeline...")

    baseline_sims, pyg_ref, pyg_stu, penalties, entity_mms = [], [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
        ref_text = str(row["reference_answer"])
        stu_text = str(row["student_answer"])
        question_text = str(row.get("question", ""))

        # 0. Coreference resolution
        if coref_resolver is not None:
            stu_text = coref_resolver.resolve(stu_text, question_text, ref_text)

        # 1. Sentence-level similarity (LaBSE)
        baseline_sims.append(embedder.cosine_similarity(ref_text, stu_text))

        # 2. Dependency parsing
        ref_parsed = dep_parser.parse(text_processor.normalize(ref_text))
        stu_parsed = dep_parser.parse(text_processor.normalize(stu_text))

        # 3. Triple extraction
        ref_triples = triple_extractor.extract_triples(ref_parsed, raw_text=ref_text)
        stu_triples = triple_extractor.extract_triples(stu_parsed, raw_text=stu_text)

        # 4. Knowledge Graph construction
        ref_kg = kg_constructor.build_graph(ref_triples)
        stu_kg = kg_constructor.build_graph(stu_triples)

        # 5. Node embeddings
        ref_emb = embedder.generate_node_embeddings(ref_kg)
        stu_emb = embedder.generate_node_embeddings(stu_kg)

        # 6. PyG conversion
        pyg_ref.append(nx_to_pyg_data(ref_kg, ref_emb))
        pyg_stu.append(nx_to_pyg_data(stu_kg, stu_emb))

        # 7. Karak validation
        penalty, _ = validator.validate(ref_parsed, stu_parsed)
        penalties.append(penalty)

        # 8. Soft entity mismatch
        entity_mms.append(
            compute_entity_mismatch(
                ref_parsed, stu_parsed, ref_text, stu_text,
                embedder_obj=embedder, ner_pipeline_obj=ner_pipeline_obj,
            )
        )

    print("Feature extraction complete.")
    return baseline_sims, pyg_ref, pyg_stu, penalties, entity_mms

# ── GNN training / inference ──────────────────────────────────────────────────

def train_gnn(gnn_model, pyg_ref_data, pyg_stu_data, target_scores, epochs: int = 5):
    """Train the Siamese GAT to learn structural similarity."""
    print(f"\nTraining GAT-GNN for {epochs} epochs...")
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    gnn_model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for ref_g, stu_g, score in zip(pyg_ref_data, pyg_stu_data, target_scores):
            optimizer.zero_grad()
            pred_sim = gnn_model(ref_g, stu_g)
            target = torch.tensor([score / 100.0], dtype=torch.float32)
            loss = criterion(pred_sim, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(
            f"  Epoch {ep+1}/{epochs} - Avg Loss: {total_loss/len(target_scores):.4f}"
            f" - LR: {scheduler.get_last_lr()[0]:.6f}"
        )
    return gnn_model

def get_gnn_similarities(gnn_model, pyg_ref_data, pyg_stu_data) -> list:
    """Return GAT-GNN similarity scores for every sample pair."""
    gnn_model.eval()
    sims = []
    with torch.no_grad():
        for ref_g, stu_g in zip(pyg_ref_data, pyg_stu_data):
            sims.append(max(0.0, gnn_model(ref_g, stu_g).item()))
    return sims

# ── metrics / plotting ────────────────────────────────────────────────────────

def evaluate_metrics(y_true, y_pred, name: str):
    """Print Pearson, Spearman, MAE and return them."""
    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"  Pearson:  {p_corr:.4f}")
    print(f"  Spearman: {s_corr:.4f}")
    print(f"  MAE:      {mae:.2f}")
    return p_corr, s_corr, mae

def plot_correlation(y_true, predictions_dict: dict, title: str, filename: str):
    """Plot predicted vs human scores; save to *filename*."""
    plt.figure(figsize=(10, 6))
    for label, y_pred in predictions_dict.items():
        sns.regplot(
            x=np.array(y_true), y=np.array(y_pred),
            label=label, scatter_kws={"alpha": 0.4, "s": 20},
        )
    plt.xlabel("Human Score")
    plt.ylabel("AI Predicted Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved to {filename}")
