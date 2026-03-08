"""
inference.py - Load pre-trained models and run single-sample grading.
"""

import os
import torch

from src.config import GNN_MODEL_PATH, SCORER_MODEL_PATH
from src.gnn_model import SiameseGNN, nx_to_pyg_data
from src.scorer import AnswerScorer

def load_pretrained_models(gnn_path: str = GNN_MODEL_PATH,
                            scorer_path: str = SCORER_MODEL_PATH):
    """Load saved GAT-GNN and XGBoost+MLP ensemble scorer.

    Returns
    -------
    (SiameseGNN, AnswerScorer) if both files exist, else (None, None).
    """
    if not os.path.exists(gnn_path) or not os.path.exists(scorer_path):
        print("⚠ Model files not found. Run main.py (training pipeline) first.")
        return None, None

    print("Loading pre-trained models...")
    gnn = SiameseGNN()
    gnn.load_state_dict(torch.load(gnn_path, map_location="cpu", weights_only=True))
    gnn.eval()

    scorer = AnswerScorer()
    scorer.load_model(scorer_path)

    print("✅ GAT-GNN + XGBoost+MLP Ensemble loaded and ready!")
    return gnn, scorer

def grade_single(
    question: str,
    reference_answer: str,
    student_answer: str,
    gnn_model: SiameseGNN,
    scorer: AnswerScorer,
    text_processor,
    dep_parser,
    triple_extractor,
    kg_constructor,
    embedder,
    validator,
    coref_resolver=None,
    ner_pipeline_obj=None,
) -> dict:
    """Grade a single (question, reference, student) triplet.

    Returns a dict with keys:
        score, baseline_sim, graph_sim, penalty, entity_mismatch,
        explanations, ref_triples, stu_triples
    """
    # Coreference resolution
    if coref_resolver is not None:
        student_resolved = coref_resolver.resolve(student_answer, question, reference_answer)
    else:
        student_resolved = student_answer

    # Sentence-level similarity
    baseline_sim = embedder.cosine_similarity(reference_answer, student_resolved)

    # Dependency parsing
    ref_parsed = dep_parser.parse(text_processor.normalize(reference_answer))
    stu_parsed = dep_parser.parse(text_processor.normalize(student_resolved))

    # Triple extraction
    ref_triples = triple_extractor.extract_triples(ref_parsed, raw_text=reference_answer)
    stu_triples = triple_extractor.extract_triples(stu_parsed, raw_text=student_resolved)

    # Knowledge Graphs
    ref_kg = kg_constructor.build_graph(ref_triples)
    stu_kg = kg_constructor.build_graph(stu_triples)

    # Node embeddings + PyG conversion
    ref_pyg = nx_to_pyg_data(ref_kg, embedder.generate_node_embeddings(ref_kg))
    stu_pyg = nx_to_pyg_data(stu_kg, embedder.generate_node_embeddings(stu_kg))

    # GAT-GNN similarity
    gnn_model.eval()
    with torch.no_grad():
        graph_sim = max(0.0, gnn_model(ref_pyg, stu_pyg).item())

    # Karak validation
    penalty, explanations = validator.validate(ref_parsed, stu_parsed)

    # Soft entity mismatch
    from src.pipeline import compute_entity_mismatch
    entity_mismatch = compute_entity_mismatch(
        ref_parsed, stu_parsed, reference_answer, student_resolved,
        embedder_obj=embedder, ner_pipeline_obj=ner_pipeline_obj,
    )

    # Ensemble score
    score = scorer.predict_score(baseline_sim, graph_sim, penalty, entity_mismatch)

    return dict(
        score=score,
        baseline_sim=baseline_sim,
        graph_sim=graph_sim,
        penalty=penalty,
        entity_mismatch=entity_mismatch,
        explanations=explanations,
        ref_triples=ref_triples,
        stu_triples=stu_triples,
        student_resolved=student_resolved,
    )
