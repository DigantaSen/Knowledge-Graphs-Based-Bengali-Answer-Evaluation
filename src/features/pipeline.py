"""
pipeline.py
===========
Main feature extraction pipeline, GNN training, and GNN inference.

``extract_pair_features`` is the single definition of the scorer's inputs;
training, evaluation and the demo all call it so their features agree.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from src.config import GNN_BATCH_SIZE, RANDOM_STATE, SYNONYM_THRESHOLD
from src.features.answer_slot import answer_slot_features
from src.models.siamese_gnn import SiameseGNN, nx_to_pyg_data
from src.utils.helpers import (
    compute_rouge_l, is_bengali_numeral, is_multi_sentence, negation_mismatch,
    sentence_coverage_score, set_seed, wrong_number,
)

# Order of the scorer's base features (see AnswerScorer.FEATURE_NAMES)
SCORER_FEATURES = ('base_sim', 'graph_sim', 'penalty', 'entity_mm', 'coverage', 'neg_mm', 'wrong_num',
                   'ans_recall', 'ans_precision', 'key_match')
_SCALAR_FEATURES = ('base_sim', 'penalty', 'entity_mm', 'coverage', 'neg_mm', 'wrong_num',
                    'ans_recall', 'ans_precision', 'key_match', 'rouge_l')


def compute_entity_mismatch(ref_parsed, stu_parsed, ref_text: str, stu_text: str,
                             embedder_obj=None, ner_pipeline_obj=None) -> float:
    """
    Compute a numeric entity mismatch score in [0, 1].

    Combines NER-detected entities and dependency-parsed content words.
    A missing reference word counts as present when the student added a
    word with LaBSE similarity >= SYNONYM_THRESHOLD; numbers must match exactly.
    """
    ref_word_set = set(ref_text.replace('।', '').replace(',', '').split())
    stu_word_set = set(stu_text.replace('।', '').replace(',', '').split())
    if ref_word_set == stu_word_set:
        return 0.0

    ref_entities, stu_entities = set(), set()
    if ner_pipeline_obj is not None:
        try:
            ref_ner = ner_pipeline_obj(ref_text)
            stu_ner = ner_pipeline_obj(stu_text)
            ref_entities = {e['word'].replace('##', '') for e in ref_ner}
            stu_entities = {e['word'].replace('##', '') for e in stu_ner}
        except Exception:
            pass

    content_pos  = {'NOUN', 'PROPN', 'NUM', 'ADJ'}
    ref_content, stu_content = set(), set()
    for sent in ref_parsed:
        for word in sent:
            if word.get('upos', 'X') in content_pos and word['deprel'] != 'punct':
                ref_content.add(word.get('lemma') or word['text'])
    for sent in stu_parsed:
        for word in sent:
            if word.get('upos', 'X') in content_pos and word['deprel'] != 'punct':
                stu_content.add(word.get('lemma') or word['text'])

    ref_content |= ref_entities
    stu_content |= stu_entities

    if not ref_content:
        overlap = len(ref_word_set & stu_word_set) / max(len(ref_word_set), 1)
        return max(0.0, 1.0 - overlap)

    missing = ref_content - stu_content
    added   = stu_content - ref_content
    truly_missing = set()
    for ref_word in missing:
        found = False
        if embedder_obj is not None and added and not is_bengali_numeral(ref_word):
            for stu_word in added:
                if is_bengali_numeral(stu_word):
                    continue
                try:
                    if embedder_obj.word_cosine_similarity(ref_word, stu_word) >= SYNONYM_THRESHOLD:
                        found = True
                        break
                except Exception:
                    pass
        if not found:
            truly_missing.add(ref_word)

    missing_ratio        = len(truly_missing) / len(ref_content)
    substitution_penalty = 0.2 if truly_missing and added else 0.0
    return min(1.0, missing_ratio + substitution_penalty)


def extract_pair_features(question: str, ref_text: str, stu_text: str, comps: dict) -> dict:
    """
    Compute every feature for one (question, reference, student) triple.

    *comps* holds the NLP components: text_processor, dep_parser,
    triple_extractor, kg_constructor, embedder, validator, and optionally
    coref_resolver and ner.

    Returns a dict with the scalar features in SCORER_FEATURES (except
    graph_sim, which needs a trained GNN), plus rouge_l, multi_sentence,
    karak_explanations, the PyG graphs (ref_pyg, stu_pyg) and the NetworkX
    graphs (ref_kg, stu_kg).
    """
    tp, dp = comps['text_processor'], comps['dep_parser']
    te, kgc = comps['triple_extractor'], comps['kg_constructor']
    emb, val = comps['embedder'], comps['validator']
    cr, ner = comps.get('coref_resolver'), comps.get('ner')

    # Both sides are resolved so a verbatim copy stays a verbatim copy
    if cr:
        ref_res = cr.resolve(ref_text, question, '')
        stu_res = cr.resolve(stu_text, question, ref_text)
    else:
        ref_res, stu_res = ref_text, stu_text

    ref_parsed = dp.parse(tp.normalize(ref_res))
    stu_parsed = dp.parse(tp.normalize(stu_res))
    ref_kg = kgc.build_graph(te.extract_triples(raw_text=ref_res))
    stu_kg = kgc.build_graph(te.extract_triples(raw_text=stu_res))
    penalty, explanations = val.validate(ref_parsed, stu_parsed, ref_raw=ref_res, stu_raw=stu_res)

    return {
        # Answer-slot features: compare only the content that answers the question
        **answer_slot_features(question, ref_res, stu_res, comps),
        'base_sim':  emb.cosine_similarity(ref_res, stu_res),
        'penalty':   penalty,
        'entity_mm': compute_entity_mismatch(ref_parsed, stu_parsed, ref_res, stu_res,
                                             embedder_obj=emb, ner_pipeline_obj=ner),
        'coverage':  sentence_coverage_score(ref_res, stu_res, emb),
        'neg_mm':    float(negation_mismatch(ref_res, stu_res)),
        'wrong_num': float(wrong_number(ref_res, stu_res)),
        'rouge_l':   compute_rouge_l(ref_res, stu_res),
        'multi_sentence': is_multi_sentence(ref_text),
        'karak_explanations': explanations,
        'ref_kg': ref_kg, 'stu_kg': stu_kg,
        'ref_pyg': nx_to_pyg_data(ref_kg, emb.generate_node_embeddings(ref_kg)),
        'stu_pyg': nx_to_pyg_data(stu_kg, emb.generate_node_embeddings(stu_kg)),
    }


def create_graphs_and_features(df, comps: dict) -> dict:
    """
    Extract all features for every row in *df*.

    Returns
    -------
    dict of per-row values: float arrays for the scalar features
    ('base_sim', 'penalty', 'entity_mm', 'coverage', 'neg_mm', 'wrong_num',
    'ans_recall', 'ans_precision', 'key_match', 'rouge_l'), a bool array
    'multi_sentence', and lists 'ref_pyg' /
    'stu_pyg' of PyG graphs.
    """
    print(f'Processing {len(df)} samples...')
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Feature extraction'):
        f = extract_pair_features(str(row.get('question', '')), str(row['reference_answer']),
                                  str(row['student_answer']), comps)
        rows.append(f)
    print('Feature extraction complete.')
    out = {k: np.array([r[k] for r in rows], dtype=float) for k in _SCALAR_FEATURES}
    out['multi_sentence'] = np.array([r['multi_sentence'] for r in rows], dtype=bool)
    out['ref_pyg'] = [r['ref_pyg'] for r in rows]
    out['stu_pyg'] = [r['stu_pyg'] for r in rows]
    return out


def subset_features(feats: dict, idx) -> dict:
    """Select rows *idx* from every feature (lists of PyG graphs and arrays)."""
    return {k: ([v[i] for i in idx] if isinstance(v, list) else v[idx]) for k, v in feats.items()}


def concat_features(a: dict, b: dict) -> dict:
    return {k: (a[k] + b[k] if isinstance(a[k], list) else np.concatenate([a[k], b[k]]))
            for k in a}


def scorer_matrix(feats: dict, gsims) -> np.ndarray:
    """Stack the scorer's base features in SCORER_FEATURES order."""
    cols = {**feats, 'graph_sim': np.asarray(gsims, dtype=float)}
    return np.column_stack([cols[k] for k in SCORER_FEATURES])


def _batches(n: int, batch_size: int, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for s in range(0, n, batch_size):
        yield order[s:s + batch_size]


def train_gnn(gnn_model, pyg_ref_data, pyg_stu_data, target_scores, epochs: int = 20,
              seed: int = RANDOM_STATE, batch_size: int = GNN_BATCH_SIZE):
    """
    Train *gnn_model* on (ref_graph, stu_graph) pairs supervised by *target_scores*.

    Mini-batches are reshuffled every epoch from a seeded RNG, so a run is
    reproducible.

    Returns
    -------
    trained gnn_model
    """
    print(f'\nTraining GAT-GNN for {epochs} epochs...')
    set_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    targets = torch.tensor(np.asarray(target_scores, dtype=float) / 100.0, dtype=torch.float32)
    gnn_model.train()

    for ep in range(epochs):
        total_loss = 0.0
        for idx in _batches(len(targets), batch_size, rng):
            optimizer.zero_grad()
            ref_b = Batch.from_data_list([pyg_ref_data[i] for i in idx])
            stu_b = Batch.from_data_list([pyg_stu_data[i] for i in idx])
            loss = criterion(gnn_model(ref_b, stu_b), targets[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
        avg_loss = total_loss / max(len(targets), 1)
        lr = scheduler.get_last_lr()[0]
        print(f'  Epoch {ep+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}')

    return gnn_model


def get_gnn_similarities(gnn_model, pyg_ref_data, pyg_stu_data,
                         batch_size: int = GNN_BATCH_SIZE) -> np.ndarray:
    """Return cosine similarities (clipped at 0) for all (ref, stu) graph pairs."""
    gnn_model.eval()
    out = []
    with torch.no_grad():
        for idx in _batches(len(pyg_ref_data), batch_size):
            ref_b = Batch.from_data_list([pyg_ref_data[i] for i in idx])
            stu_b = Batch.from_data_list([pyg_stu_data[i] for i in idx])
            out.append(gnn_model(ref_b, stu_b).clamp(min=0.0).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def get_oof_gnn_similarities(pyg_ref_data, pyg_stu_data, target_scores, groups,
                             n_splits: int = 5, epochs: int = 20,
                             seed: int = RANDOM_STATE) -> np.ndarray:
    """
    Out-of-fold GNN similarities for training rows.

    The ensemble scorer must be trained on graph_sim values from a GNN that
    did not see those rows.  In-sample similarities are optimistic, so the
    scorer learns to over-trust graph_sim and then meets a different
    distribution at test time.  *groups* keeps duplicated rows in one fold.
    """
    target_scores = np.asarray(target_scores, dtype=float)
    oof = np.zeros(len(target_scores))
    folds = GroupKFold(n_splits=n_splits).split(oof, groups=groups)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f'\n  [OOF {fold+1}/{n_splits}] GNN on {len(tr_idx)} rows -> predict {len(va_idx)}')
        gnn = train_gnn(SiameseGNN(),
                        [pyg_ref_data[i] for i in tr_idx], [pyg_stu_data[i] for i in tr_idx],
                        target_scores[tr_idx], epochs=epochs, seed=seed + fold + 1)
        oof[va_idx] = get_gnn_similarities(gnn, [pyg_ref_data[i] for i in va_idx],
                                           [pyg_stu_data[i] for i in va_idx])
    return oof
