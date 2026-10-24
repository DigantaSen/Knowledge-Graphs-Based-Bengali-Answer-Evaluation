"""
experiment.py
=============
Steps shared by train.py, evaluate.py and demo.py, so all of them build the
same components, the same training set, and the same predictions.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import (
    CV_N_SPLITS, NER_MODEL, NUM_ADVERSARIAL, RANDOM_STATE, SYNTHETIC_ERRORS, SYNTHETIC_FRACTION,
    SYNTHETIC_MAX,
    TEST_SIZE,
)
from src.data.dataset import (
    find_dataset_csv, is_templated, load_and_augment_dataset, split_dataset,
)
from src.data.synthetic import make_synthetic_examples
from src.evaluation.metrics import compute_full_metrics, paired_bootstrap, print_regression_table
from src.evaluation.overrides import apply_hard_overrides
from src.features.pipeline import (
    get_gnn_similarities, get_oof_gnn_similarities, scorer_matrix, train_gnn,
)
from src.graph.embedder import BanglaBERTEmbedder
from src.graph.kg_constructor import KnowledgeGraphConstructor
from src.models.answer_scorer import AnswerScorer
from src.models.siamese_gnn import SiameseGNN
from src.nlp.coreference import BengaliCoreferenceResolver
from src.nlp.dependency_parser import BengaliDependencyParser
from src.nlp.triple_extractor import TripleExtractor
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.validation.karak_validator import KarakValidator


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_split(dataset_path, group_by: str):
    """
    Locate / augment the dataset and split it.

    Each dataset gets its own '<stem>_augmented.csv' next to the source file,
    so switching datasets never reuses stale augmented data.

    Returns (original_csv, stem, augmented_csv, df_augmented, train_df, test_df).
    """
    original_csv  = dataset_path or find_dataset_csv()
    stem          = os.path.splitext(os.path.basename(original_csv))[0]
    augmented_csv = os.path.join(os.path.dirname(os.path.abspath(original_csv)),
                                 f'{stem}_augmented.csv')
    if not os.path.exists(augmented_csv):
        print(f'[INFO] Generating augmented dataset from: {original_csv}')
        df_augmented = load_and_augment_dataset(original_csv, augmented_csv)
    else:
        df_augmented = pd.read_csv(augmented_csv)
        print(f'[INFO] Loaded augmented dataset: {augmented_csv} ({len(df_augmented)} rows)')

    train_df, test_df = split_dataset(df_augmented, num_adversarial=NUM_ADVERSARIAL,
                                      test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                      group_by=group_by)
    print(f'  Train: {len(train_df)} | Test: {len(test_df)} ({NUM_ADVERSARIAL} adversarial)'
          f'  [deduplicated, grouped by {group_by}]')
    return original_csv, stem, augmented_csv, df_augmented, train_df, test_df


def with_synthetic(df: pd.DataFrame, dep_parser, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return *df* plus synthetic wrong answers built from its reference sentences."""
    if not SYNTHETIC_ERRORS:
        return df
    syn = make_synthetic_examples(df, dep_parser, max_rows=SYNTHETIC_MAX, random_state=seed,
                                  max_total=int(SYNTHETIC_FRACTION * len(df)))
    print(f'  Synthetic training rows: {len(syn)} {syn["synthetic"].value_counts().to_dict()}')
    return pd.concat([df, syn], ignore_index=True)


# ─── Components ───────────────────────────────────────────────────────────────

def load_ner():
    """Load the NER pipeline (cache first; --offline restricts it to the cache)."""
    from transformers import pipeline as hf_pipeline
    try:
        ner = hf_pipeline('ner', model=NER_MODEL, aggregation_strategy='simple', device=-1)
        print('  [OK] NER pipeline loaded.')
        return ner
    except Exception as e:
        print(f'  [WARN] NER unavailable: {e}')
        return None


def build_components() -> dict:
    """Instantiate all NLP and graph components."""
    triple_extractor = TripleExtractor()
    return {
        'text_processor':   TextPreprocessor(),
        'dep_parser':       BengaliDependencyParser(),
        'coref_resolver':   BengaliCoreferenceResolver(pos_tagger=triple_extractor.pos_tagger),
        'triple_extractor': triple_extractor,
        'kg_constructor':   KnowledgeGraphConstructor(),
        'embedder':         BanglaBERTEmbedder(),
        'validator':        KarakValidator(),
        'ner':              load_ner(),
    }


# ─── Models ───────────────────────────────────────────────────────────────────

def fit_gnn_and_scorer(feats: dict, y, groups, epochs: int, seed: int = RANDOM_STATE):
    """
    Train the GNN on all of *feats*, and the scorer on out-of-fold GNN
    similarities (so the scorer sees unseen-row values, as on test data).

    Returns (gnn, scorer, oof_gnn_sims).
    """
    gnn = train_gnn(SiameseGNN(), feats['ref_pyg'], feats['stu_pyg'], y, epochs=epochs, seed=seed)
    print(f'\n  Computing out-of-fold GNN similarities ({CV_N_SPLITS} folds)...')
    oof = get_oof_gnn_similarities(feats['ref_pyg'], feats['stu_pyg'], y, groups=groups,
                                   n_splits=CV_N_SPLITS, epochs=epochs, seed=seed)
    scorer = AnswerScorer()
    scorer.train(scorer_matrix(feats, oof), y, groups=groups)
    return gnn, scorer, oof


def predict(gnn, scorer, feats: dict):
    """Return (gnn_sims, raw_scores, final_scores, override_log) for *feats*."""
    gsims = get_gnn_similarities(gnn, feats['ref_pyg'], feats['stu_pyg'])
    raw = scorer.predict_batch(scorer_matrix(feats, gsims))
    final, log = apply_hard_overrides(raw, feats['neg_mm'], feats['wrong_num'], feats['multi_sentence'])
    return gsims, raw, final, log


def calibrated(cols_tr, y_tr, cols_te) -> np.ndarray:
    """
    Linear map from raw similarity feature(s) to the 0–100 score scale, fit on
    the training split.  Raw ``cosine × 100`` is not calibrated to human
    scores, which made the baselines look far worse than a fair comparison.
    """
    lr = LinearRegression().fit(np.column_stack(cols_tr), y_tr)
    return np.clip(lr.predict(np.column_stack(cols_te)), 0, 100)


# ─── Reporting ────────────────────────────────────────────────────────────────

def report(saver, predictions: list, y_test, test_df: pd.DataFrame, phrases: list,
           proposed: str):
    """
    Print and register metrics for the full test set and its subsets, plus
    bootstrap significance of *proposed* against the best baseline.

    *predictions* is [(model_name, scores)], baselines first.  Adversarial
    rows are the last NUM_ADVERSARIAL rows of *test_df*.
    """
    n_normal  = len(test_df) - NUM_ADVERSARIAL
    templated = is_templated(test_df['student_answer'], phrases)
    test_df['templated'] = templated
    subsets = {
        'normal':       ('Normal test rows (adversarial excluded)', np.arange(n_normal)),
        'nontemplated': ('Normal, non-templated student answers',
                         np.where(~templated[:n_normal])[0]),
        'adversarial':  ('Hand-written adversarial rows', np.arange(n_normal, len(test_df))),
    }
    if 'source' in test_df.columns:
        # Datasets that mix the original rows with the clean synthetic rows
        source = test_df['source'].astype(str).values[:n_normal]
        subsets['original'] = ('Normal rows from the original dataset', np.where(source == 'original')[0])
        subsets['clean'] = ('Normal rows from the clean synthetic dataset',
                            np.where(source == 'clean_synthetic')[0])

    all_metrics = [compute_full_metrics(y_test, p, name) for name, p in predictions]
    print_regression_table(all_metrics, title='Test set, normal + adversarial')
    saver.add_metrics(all_metrics)

    pred = dict(predictions)
    significance = []
    for label, (description, idx) in subsets.items():
        if len(idx) < 3:
            continue
        metrics = [compute_full_metrics(y_test[idx], p[idx], name) for name, p in predictions]
        print_regression_table(metrics, title=description)
        saver.add_subset_metrics(label, metrics, description)
        if label == 'adversarial':
            continue
        baselines = [m for m in metrics if m['name'].lower().startswith('baseline')]
        if not baselines:
            continue
        best = min(baselines, key=lambda m: m['mae'])['name']
        stats = paired_bootstrap(y_test[idx], pred[proposed][idx], pred[best][idx])
        for metric, s in stats.items():
            significance.append({'subset': label, 'model': proposed, 'baseline': best,
                                 'metric': metric, **s})
            print(f'  [{label}] {metric}: {proposed} {s["value"]:.3f} '
                  f'[{s["ci_low"]:.3f}, {s["ci_high"]:.3f}]  vs {best}: '
                  f'diff {s["diff"]:+.3f} [{s["diff_ci_low"]:+.3f}, {s["diff_ci_high"]:+.3f}] '
                  f'p={s["p_value"]:.3f}')
    saver.add_significance(significance)
    return all_metrics
