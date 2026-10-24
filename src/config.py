"""
config.py
=========
Global configuration constants for the Bengali Answer Evaluation System.
Centralises all hardcoded paths, model names, and hyper-parameters so that
any future changes only need to be made here.
"""

import os

# ─── Environment detection ────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")

# ─── Base data directory ──────────────────────────────────────────────────────
# Locally datasets live in  <project_root>/data/
# On Kaggle they come from  /kaggle/input/<dataset-name>/
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ─── Dataset paths ────────────────────────────────────────────────────────────
if IS_KAGGLE:
    # Look through typical dataset version directories
    DATASET_CSV_CANDIDATES = [
        "/kaggle/input/nlp-dominator-dataset/dataset_single_sentence.csv",
        "/kaggle/input/nlp-dominator-dataset/dataset_medium_sentences.csv",
        "/kaggle/input/nlp-dominator-dataset/dataset_long_sentences.csv",
        "/kaggle/input/nlp-dominator-dataset/train_data.csv",
    ]
else:
    DATASET_CSV_CANDIDATES = [
        os.path.join(DATA_DIR, "dataset_single_sentence.csv"),
        os.path.join(DATA_DIR, "dataset_medium_sentences.csv"),
        os.path.join(DATA_DIR, "dataset_long_sentences.csv"),
        os.path.join(DATA_DIR, "train_data.csv"),
    ]

# ─── Adversarial augmentation ─────────────────────────────────────────────────
NUM_ADVERSARIAL = 51              # must equal len(_ADVERSARIAL_SAMPLES) in src/data/dataset.py

# ─── Data cleaning / synthetic training errors ────────────────────────────────
# Identical (question, reference, student) rows are merged and their human
# scores averaged: the datasets repeat many pairs with conflicting scores.
DEDUPLICATE_PAIRS = True

# Rule-generated wrong answers (negation, role swap, wrong number) added to
# the TRAINING split only, so the scorer sees these error types.  Built from
# training reference sentences; sentences used by the hand-written
# adversarial test samples are excluded.
SYNTHETIC_ERRORS     = True
SYNTHETIC_MAX        = 150        # max perturbed rows per training set
SYNTHETIC_FRACTION   = 0.3        # synthetic rows (incl. copies) <= 30% of real training rows

# ─── Model identifiers ────────────────────────────────────────────────────────
BANGLA_BERT_MODEL = "csebuetnlp/banglabert"
LABSE_MODEL       = "sentence-transformers/LaBSE"
NER_MODEL         = "Davlan/bert-base-multilingual-cased-ner-hrl"

# Two words count as synonyms when their LaBSE cosine similarity is at least
# this.  Measured: true synonyms 0.84–0.98, different entities 0.31–0.83
# (BanglaBERT put every word pair above 0.82, so it cannot be used for this).
SYNONYM_THRESHOLD = 0.85

# ─── Saved model files ────────────────────────────────────────────────────────
# Models are saved per dataset as <models_dir>/<filename>, e.g.
#   models/nlp_dominator_gnn_dataset_single_sentence.pth
# The scorer is stored as a plain .npz (no pickle), so loading a downloaded
# file cannot execute code.
MODELS_DIR = "models"


def gnn_model_filename(dataset_stem: str) -> str:
    return f"nlp_dominator_gnn_{dataset_stem}.pth"


def scorer_model_filename(dataset_stem: str) -> str:
    return f"nlp_dominator_scorer_{dataset_stem}.npz"


# ─── Training hyper-parameters ────────────────────────────────────────────────
GNN_EPOCHS        = 5
GNN_BATCH_SIZE    = 32
RANDOM_STATE      = 42
TEST_SIZE         = 0.15          # fraction of split groups held out for test
CV_N_SPLITS       = 5             # folds for cross-validation / out-of-fold predictions
BOOTSTRAP_ROUNDS  = 2000          # resamples for confidence intervals

# Rows sharing a group never straddle a train/test (or CV) boundary.
#   "pair"     – identical (reference, student) pairs; the datasets repeat many
#                pairs, so a plain row split leaks copies into the test set.
#   "question" – all answers to a question (stricter unseen-question setting).
SPLIT_GROUP_BY    = "pair"

# ─── Scoring thresholds ───────────────────────────────────────────────────────
PASS_THRESHOLD    = 50            # scores >= this are "pass" for F1 computation
# Hard caps are fixed a-priori rules, not tuned values.  They apply only when
# the reference is a single sentence: there a negation or a wrong number makes
# the whole answer wrong, whereas in multi-sentence answers human graders give
# partial credit, so the learned scorer handles those cases from features.
HARD_NEG_CAP      = 30.0          # max score when negation mismatch detected
HARD_NUM_CAP      = 25.0          # max score when numeric error detected
