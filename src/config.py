"""
config.py - Global configuration, constants, and dataset path resolution.
"""

import os
import glob

CSV_FILENAME = "bengali_answer_evaluation_dataset.csv"
AUGMENTED_CSV = "bengali_answer_evaluation_dataset_augmented.csv"

GNN_MODEL_PATH = "nlp_dominator_gnn.pth"
SCORER_MODEL_PATH = "nlp_dominator_scorer.pkl"

SENTENCE_MODEL_NAME = "sentence-transformers/LaBSE"
NODE_MODEL_NAME = "csebuetnlp/banglabert"
NER_MODEL_NAME = "Davlan/bert-base-multilingual-cased-ner-hrl"

GNN_EPOCHS = 5
GNN_EPOCHS_CV = 3
TEST_SIZE = 0.2
NUM_ADVERSARIAL = 50
RANDOM_STATE = 42
CV_SPLITS = 5

ENSEMBLE_WEIGHTS = (0.6, 0.4)
SYNONYM_THRESHOLD = 0.75

PLOT_CORRELATION = "correlation_plot.png"
PLOT_CONFUSION = "confusion_matrices.png"
PLOT_ERROR_DIST = "error_distribution.png"
PLOT_SCORE_DIST = "score_distributions.png"
PLOT_ABLATION = "ablation_bar_charts.png"
PLOT_SCATTER = "scatter_comparison.png"
PLOT_SUBJECT = "per_subject_comparison.png"
PLOT_INCREMENTAL = "incremental_ablation.png"


def resolve_dataset_path() -> str:
    """
    Locate the original CSV on both Kaggle and local environments.
    Returns the absolute path to the CSV file.
    Raises FileNotFoundError if the file cannot be found.
    """
    kaggle_candidates = [
        f"/kaggle/input/bengali_answer_evaluation_dataset/{CSV_FILENAME}",
        f"/kaggle/input/bengali-answer-evaluation-dataset/{CSV_FILENAME}",
        f"/kaggle/input/bengalianswerevaluationdataset/{CSV_FILENAME}",
    ]
    for path in kaggle_candidates:
        if os.path.exists(path):
            print(f"✓ Running on Kaggle. Dataset found at: {path}")
            return path

    if os.path.exists("/kaggle/input"):
        matches = glob.glob(f"/kaggle/input/**/{CSV_FILENAME}", recursive=True)
        if not matches:
            matches = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        if matches:
            print(f"✓ Running on Kaggle. Dataset found at: {matches[0]}")
            return matches[0]

    if os.path.exists(CSV_FILENAME):
        print(f"✓ Running locally. Dataset found at: {CSV_FILENAME}")
        return CSV_FILENAME

    if os.path.exists("/kaggle/input"):
        print("\n⚠ Dataset NOT found. Contents of /kaggle/input/:")
        for item in sorted(os.listdir("/kaggle/input")):
            sub = os.path.join("/kaggle/input", item)
            print(f"  📁 {item}/")
            if os.path.isdir(sub):
                for f in sorted(os.listdir(sub)):
                    print(f"      📄 {f}")

    raise FileNotFoundError(
        f"'{CSV_FILENAME}' not found. "
        "Add the dataset via 'Add Data' in Kaggle's right sidebar, "
        "or place the CSV in the working directory."
    )
