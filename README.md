# Knowledge Graphs Based Bengali Answer Evaluation

**Course Project — Introduction to Natural Language Processing (INLP)**  
School of Computer Science & Engineering  

**Team:** NLP Dominator

---

## Team

| Name | Roll Number |
|------|-------------|
| Diganta Sen | 2025201050 |
| Sourav Nath | 2025201054 |
| Nilkanta Karak | 2025201031 |
| Kushal Mukherjee | 2025201072 |

---

## About the Project

An automated short-answer grading system for Bengali (বাংলা) that scores student answers against teacher reference answers on a **0–100 scale**.

The system is designed to tackle the unique challenges of Bengali NLP — agglutinative morphology, free word order, and Karak (কারক) semantic roles — and is specifically hardened against adversarial attacks: agent/object role swaps, negation, and passive-voice rewriting that preserve surface similarity while reversing meaning.

### What it does

Given a **question**, a **reference (teacher) answer**, and a **student answer** in Bengali, the system:

1. Resolves pronouns via rule-based coreference resolution
2. Parses dependency trees (Stanza / SOV heuristic fallback)
3. Extracts (Subject, Relation, Object) triples
4. Builds knowledge graphs for both reference and student answers
5. Computes sentence-level semantic similarity (LaBSE)
6. Computes structural graph similarity via a Siamese GAT-GNN
7. Validates semantic roles — detects agent/object swaps (Karak penalty)
8. Detects entity mismatches via NER + soft BanglaBERT cosine matching
9. Feeds all signals into an XGBoost + MLP ensemble to produce a final score with a human-readable explanation

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Running the Project](#running-the-project)
7. [Module Reference](#module-reference)
8. [Model Details](#model-details)
9. [Evaluation Outputs](#evaluation-outputs)
10. [Kaggle Usage](#kaggle-usage)
11. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
Student Answer
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Coreference Resolution  (BengaliCoreferenceResolver)    │
│     Pronouns → antecedent nouns from question/reference     │
└─────────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
┌─────────────────────┐              ┌────────────────────────┐
│  Sentence-level      │              │  Dependency Parsing     │
│  LaBSE Similarity   │              │  (Stanza / SOV fallback)│
└────────┬────────────┘              └──────────┬─────────────┘
         │                                      │
         │                          ┌───────────┴────────────┐
         │                          ▼                        ▼
         │                ┌──────────────────┐  ┌─────────────────────┐
         │                │  Triple Extractor │  │  Karak Validator    │
         │                │  (S, R, O) tuples │  │  Agent/Object swap  │
         │                └────────┬─────────┘  └──────────┬──────────┘
         │                         │                       │
         │                ┌────────▼────────┐              │
         │                │ Knowledge Graph  │              │
         │                │ (NetworkX DiGraph│              │
         │                └────────┬────────┘              │
         │                         │                       │
         │                ┌────────▼────────────┐          │
         │                │  Siamese GAT-GNN     │          │
         │                │  Structural Sim      │          │
         │                └────────┬────────────┘          │
         │                         │                       │
         ▼                         ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Feature Vector  [semantic_sim, graph_sim,         │
│                            karak_penalty, entity_mismatch]  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  XGBoost + MLP     │
                    │  Ensemble Scorer   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Score  0 – 100   │
                    │  + Explanation     │
                    └────────────────────┘
```

### Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Sentence similarity | **LaBSE** | Cross-lingual, optimised for semantic similarity; outperforms raw BERT mean-pooling |
| Node embeddings | **BanglaBERT** | Token-level representations better for individual entity words |
| Graph encoder | **GAT** (not GCN) | Attention weights focus on important relations; multi-head provides robustness |
| Generative fallback | **Removed** | mt5-small not fine-tuned on Bengali triple extraction — produces noise |
| Final scorer | **XGBoost + MLP** | Non-linear interactions between features (e.g. high sim + role swap = 0) |

---

## Project Structure

```
Project3/
├── main.py                  ← Full training + evaluation pipeline
├── app.py                   ← Gradio web interface
├── requirements.txt         ← Python dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── config.py            ← Constants, paths, hyperparameters
    ├── data_processing.py   ← Dataset loading & adversarial augmentation
    ├── preprocessing.py     ← Bengali text normalization & tokenization
    ├── coreference.py       ← Rule-based pronoun resolution
    ├── dependency_parser.py ← Stanza dependency parser (+ SOV fallback)
    ├── triple_extractor.py  ← (S, R, O) triple extraction
    ├── knowledge_graph.py   ← NetworkX knowledge graph builder
    ├── embeddings.py        ← LaBSE + BanglaBERT dual-encoder
    ├── gnn_model.py         ← Siamese GAT-GNN + PyG data converter
    ├── karak_validator.py   ← Semantic role (কারক) validator
    ├── scorer.py            ← XGBoost + MLP ensemble scorer
    ├── pipeline.py          ← Feature extraction, GNN training, metric utils
    ├── evaluation.py        ← Full metrics, tables, and all visualisations
    └── inference.py         ← Load models, grade single sample
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 – 3.11 |
| pip | ≥ 23.0 |
| CUDA (optional) | 11.8 / 12.1 for GPU acceleration |
| RAM | ≥ 16 GB recommended (models load into memory) |
| Disk space | ~8 GB (model weights + dataset) |

---

## Installation

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd Project3
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (CPU or GPU)

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install PyTorch Geometric

PyTorch Geometric 2.4+ does **not** require `pyg_lib`, `torch_scatter`, or similar optional packages for the GAT operations used by this project. A simple one-line install is sufficient:

```powershell
pip install torch_geometric
```

**Optional acceleration packages** (only if you want maximum performance and pre-built wheels exist for your exact torch + CUDA version):

```powershell
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<VERSION>+<CUDA>.html
```

Replace `<VERSION>` with your PyTorch version and `<CUDA>` with `cpu`, `cu121`, `cu124`, etc.  
Find the correct URL at [data.pyg.org/whl](https://data.pyg.org/whl/).

> ℹ️ **Check your PyTorch version** before using the optional wheel URL:
> ```powershell
> python -c "import torch; print(torch.__version__)"
> ```
> If pre-built wheels are not available for your version, skip the optional packages — the project works without them.

> ⚠️ **`pyg_lib` is not required** and has no pre-built wheels for many PyTorch versions. Do not include it in the install command.

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the Stanza Bengali model

This is handled automatically on first run, but you can pre-download it:

```python
import stanza
stanza.download('bn')
```

### 7. (Optional) Install Bengali font for graph labels

```bash
# Ubuntu / Debian
sudo apt-get install -y fonts-noto-cjk fonts-noto
fc-cache -f -v
```

On Windows, download and install [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali) manually.

---

## Dataset Setup

The system expects a **tab-separated** CSV file named:

```
bengali_answer_evaluation_dataset.csv
```

placed in the **project root** (`Project3/`).

The file should have the following columns (tab-separated, `\t`):

| Column | Accepted names | Description |
|--------|----------------|-------------|
| `id` | `id` | Unique row identifier |
| `question` | `question` | Bengali question |
| `reference_answer` | `teacher_ans` / `reference_answer` | Correct reference answer |
| `student_answer` | `student_ans` / `student_answer` | Student's answer to grade |
| `label` | `label` | `correct` / `partially_correct` / `incorrect` |
| `human_score` | `human_score` | Human-assigned score (0–100) |
| `subject` | `subject` *(optional)* | Subject domain (e.g. বিজ্ঞান, ইতিহাস) |

> ℹ️ Column names `teacher_ans` and `student_ans` are automatically renamed on load. A missing `subject` column is filled with `"সাধারণ"`.

On first run, the pipeline creates an augmented version:

```
bengali_answer_evaluation_dataset_augmented.csv
```

which adds **50 adversarial examples** (17 Karak reversal, 16 negation, 17 voice change) to stress-test the grader.

---

## Running the Project

### Quick Start (TL;DR)

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (first time only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install -r requirements.txt

# 3. Place dataset in project root
#    bengali_answer_evaluation_dataset.csv  ← must be here

# 4. Train + evaluate full pipeline
python main.py

# 5. (Optional) Launch interactive web app
python app.py
```

---

### Full Training + Evaluation Pipeline

```bash
python main.py
```

Runs all **8 pipeline stages** in order:

| Stage | What happens | Output |
|-------|-------------|--------|
| 1 | Initialise Stanza NLP + NER pipelines | Model weights downloaded on first run |
| 2 | Load dataset, augment with 50 adversarial examples | `bengali_answer_evaluation_dataset_augmented.csv` |
| 3 | Extract features for training split | Feature matrix (semantic sim, graph sim, penalty, entity mismatch) |
| 4 | Train Siamese GAT-GNN for 5 epochs | `nlp_dominator_gnn.pth` |
| 5 | Train XGBoost + MLP Ensemble Scorer | `nlp_dominator_scorer.pkl` |
| 6 | Extract features for test split | Test feature matrix |
| 7 | Compute all evaluation metrics | Printed tables (MAE, RMSE, Pearson r, accuracy, F1) |
| 8 | Generate all visualisation plots | PNG files in project root |

**Generated plots:**

| File | Content |
|------|---------|
| `correlation_plot.png` | Predicted vs. human score scatter |
| `confusion_matrices.png` | Classification confusion matrix |
| `error_distribution.png` | Score error histogram |
| `score_distributions.png` | Predicted / human score distributions |
| `ablation_bar_charts.png` | Component ablation study |
| `scatter_comparison.png` | System variants comparison |
| `per_subject_comparison.png` | Accuracy broken down by subject |
| `incremental_ablation.png` | Incremental feature contribution |

**Skip 5-fold cross-validation** (significantly faster, ~30% less runtime):

```bash
python main.py --skip-cv
```

**Expected runtime** (approximate, GPU recommended):

| Hardware | With CV | Without CV |
|----------|---------|-----------|
| GPU (RTX 3060+) | ~20–30 min | ~8–12 min |
| CPU only | ~60–90 min | ~25–40 min |

---

### Launch Interactive Web App (Gradio)

Once models are trained (`.pth` and `.pkl` files present), start the web interface:

```bash
python app.py
```

Gradio will print the URL — open it in your browser:

```
Running on local URL:  http://localhost:7860
Running on public URL: https://xxxx.gradio.live   ← shareable link (share=True)
```

**Features of the web interface:**
- Enter question, reference answer, and student answer in Bengali
- Instantly receives a **0–100 score** with a textual explanation
- Displays **Knowledge Graph** comparison plots for both answers
- Shows detected role errors (Karak penalty), entity mismatches, and similarity scores

> ℹ️ If port 7860 is busy, Gradio will automatically pick the next available port (7861, 7862, …) and print it.

---

### Grade a Single Answer Programmatically

```python
from src.inference import load_pretrained_models, grade_single

components = load_pretrained_models()

result = grade_single(
    question="বাংলাদেশের মুক্তিযুদ্ধ কত সালে হয়েছিল?",
    reference_answer="বাংলাদেশের মুক্তিযুদ্ধ ১৯৭১ সালে হয়েছিল।",
    student_answer="১৯৭১ সালে বাংলাদেশ স্বাধীন হয়েছিল।",
    **components
)

print(f"Score: {result['score']:.1f} / 100")
print(result['explanation'])
```

---

## Module Reference

### `src/config.py`
Central configuration file. Edit here to change model names, paths, training epochs, ensemble weights, or synonym thresholds.

Key constants:
```python
SENTENCE_MODEL_NAME = "sentence-transformers/LaBSE"
NODE_MODEL_NAME     = "csebuetnlp/banglabert"
NER_MODEL_NAME      = "Davlan/bert-base-multilingual-cased-ner-hrl"
GNN_EPOCHS          = 5
ENSEMBLE_WEIGHTS    = (0.6, 0.4)   # XGBoost, MLP
SYNONYM_THRESHOLD   = 0.75
```

### `src/data_processing.py`
- `get_augmented_dataset()` — resolve dataset path, augment with 50 adversarial examples, return DataFrame
- `setup_bengali_font()` — configure matplotlib for Bengali text rendering

### `src/preprocessing.py`
- `TextPreprocessor` — Indic normalisation → punctuation removal → Stanza lemmatisation (whitespace fallback)

### `src/coreference.py`
- `BengaliCoreferenceResolver` — replaces Bengali pronouns (তিনি, সে, এটি…) with antecedent nouns from the question/reference context

### `src/dependency_parser.py`
- `BengaliDependencyParser` — Stanza-based dependency tree parser; falls back to a Subject-Object-Verb heuristic when Stanza is unavailable

### `src/triple_extractor.py`
- `TripleExtractor` — extracts `(subject, relation, object)` triples via dependency parse; rule-based fallback handles nominal predicates, passive voice, negation, and postpositional objects

### `src/knowledge_graph.py`
- `KnowledgeGraphConstructor` — builds a `networkx.DiGraph` from triples; optional spring-layout visualisation

### `src/embeddings.py`
- `BanglaBERTEmbedder` — dual encoder:
  - `cosine_similarity(text1, text2)` → LaBSE sentence similarity
  - `get_embedding(word)` → BanglaBERT 768-dim node embedding
  - `word_cosine_similarity(w1, w2)` → soft synonym check

### `src/gnn_model.py`
- `SiameseGNN` — two-tower GAT network; outputs cosine similarity between graph-level embeddings
- `nx_to_pyg_data(nx_graph, embeddings)` — converts NetworkX + embedding dict to PyG `Data`

### `src/karak_validator.py`
- `KarakValidator` — detects agent↔object role swaps using dependency parse roles; returns `(penalty, explanations)`

### `src/scorer.py`
- `AnswerScorer` — engineers 11 interaction features from the 4 base signals, trains XGBoost + MLP, and produces an ensemble score in [0, 100]

### `src/pipeline.py`
- `compute_entity_mismatch(...)` — NER + BanglaBERT soft entity matching  
- `create_graphs_and_features(df, ...)` — full batch feature extraction  
- `train_gnn(...)` / `get_gnn_similarities(...)` — GNN training and inference  
- `evaluate_metrics(...)` / `plot_correlation(...)` — basic metric helpers

### `src/evaluation.py`
- `compute_full_metrics(y_true, y_pred, name)` — all regression + classification metrics  
- `print_all_metric_tables(all_metrics)` — regression, tolerance, classification, and ablation tables  
- `run_cross_validation(...)` — 5-fold CV on non-adversarial samples  
- `per_subject_breakdown(...)` / `per_category_breakdown(...)` — fine-grained analysis  
- `plot_*` functions — confusion matrices, error histograms, scatter plots, ablation bar charts

### `src/inference.py`
- `load_pretrained_models()` — loads `.pth` and `.pkl` from disk  
- `grade_single(question, reference, student, ...)` — full pipeline for one sample; returns score + all intermediate signals

---

## Model Details

### Feature Vector (4 base + 7 engineered = 11 features)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `base_sim` | LaBSE cosine similarity |
| 2 | `graph_sim` | GAT-GNN cosine similarity |
| 3 | `penalty` | Karak role-swap penalty [0, 1] |
| 4 | `entity_mm` | Soft entity mismatch [0, 1] |
| 5 | `sim×penalty` | Interaction: high sim + role error |
| 6 | `sim×entity` | Interaction: high sim + entity swap |
| 7 | `graph×penalty` | Structural sim × role errors |
| 8 | `penalty×entity` | Combined penalty signal |
| 9 | `sim_diff` | Semantic – structural similarity gap |
| 10 | `avg_sim` | Mean of base and graph similarity |
| 11 | `max_penalty` | Worst penalty signal |

### Adversarial Test Set (50 samples)

| Type | Count | Description |
|------|-------|-------------|
| Karak Reversal | 17 | Agent and object words swapped |
| Negation | 16 | Correct sentence + না/নয়/নি appended |
| Voice Change | 17 | Active → passive with reversed meaning |

All adversarial samples have `human_score = 0` and are held out of training entirely.

---

## Evaluation Outputs

After `python main.py` the following files are created:

| File | Content |
|------|---------|
| `nlp_dominator_gnn.pth` | Trained GAT-GNN weights |
| `nlp_dominator_scorer.pkl` | Trained XGBoost + MLP scorer |
| `bengali_answer_evaluation_dataset_augmented.csv` | Extended dataset |
| `correlation_plot.png` | Predicted vs human score regression plots |
| `confusion_matrices.png` | Correct/Incorrect classification matrices |
| `error_distribution.png` | Residual histograms per model variant |
| `score_distributions.png` | Human vs predicted score distributions |
| `ablation_bar_charts.png` | Side-by-side ablation comparison |
| `scatter_comparison.png` | Scatter plots: normal vs adversarial samples |
| `per_subject_comparison.png` | MAE & F1 per Bengali subject |
| `incremental_ablation.png` | Delta contribution of each component |

Printed metric tables compare **5 model variants**:

| Step | Model |
|------|-------|
| B | Baseline — LaBSE similarity only |
| +E | + Soft Entity Matching |
| +G | + GAT-GNN Knowledge Graph |
| +K | + Karak Role Validation |
| +X | + XGBoost + MLP Ensemble (proposed) |

---

## Kaggle Usage

1. Upload `bengali_answer_evaluation_dataset.csv` as a Kaggle Dataset.
2. Add it to your notebook via **Add Data** in the right sidebar.
3. Ensure **Internet** is turned **ON** in notebook settings (required for model downloads).
4. The dataset path resolver in `src/config.py` automatically locates the CSV under `/kaggle/input/`.
5. Run the notebook (`kaggle_bengali_eval_notebook (1).ipynb`) or adapt `main.py` for the Kaggle environment.

---

## Troubleshooting

**`FileNotFoundError: bengali_answer_evaluation_dataset.csv not found`**  
→ Place the CSV in the project root, or on Kaggle add the dataset via "Add Data".

**`stanza.download` network error**  
→ Ensure internet access.  On a restricted network, manually download the Bengali model:  
```python
import stanza
stanza.download('bn', model_dir='/path/to/stanza_resources')
```
Then set `STANZA_RESOURCES_DIR` in your environment.

**`torch_geometric` import error / version mismatch**  
→ Run `python -c "import torch; print(torch.__version__)"` to get your exact version.  
→ For basic usage (GAT, global_mean_pool), `pip install torch_geometric` alone is sufficient — no optional packages (`pyg_lib`, `torch_scatter`, etc.) are needed.  
→ If you still want the optional acceleration packages, use the wheel URL matching your exact version from [data.pyg.org/whl](https://data.pyg.org/whl/) and **omit `pyg_lib`** (no pre-built wheel exists for most versions).

**`ValueError: Due to a serious vulnerability … upgrade torch to at least v2.6`**  
→ `transformers 5.x` enforces `torch >= 2.6` (CVE-2025-32434). If you have `torch 2.5.x`, downgrade transformers:
```powershell
pip install "transformers>=4.35.0,<5.0.0" "huggingface-hub<1.0"
```
This is already pinned in `requirements.txt` for fresh installs.

**PowerShell: `ERROR: Directory '\\'` or `'-f' is not recognized`**  
→ PowerShell does not support `\` for line continuation in `pip` commands. Keep the entire install on one line:
```powershell
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<VERSION>+<CUDA>.html
```

**`ERROR: Could not find a version that satisfies the requirement pyg_lib`**  
→ `pyg_lib` has no pre-built wheel for most PyTorch versions. Simply remove it from your install command. The project does not use it.

**Out-of-memory error when loading models**  
→ Both LaBSE (~500 MB) and BanglaBERT (~500 MB) load into RAM.  
Ensure ≥ 6 GB free RAM.  On CPU, inference is slower but functional.

**Bengali text not rendering in plots**  
→ Install the Noto Sans Bengali font and clear the matplotlib font cache:
```python
import matplotlib
import os
cache = matplotlib.get_cachedir()
for f in os.listdir(cache):
    if f.startswith('fontlist'):
        os.remove(os.path.join(cache, f))
```

**Gradio share link not appearing**  
→ Pass `share=False` in `app.py` and access the app at `http://localhost:7860`.
