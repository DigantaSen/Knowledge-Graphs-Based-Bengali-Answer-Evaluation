# 🏛️ Bengali Answer Evaluation System

> **Knowledge Graph · LaBSE/BanglaBERT Embeddings · Siamese GAT-GNN · Karak Validator · XGBoost + MLP Ensemble**

A modular, end-to-end pipeline for automatically scoring Bengali short-answer responses against reference answers.

---

## 📂 Project Structure

```
Knowledge-Graphs-Based-Bengali-Answer-Evaluation/
│
├── data/                            ← Dataset CSVs (place your data here)
│   ├── dataset_single_sentence.csv
│   ├── dataset_medium_sentences.csv
│   ├── dataset_medium_sentences_augmented.csv  ← auto-generated on first run
│   └── dataset_long_sentences.csv
│
├── datasets/clean_synthetic/        ← Clean synthetic data sources (JSON) + build.py + rubric README
├── report/                          ← Revised report: report.tex (XeLaTeX), figures/, report.pdf
│
├── models/                          ← Saved GNN weights + scorer (auto-created)
│   ├── nlp_dominator_gnn_<dataset_stem>.pth
│   └── nlp_dominator_scorer_<dataset_stem>.npz   ← plain arrays, no pickle
│
├── results/                         ← Timestamped run output folders (auto-created)
│   └── train_YYYYMMDD_HHMMSS/
│       ├── metrics_table.csv
│       ├── cv_results.csv
│       ├── predictions.csv
│       ├── adversarial_breakdown.csv
│       ├── feature_importances.csv
│       ├── run_config.json
│       ├── console_log.txt
│       ├── run_summary.md
│       └── figures/
│           ├── ablation_bar.png
│           ├── pred_vs_true_scatter.png
│           ├── confusion_matrix.png
│           ├── adversarial_breakdown.png
│           └── cv_folds.png
│
├── src/
│   ├── config.py                    ← All paths, model IDs, hyper-parameters
│   ├── data/
│   │   ├── dataset.py               ← Loading, adversarial samples, dedup, split, templates
│   │   └── synthetic.py             ← Synthetic negation / swap / wrong-number training rows
│   ├── preprocessing/
│   │   └── text_preprocessor.py     ← Indic NLP normalisation
│   ├── nlp/
│   │   ├── dependency_parser.py     ← BNLP POS-based roles (subject, -কে object, passive agent)
│   │   ├── coreference.py           ← Rule-based coreference resolution
│   │   └── triple_extractor.py      ← Predicate-centred triples (verb + negation as a node)
│   ├── graph/
│   │   ├── kg_constructor.py        ← Knowledge graph (NetworkX)
│   │   └── embedder.py              ← LaBSE (sentences, words, nodes) + BanglaBERT baseline
│   ├── models/
│   │   ├── siamese_gnn.py           ← Siamese GAT-GNN + PyG data converter
│   │   └── answer_scorer.py         ← XGBoost + MLP ensemble scorer
│   ├── validation/
│   │   └── karak_validator.py       ← Semantic role (Karak) validation
│   ├── evaluation/
│   │   ├── overrides.py             ← Hard post-processing overrides
│   │   └── metrics.py               ← Pearson / MAE / F1 / macro-F1 + bootstrap CIs
│   ├── experiment.py                ← Steps shared by train.py / evaluate.py / demo.py
│   ├── features/
│   │   └── pipeline.py              ← Feature extraction (shared by train/eval/demo) + GNN
│   └── utils/
│       ├── helpers.py               ← Tokenising, sentence alignment, negation, numbers, ROUGE-L
│       └── results_saver.py         ← Run output persistence (CSV, MD, JSON, plots)
│
├── train.py                         ← Full training pipeline (entry point)
├── evaluate.py                      ← Stand-alone evaluation + 5-fold CV
├── demo.py                          ← Interactive Gradio web demo
├── run_all_datasets.py              ← Run train + evaluate on all 3 datasets
├── plot_results.py                  ← Regenerate figures from a results directory
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### 1 — Python version

Tested with **Python 3.12** on CPU, using the exact versions pinned in `requirements.txt`.

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` also installs BanglaBERT's text normaliser from GitHub
(`csebuetnlp/normalizer`), so `git` must be available.

> **Note on PyTorch Geometric:** Install the wheels that match your CUDA version.
> For CPU-only (simplest):
> ```bash
> pip install torch-geometric torch-scatter torch-sparse \
>     -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
> ```
> For CUDA 11.8:
> ```bash
> pip install torch-geometric torch-scatter torch-sparse \
>     -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
> ```

### 3 — Download BNLP / NLTK resources (first run only)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Without `punkt`, BNLP retries the download on every start. The BNLP dependency parser downloads its POS model automatically the first time it is used. Make sure you have an internet connection on the first run (or use `--offline` after your initial setup — see below).

---

## 📁 Dataset

All dataset CSVs must be placed inside the **`data/`** folder:

```
data/
├── dataset_single_sentence.csv      ← short, single-sentence answers
├── dataset_medium_sentences.csv     ← 2–3 sentence answers
└── dataset_long_sentences.csv       ← paragraph-length answers
```

Each CSV must have the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique integer row ID |
| `question` | Question text (Bengali) |
| `reference_answer` | Correct model answer (Bengali) |
| `student_answer` | Student's answer to be graded (Bengali) |
| `label` | `correct` / `partially_correct` / `incorrect` |
| `human_score` | Human-assigned score, integer 0–100 |
| `subject` | Subject area (e.g., `বিজ্ঞান`, `ইতিহাস`) |

> The pipeline auto-detects which file to use: it tries `dataset_single_sentence.csv` first, then `dataset_medium_sentences.csv`, then `dataset_long_sentences.csv`, then `train_data.csv`.

On the **first run** for each dataset, an augmented CSV is automatically written alongside the source file (e.g., `data/dataset_medium_sentences_augmented.csv`). Subsequent runs load this cached file directly.

**Clean synthetic data:** `datasets/clean_synthetic/` holds 514 additional rows (54 new questions) written without templates and scored by a fixed rubric — synthetic, not human grades. `python datasets/clean_synthetic/build.py` writes `data/dataset_clean_<tier>.csv` and `data/dataset_<tier>_plus_clean.csv` (original + clean, with a `source` column). See `datasets/clean_synthetic/README.md`.

**Data handling (see `src/config.py`):**

- `DEDUPLICATE_PAIRS` — identical (question, reference, student) rows are merged and their scores averaged (1000 → 669 single, 1000 → 492 medium, 320 → 291 long), because the datasets repeat pairs with conflicting scores.
- `SPLIT_GROUP_BY` / `--group-by` — `pair` (unseen answers, default) or `question` (unseen questions).
- `SYNTHETIC_ERRORS` — rule-generated negated / role-swapped / wrong-number answers (score 0) plus unchanged copies are added to the **training split only**. Sentences of the hand-written adversarial test samples are excluded.
- Many student answers are templated (e.g. 695/1000 medium answers start or end with phrases such as `এই প্রশ্নের উত্তরে`, `আরও তথ্য প্রয়োজন।`). Templates are detected from the data and results are also reported on the non-templated subset.

---

## 🚀 Running the Project

### A. Train (full pipeline)

```bash
python train.py
```

This will:
1. Load `data/dataset_single_sentence.csv` (or whichever CSV is found first)
2. Write / load a per-dataset augmented CSV with 51 adversarial examples
3. Split train/test by group (`SPLIT_GROUP_BY` in `src/config.py`: identical answer pairs by default, or whole questions) so duplicated rows never straddle the split; all adversarial rows go to test
4. Add synthetic error rows to the training split, then extract KG + GNN + Karak features
5. Train the Siamese GAT-GNN (5 epochs by default; seeded, mini-batched)
6. Train the XGBoost + MLP ensemble scorer on **out-of-fold** GNN similarities; the XGBoost/MLP weight is chosen on out-of-fold training predictions
7. Apply hard overrides (negation cap 30 / numeric cap 25) — single-sentence references only
8. Print metric tables for the full test set, normal rows, non-templated rows and adversarial rows, with bootstrap 95% CIs against the best baseline
9. Save `models/nlp_dominator_gnn_<dataset_stem>.pth` and `models/nlp_dominator_scorer_<dataset_stem>.npz`
10. Persist a timestamped results folder under `results/train_YYYYMMDD_HHMMSS/`

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset PATH` | auto-detected | Path to a specific CSV file |
| `--epochs N` | `5` | Number of GNN training epochs |
| `--output-dir DIR` | `models` | Directory to save trained models |
| `--output-results DIR` | `results` | Parent folder for the timestamped run-output directory |
| `--group-by pair\|question` | `pair` | Split unit (unseen answers / unseen questions) |
| `--offline` | off | Force offline mode — use only locally cached HuggingFace models |

```bash
# Train on the medium-sentences dataset for 10 epochs
python train.py --dataset data/dataset_medium_sentences.csv --epochs 10

# Save results to a custom folder
python train.py --output-results my_results

# Use cached HuggingFace models (no internet required after first run)
python train.py --offline
```

---

### B. Evaluate / cross-validate

```bash
python evaluate.py
```

Loads (or trains) dataset-specific models and evaluates them on the held-out test set, then runs 5-fold cross-validation on the normal samples.

**Model loading behaviour:**

- Looks for `<models-dir>/nlp_dominator_gnn_<dataset_stem>.pth` and the matching scorer (the files `train.py` saves), or the explicit `--gnn-model` / `--scorer-model` paths.
- Missing files are downloaded from the HuggingFace repo when default paths are used.
- If no GNN weights are found, the GNN and scorer are trained from scratch and saved. A saved scorer is only reused together with its saved GNN.
- Baselines (LaBSE, ROUGE-L, BanglaBERT, GNN-only) are linearly calibrated on the train split before scoring, so they are compared fairly with the trained ensemble.
- Cross-validation extracts features once, uses grouped folds, and trains a fresh GNN and scorer per fold (with that fold's synthetic rows).

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset PATH` | auto-detected | Dataset CSV to evaluate on |
| `--models-dir DIR` | `models` | Directory holding the per-dataset model files |
| `--gnn-model PATH` | `<models-dir>/nlp_dominator_gnn_<stem>.pth` | Explicit GNN weights path |
| `--scorer-model PATH` | `<models-dir>/nlp_dominator_scorer_<stem>.npz` | Explicit scorer path |
| `--epochs N` | `5` | GNN epochs when training is needed (including every CV fold) |
| `--group-by pair\|question` | `pair` | Split unit (unseen answers / unseen questions) |
| `--skip-cv` | off | Skip 5-fold cross-validation (much faster) |
| `--output-results DIR` | `results` | Parent folder for the timestamped run-output directory |
| `--offline` | off | Force offline mode — use only locally cached HuggingFace models |

```bash
# Full evaluation with 5-fold cross-validation
python evaluate.py

# Skip CV for a quick metrics check
python evaluate.py --skip-cv

# Evaluate on a specific dataset
python evaluate.py --dataset data/dataset_long_sentences.csv

# Evaluate using explicit model paths
python evaluate.py --gnn-model models/nlp_dominator_gnn_dataset_medium_sentences.pth \
                   --scorer-model models/nlp_dominator_scorer_dataset_medium_sentences.npz
```

---

### C. Interactive Gradio demo

```bash
python demo.py
```

Opens a web UI at **http://localhost:7860** where you can paste a question, a reference answer, and a student answer, and see the score + detailed breakdown.

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--share` | off | Create a public Gradio link (useful on Colab / Kaggle) |
| `--server-port N` | `7860` | Port to bind the Gradio server |

```bash
# Public link (Kaggle / Colab)
python demo.py --share
```

> **Important:** The demo scans `models/` for `nlp_dominator_gnn_<dataset_stem>.pth` / `nlp_dominator_scorer_<dataset_stem>.npz` pairs and picks one by answer length. Run `train.py` (or `run_all_datasets.py`) first. Without trained files it falls back to a heuristic score.

---

### D. Run all 3 datasets automatically

```bash
python run_all_datasets.py
```

Runs `train.py` then `evaluate.py` on **all three datasets one by one**; `evaluate.py` loads the models `train.py` just saved. Model files are named per dataset, so they share one folder (which is where `evaluate.py` and `demo.py` look):

```
models/
├── nlp_dominator_{gnn,scorer}_dataset_single_sentence.{pth,pkl}
├── nlp_dominator_{gnn,scorer}_dataset_medium_sentences.{pth,pkl}
└── nlp_dominator_{gnn,scorer}_dataset_long_sentences.{pth,pkl}

results/
├── single/    ← metrics, predictions, figures for single-sentence
├── medium/    ← ...
├── long/      ← ...
└── comparison_all_datasets.csv  ← side-by-side Pearson/MAE/F1 table (from each evaluate_* run)
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `5` | GNN epochs per dataset |
| `--datasets CSV1 CSV2 …` | all 3 | Run only specific CSVs |
| `--skip-cv` | off | Pass `--skip-cv` to `evaluate.py` for each dataset |
| `--group-by pair\|question` | `pair` | Split unit; `question` writes to `models/question_split/` and `results/<name>_question/` |
| `--offline` | off | Use locally cached HuggingFace models only |

```bash
# Run all 3 with 10 epochs each
python run_all_datasets.py --epochs 10

# Run only two specific datasets
python run_all_datasets.py --datasets data/dataset_single_sentence.csv data/dataset_medium_sentences.csv

# Offline mode (after a successful first online run)
python run_all_datasets.py --offline --skip-cv
```

---

## 📊 Run Outputs (for Report Writing)

Every time you run `train.py` or `evaluate.py`, a **timestamped results folder** is automatically created:

```
results/
└── train_YYYYMMDD_HHMMSS/          ← one folder per run
    ├── metrics_table.csv            ← all model rows: Pearson, Spearman, MAE, RMSE, R², F1, …
    ├── metrics_table_normal.csv     ← same, normal (non-adversarial) test rows only
    ├── metrics_table_nontemplated.csv ← normal rows whose student answer is not templated
    ├── metrics_table_adversarial.csv  ← the 51 hand-written adversarial rows
    ├── significance.csv             ← bootstrap 95% CIs + paired test vs the best baseline
    ├── cv_results.csv               ← per-fold + mean±std (evaluate.py only)
    ├── predictions.csv              ← per-sample y_true / y_pred_raw / y_pred_final / abs_error
    ├── adversarial_breakdown.csv    ← metrics split by error category (Karak/Negation/…)
    ├── feature_importances.csv      ← XGBoost feature ranking
    ├── run_config.json              ← dataset, epochs, seed, timestamp …
    ├── console_log.txt              ← full stdout from the run
    ├── run_summary.md               ← ✅ paste-ready Markdown for your report
    └── figures/                     ← auto-generated plots
        ├── ablation_bar.png         ← Pearson / F1 per model variant
        ├── pred_vs_true_scatter.png ← predicted vs true, normal vs adversarial
        ├── confusion_matrix.png     ← binary correct/incorrect heatmap
        ├── adversarial_breakdown.png← MAE and score per error category
        └── cv_folds.png             ← Pearson / MAE stability across folds
```

> Figures are generated **automatically** at the end of each run.  
> To regenerate them manually (e.g. after tweaking the plot style):

```bash
python plot_results.py --results-dir results/train_YYYYMMDD_HHMMSS
```

| Flag | Default | Description |
|------|---------|-------------|
| `--results-dir DIR` | *(required)* | Path to a results folder |
| `--num-adversarial N` | `51` | Number of adversarial samples at the end of predictions.csv |
| `--save-pdf` | off | Also save each figure as a `.pdf` |

### Quick reference

| File | Where to use it in the report |
|------|-------------------------------|
| `run_summary.md` | Copy tables directly into your report |
| `figures/ablation_bar.png` | Results / model comparison figure |
| `figures/pred_vs_true_scatter.png` | Error analysis figure |
| `figures/confusion_matrix.png` | Classification results figure |
| `figures/adversarial_breakdown.png` | Adversarial evaluation figure |
| `figures/cv_folds.png` | Cross-validation stability figure |
| `metrics_table.csv` | Results / comparison table |
| `cv_results.csv` | CV stability analysis |
| `predictions.csv` | Qualitative error examples |
| `feature_importances.csv` | Feature analysis / ablation section |
| `run_config.json` | Experimental setup / reproducibility |

> Use `--output-results <DIR>` to change the parent folder (default: `results/`).

---

## 🔬 Expected Evaluation Output

```
====================================================================================================
  REGRESSION METRICS (Test Set)
====================================================================================================
  Model                                  |  Pearson | Spearman |    MAE |   RMSE |    R^2
  ---------------------------------------+----------+----------+--------+--------+-------
  Baseline: LaBSE Cosine Sim             |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
  Baseline: ROUGE-L Only                 |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
  Baseline: LaBSE + ROUGE-L Avg          |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
  Baseline: BanglaBERT Cosine Sim        |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
  Ablation: GNN Only                     |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
  Proposed: Full Ensemble (no override)  |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
  Proposed: Full + Overrides (ours)      |   0.xxxx |   0.xxxx |  xx.xx |  xx.xx | x.xxxx
====================================================================================================

===========================================================================
  CV SUMMARY (mean ± std)   [5-fold, non-adversarial samples]
===========================================================================
  Pearson:  0.xxxx ± 0.xxxx
  MAE:      xx.xx  ± x.xx
  F1:       xx.x%  ± x.x%
===========================================================================
```

> The exact numbers depend on the dataset used (single / medium / long sentences).

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11.0 | Deep learning core |
| `torch-geometric` | 2.7.0 | Graph Neural Networks |
| `transformers` | 5.5.3 | LaBSE, BanglaBERT, NER models |
| `sentence-transformers` | 5.4.0 | Cosine similarity via LaBSE |
| `bnlp_toolkit` | 4.4.1 | BNLP POS tagger |
| `indic-nlp-library` | 0.92 | Bengali text normalisation |
| `xgboost` | 3.2.0 | Ensemble scoring |
| `networkx` | 3.6.1 | Knowledge graph construction |
| `gradio` | 6.11.0 | Interactive demo UI |
| `scikit-learn` | 1.8.0 | Cross-validation, metrics |
| `scipy` | 1.17.1 | Pearson / Spearman correlation |
| `rouge-score` | 0.1.2 | ROUGE-L baseline |
| `matplotlib` | 3.10.8 | Result figures |
| `seaborn` | 0.13.2 | Confusion matrix styling |
| `normalizer` (csebuetnlp) | git | BanglaBERT input normalisation |
| `nltk` | 3.9.4 | Tokeniser resources for BNLP |

---

## 🐍 Kaggle Usage

1. Upload this repository as a **Kaggle Dataset**.
2. Upload your data CSVs as a separate **Kaggle Dataset** named `nlp-dominator-dataset`.
3. In your notebook:

```python
import subprocess, sys
subprocess.run([sys.executable, "train.py"], check=True)
```

The code automatically detects the Kaggle environment (`/kaggle/input` exists) and reads CSVs from the Kaggle dataset path.

---

## 📝 Citation / Report

This project was developed as the final submission for the **Introduction to NLP** course.
See `report.pdf` for the full system description, methodology, and results.
