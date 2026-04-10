# 🏛️ Bengali Answer Evaluation System

> **Knowledge Graph · LaBSE/BanglaBERT Embeddings · Siamese GAT-GNN · Karak Validator · XGBoost + MLP Ensemble**

A modular, end-to-end pipeline for automatically scoring Bengali short-answer responses against reference answers.

---

## 📂 Project Structure

```
ProjectFinal3/
│
├── data/                            ← Dataset CSVs (place your data here)
│   ├── dataset_single_sentence.csv
│   ├── dataset_medium_sentences.csv
│   ├── dataset_medium_sentences_augmented.csv  ← auto-generated on first run
│   └── dataset_long_sentences.csv
│
├── models/                          ← Saved GNN weights + scorer (auto-created)
│   ├── nlp_dominator_gnn_<dataset_stem>.pth
│   └── nlp_dominator_scorer_<dataset_stem>.pkl
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
│   │   └── dataset.py               ← Data loading + adversarial augmentation
│   ├── preprocessing/
│   │   └── text_preprocessor.py     ← Indic NLP normalisation + tokenisation
│   ├── nlp/
│   │   ├── dependency_parser.py     ← BNLP POS-based dependency parser
│   │   ├── coreference.py           ← Rule-based coreference resolution
│   │   └── triple_extractor.py      ← (Subject, Relation, Object) extraction
│   ├── graph/
│   │   ├── kg_constructor.py        ← Knowledge graph (NetworkX)
│   │   └── embedder.py              ← LaBSE / BanglaBERT embedder + node embs
│   ├── models/
│   │   ├── siamese_gnn.py           ← Siamese GAT-GNN + PyG data converter
│   │   └── answer_scorer.py         ← XGBoost + MLP ensemble scorer
│   ├── validation/
│   │   └── karak_validator.py       ← Semantic role (Karak) validation
│   ├── evaluation/
│   │   ├── overrides.py             ← Hard post-processing overrides
│   │   └── metrics.py               ← Pearson / Spearman / MAE / F1 metrics
│   ├── features/
│   │   └── pipeline.py              ← Feature extraction + GNN train/infer
│   └── utils/
│       ├── helpers.py               ← ROUGE-L, negation check, Bengali font
│       ├── results_saver.py         ← Run output persistence (CSV, MD, JSON, plots)
│       └── results_logger.py        ← Logging helper
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

Python **3.9 or 3.10** is recommended (PyTorch Geometric wheels are most stable there).

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

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

### 3 — Download BNLP models (first run only)

The BNLP dependency parser downloads its POS model automatically the first time it is used. Make sure you have an internet connection on the first run (or use `--offline` after your initial setup — see below).

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

---

## 🚀 Running the Project

### A. Train (full pipeline)

```bash
python train.py
```

This will:
1. Load `data/dataset_single_sentence.csv` (or whichever CSV is found first)
2. Write / load a per-dataset augmented CSV with 50 adversarial examples
3. Extract KG + GNN + Karak features
4. Train the Siamese GAT-GNN (5 epochs by default)
5. Train the XGBoost + MLP ensemble scorer
6. Apply hard overrides (negation / numeric caps)
7. Print a metric comparison table
8. Save `models/nlp_dominator_gnn_<dataset_stem>.pth` and `models/nlp_dominator_scorer_<dataset_stem>.pkl`
9. Persist a timestamped results folder under `results/train_YYYYMMDD_HHMMSS/`

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset PATH` | auto-detected | Path to a specific CSV file |
| `--epochs N` | `5` | Number of GNN training epochs |
| `--output-dir DIR` | `models` | Directory to save trained models |
| `--output-results DIR` | `results` | Parent folder for the timestamped run-output directory |
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

- Looks for `models/nlp_dominator_gnn_<dataset_stem>.pth` and the matching scorer.
- If dataset-specific weights are missing, falls back to the paths given via `--gnn-model` / `--scorer-model`.
- If no weights are found at all, the GNN is fine-tuned from scratch and saved automatically.

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset PATH` | auto-detected | Dataset CSV to evaluate on |
| `--gnn-model PATH` | `models/nlp_dominator_gnn.pth` | Fallback GNN weights path |
| `--scorer-model PATH` | `models/nlp_dominator_scorer.pkl` | Fallback scorer weights path |
| `--skip-cv` | off | Skip 5-fold cross-validation (much faster) |
| `--output-results DIR` | `results` | Parent folder for the timestamped run-output directory |

```bash
# Full evaluation with 5-fold cross-validation
python evaluate.py

# Skip CV for a quick metrics check
python evaluate.py --skip-cv

# Evaluate on a specific dataset
python evaluate.py --dataset data/dataset_long_sentences.csv

# Evaluate using explicit model paths
python evaluate.py --gnn-model models/nlp_dominator_gnn_dataset_medium_sentences.pth \
                   --scorer-model models/nlp_dominator_scorer_dataset_medium_sentences.pkl
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

> **Important:** The demo looks for `models/nlp_dominator_gnn.pth` and `models/nlp_dominator_scorer.pkl` by default. Run `train.py` first, or copy the dataset-specific model files to those paths. The demo will launch with random weights if no files are found, but scores will be meaningless.

---

### D. Run all 3 datasets automatically

```bash
python run_all_datasets.py
```

Runs `train.py` then `evaluate.py` on **all three datasets one by one**, with separate model and results folders per dataset:

```
models/
├── single/    ← GNN + scorer trained on dataset_single_sentence.csv
├── medium/    ← GNN + scorer trained on dataset_medium_sentences.csv
└── long/      ← GNN + scorer trained on dataset_long_sentences.csv

results/
├── single/    ← metrics, predictions, figures for single-sentence
├── medium/    ← ...
├── long/      ← ...
└── comparison_all_datasets.csv  ← side-by-side Pearson/MAE/F1 table
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `5` | GNN epochs per dataset |
| `--datasets CSV1 CSV2 …` | all 3 | Run only specific CSVs |
| `--skip-cv` | off | Pass `--skip-cv` to `evaluate.py` for each dataset |
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
| `--num-adversarial N` | `50` | Number of adversarial samples at the end of predictions.csv |
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
| `torch` | ≥ 2.0 | Deep learning core |
| `torch-geometric` | ≥ 2.4 | Graph Neural Networks |
| `transformers` | ≥ 4.35 | LaBSE, BanglaBERT, NER models |
| `sentence-transformers` | ≥ 2.2 | Cosine similarity via LaBSE |
| `bnlp_toolkit` | ≥ 3.0 | BNLP POS tagger |
| `indic-nlp-library` | ≥ 0.92 | Bengali text normalisation |
| `xgboost` | ≥ 2.0 | Ensemble scoring |
| `networkx` | ≥ 3.0 | Knowledge graph construction |
| `gradio` | ≥ 4.0 | Interactive demo UI |
| `scikit-learn` | ≥ 1.3 | Cross-validation, metrics |
| `scipy` | ≥ 1.11 | Pearson / Spearman correlation |
| `rouge-score` | ≥ 0.1.2 | ROUGE-L baseline |
| `matplotlib` | ≥ 3.7 | Result figures |
| `seaborn` | ≥ 0.12 | Confusion matrix styling |

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
