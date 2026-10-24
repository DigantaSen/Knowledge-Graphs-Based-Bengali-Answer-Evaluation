"""
train.py
========
End-to-end training script for the Bengali Answer Evaluation System.

Usage
-----
    python train.py [--dataset PATH] [--epochs N] [--output-dir DIR]
                    [--group-by pair|question] [--offline]

This script reproduces the full training pipeline:
  1. Loads, augments, deduplicates and splits the dataset
  2. Adds synthetic negation / role-swap / wrong-number rows to the training split
  3. Initialises all NLP/graph components
  4. Extracts features for train and test sets
  5. Trains the Siamese GAT-GNN (epochs controlled by --epochs)
  6. Trains the XGBoost + MLP ensemble scorer on out-of-fold GNN similarities
  7. Applies hard overrides on the test set
  8. Prints evaluation metrics (full test set, normal rows, non-templated rows,
     adversarial rows) with bootstrap confidence intervals
  9. Saves trained models to --output-dir
"""

import argparse
import os

from src.config import (
    GNN_EPOCHS, MODELS_DIR, NUM_ADVERSARIAL, RANDOM_STATE, SPLIT_GROUP_BY, TEST_SIZE,
    gnn_model_filename, scorer_model_filename,
)
from src.data.dataset import normal_rows, split_groups, template_phrases
from src.experiment import (
    build_components, calibrated, fit_gnn_and_scorer, load_split, predict, report, with_synthetic,
)
from src.features.pipeline import create_graphs_and_features
from src.utils.helpers import ensure_utf8_stdio, set_seed, setup_bengali_font
from src.utils.results_saver import ResultsSaver

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Bengali Answer Evaluation — Training')
    parser.add_argument('--dataset',        type=str, default=None,
                        help='Path to the original dataset CSV (auto-detected if omitted)')
    parser.add_argument('--epochs',         type=int, default=GNN_EPOCHS,
                        help=f'GNN training epochs (default: {GNN_EPOCHS})')
    parser.add_argument('--output-dir',     type=str, default=MODELS_DIR,
                        help=f'Directory to save trained models (default: {MODELS_DIR}/)')
    parser.add_argument('--output-results', type=str, default='results',
                        help='Parent directory for run result folders (default: results)')
    parser.add_argument('--group-by',       choices=('pair', 'question'), default=SPLIT_GROUP_BY,
                        help='Split unit: identical answer pairs (unseen answers) or '
                             'whole questions (unseen questions)')
    parser.add_argument('--offline',        action='store_true',
                        help='Force offline mode — use only locally cached HuggingFace models')
    return parser.parse_args()


def main():
    ensure_utf8_stdio()
    args = parse_args()
    set_seed(RANDOM_STATE)

    # ── Offline mode: use only cached HuggingFace models ────────────────────
    if args.offline:
        os.environ['HF_HUB_OFFLINE']       = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print('[INFO] Offline mode — using locally cached models only.')

    setup_bengali_font()

    # ── Results saver (starts log capture immediately) ────────────────────────
    saver = ResultsSaver(base_dir=args.output_results, label='train')
    saver.start_log_capture()

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    print('\n' + '=' * 57)
    print('  Bengali Answer Evaluation — Training')
    print('=' * 57)
    original_csv, src_stem, _, df_augmented, train_df, test_df = load_split(args.dataset, args.group_by)

    # ── 2. Components ────────────────────────────────────────────────────────
    print('\n[1/6] Initializing components...')
    comps = build_components()

    # ── 3. Training set (+ synthetic errors) and features ────────────────────
    print('\n[2/6] Extracting TRAINING features...')
    train_fit = with_synthetic(train_df, comps['dep_parser'])
    tr = create_graphs_and_features(train_fit, comps)
    y_train   = train_fit['human_score'].values.astype(float)
    groups_tr = split_groups(train_fit, args.group_by)

    # ── 4. GNN + scorer ──────────────────────────────────────────────────────
    print(f'\n[3/6] Training Siamese GAT-GNN ({args.epochs} epochs) and the ensemble scorer...')
    gnn, scorer, gsims_tr = fit_gnn_and_scorer(tr, y_train, groups_tr, args.epochs)

    # ── 5. Test features + predictions ───────────────────────────────────────
    print('\n[4/6] Extracting TEST features...')
    te = create_graphs_and_features(test_df, comps)
    y_test = test_df['human_score'].values.astype(float)
    gsims_te, y_pred_raw, y_pred_proposed, override_log = predict(gnn, scorer, te)
    print(f'\n[OK] Hard overrides applied to {sum(1 for r in override_log if r)}/{len(override_log)} samples.')

    # ── 6. Evaluate (baselines calibrated on the same training rows) ─────────
    print('\n[5/6] Evaluating...')
    predictions = [
        ('Baseline (LaBSE only)',       calibrated([tr['base_sim']], y_train, [te['base_sim']])),
        ('Baseline (ROUGE-L only)',     calibrated([tr['rouge_l']], y_train, [te['rouge_l']])),
        ('LaBSE + KG + GAT-GNN',        calibrated([tr['base_sim'], gsims_tr, tr['entity_mm']], y_train,
                                                   [te['base_sim'], gsims_te, te['entity_mm']])),
        ('LaBSE + KG + GAT + Karak',    calibrated([tr['base_sim'], gsims_tr, tr['penalty']], y_train,
                                                   [te['base_sim'], gsims_te, te['penalty']])),
        ('Full Ensemble (no override)', y_pred_raw),
        ('Proposed (Full + Overrides)', y_pred_proposed),
    ]
    phrases = template_phrases(normal_rows(df_augmented))
    all_metrics = report(saver, predictions, y_test, test_df, phrases,
                         proposed='Proposed (Full + Overrides)')

    b, p = all_metrics[0], all_metrics[-1]
    print('\n' + '=' * 60)
    print('  IMPROVEMENT: PROPOSED vs BASELINE (full test set)')
    print('=' * 60)
    print(f"  Pearson:    {b['pearson']:.4f} -> {p['pearson']:.4f}  ({p['pearson']-b['pearson']:+.4f})")
    print(f"  MAE:        {b['mae']:.2f} -> {p['mae']:.2f}  ({p['mae']-b['mae']:+.2f})")
    print(f"  F1:         {b['cls_f1']:.1f}% -> {p['cls_f1']:.1f}%  ({p['cls_f1']-b['cls_f1']:+.1f}%)")
    print(f"  Acc+/-10:   {b['within_10']:.1f}% -> {p['within_10']:.1f}%  ({p['within_10']-b['within_10']:+.1f}%)")
    print('=' * 60)

    # ── 7. Save models ────────────────────────────────────────────────────────
    print('\n[6/6] Saving models...')
    os.makedirs(args.output_dir, exist_ok=True)
    gnn_path    = os.path.join(args.output_dir, gnn_model_filename(src_stem))
    scorer_path = os.path.join(args.output_dir, scorer_model_filename(src_stem))
    torch.save(gnn.state_dict(), gnn_path)
    print(f'[OK] GNN saved to {gnn_path}')
    scorer.save_model(scorer_path)

    # ── 8. Persist run outputs for report ────────────────────────────────────
    saver.set_config({
        'script':          'train.py',
        'dataset':         original_csv,
        'epochs':          args.epochs,
        'random_state':    RANDOM_STATE,
        'test_size':       TEST_SIZE,
        'num_adversarial': NUM_ADVERSARIAL,
        'split_group_by':  args.group_by,
        'n_train':         len(train_df),
        'n_train_synthetic': int(len(train_fit) - len(train_df)),
        'n_test':          len(test_df),
        'ensemble_weights': list(scorer.ensemble_weights),
        'output_dir':      args.output_dir,
    })
    saver.add_predictions(test_df, y_test, y_pred_raw, y_pred_proposed, override_log)
    saver.add_feature_importances(scorer)
    saver.save_all(num_adversarial=NUM_ADVERSARIAL)


if __name__ == '__main__':
    main()
