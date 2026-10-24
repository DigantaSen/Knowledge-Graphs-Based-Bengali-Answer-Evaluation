"""
evaluate.py
===========
Stand-alone evaluation script for the Bengali Answer Evaluation System.

Loads pretrained models and evaluates them on the test split of the
augmented dataset. Also runs 5-fold cross-validation on the normal samples.

Usage
-----
    python evaluate.py [--dataset PATH] [--models-dir DIR]
                       [--gnn-model PATH] [--scorer-model PATH] [--epochs N]
                       [--group-by pair|question] [--skip-cv] [--offline]
"""

import argparse
import os
import shutil

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

import torch

from src.config import (
    CV_N_SPLITS, GNN_EPOCHS, MODELS_DIR, NUM_ADVERSARIAL, PASS_THRESHOLD,
    RANDOM_STATE, SPLIT_GROUP_BY, TEST_SIZE,
    gnn_model_filename, scorer_model_filename,
)
from src.data.dataset import normal_rows, split_groups, template_phrases
from src.experiment import (
    build_components, calibrated, fit_gnn_and_scorer, load_split, predict, report, with_synthetic,
)
from src.features.pipeline import (
    concat_features, create_graphs_and_features, get_gnn_similarities, subset_features,
)
from src.models.answer_scorer import AnswerScorer
from src.models.siamese_gnn import SiameseGNN
from src.utils.helpers import ensure_utf8_stdio, set_seed
from src.utils.results_saver import ResultsSaver


# ── HuggingFace auto-download ──────────────────────────────────────────────────

_HF_REPO_ID = "k040902/bengali_question_ans"

# Original-data models and original + clean-synthetic models, per tier
_HF_STEMS = [f'dataset_{t}{suffix}' for suffix in ('', '_plus_clean')
             for t in ('single_sentence', 'medium_sentences', 'long_sentences')]
_HF_ALL_MODEL_FILES = ([gnn_model_filename(s) for s in _HF_STEMS]
                       + [scorer_model_filename(s) for s in _HF_STEMS])


def _ensure_model_downloaded(ds_stem: str, models_dir: str = MODELS_DIR) -> None:
    """
    Ensure the GNN (.pth) and scorer (.npz) for *ds_stem* exist in *models_dir*.
    Downloads only the missing files from HuggingFace; unknown stems are skipped.
    """
    os.makedirs(models_dir, exist_ok=True)
    stem_files = [f for f in _HF_ALL_MODEL_FILES if f.endswith(f'_{ds_stem}.pth')
                  or f.endswith(f'_{ds_stem}.npz')]
    missing = [f for f in stem_files if not os.path.exists(os.path.join(models_dir, f))]
    if not stem_files or not missing:
        if stem_files:
            print(f'[INFO] Model files for "{ds_stem}" already present in {models_dir}/')
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('[WARN] huggingface_hub not installed. '
              'Run: pip install huggingface_hub\n'
              '      Model download skipped.')
        return

    print(f'[INFO] {len(missing)} model file(s) missing for "{ds_stem}". '
          f'Downloading from {_HF_REPO_ID} …')
    for filename in missing:
        dest = os.path.join(models_dir, filename)
        print(f'  ↓  {filename} …', end=' ', flush=True)
        try:
            cached_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=filename,
                                          repo_type='model', local_dir=models_dir)
            # With local_dir the file usually already sits at dest; copying a
            # file onto itself raised WinError 32 and was reported as FAILED.
            if not os.path.exists(dest):
                shutil.copy2(cached_path, dest)
            print(f'done  →  {dest}')
        except Exception as e:
            print(f'FAILED ({e})')
    print('[INFO] Download step complete.')


def parse_args():
    p = argparse.ArgumentParser(description='Bengali Answer Evaluation — Evaluation')
    p.add_argument('--dataset',        type=str, default=None)
    p.add_argument('--models-dir',     type=str, default=MODELS_DIR,
                   help=f'Directory with the per-dataset model files (default: {MODELS_DIR})')
    p.add_argument('--gnn-model',      type=str, default=None,
                   help='GNN weights (default: <models-dir>/nlp_dominator_gnn_<dataset_stem>.pth)')
    p.add_argument('--scorer-model',   type=str, default=None,
                   help='Scorer (default: <models-dir>/nlp_dominator_scorer_<dataset_stem>.npz)')
    p.add_argument('--epochs',         type=int, default=GNN_EPOCHS,
                   help=f'GNN epochs when training is needed, incl. CV (default: {GNN_EPOCHS})')
    p.add_argument('--group-by',       choices=('pair', 'question'), default=SPLIT_GROUP_BY,
                   help='Split unit: identical answer pairs or whole questions')
    p.add_argument('--skip-cv',        action='store_true',
                   help='Skip 5-fold cross-validation (much faster)')
    p.add_argument('--output-results', type=str, default='results',
                   help='Parent directory for run result folders (default: results)')
    p.add_argument('--offline',        action='store_true',
                   help='Force offline mode — use only locally cached HuggingFace models')
    return p.parse_args()


def _run_cv(df_augmented, comps, args, saver):
    """Grouped 5-fold CV of the full pipeline on the normal (deduplicated) rows."""
    df_normal = normal_rows(df_augmented)
    groups_cv = split_groups(df_normal, args.group_by)
    print(f'\n{"="*75}')
    print(f'  5-FOLD CROSS-VALIDATION (non-adversarial, proposed model)')
    print(f'{"="*75}')
    print(f'  CV pool size: {len(df_normal)} (adversarial excluded, grouped by {args.group_by})\n')

    # Only the GNN and scorer are learned; every other feature is fixed per
    # row, so features are extracted once.  Synthetic rows are built once from
    # all references and a fold uses those whose question is in its train part.
    cv   = create_graphs_and_features(df_normal, comps)
    syn_df = with_synthetic(df_normal, comps['dep_parser']).iloc[len(df_normal):]
    syn  = create_graphs_and_features(syn_df, comps) if len(syn_df) else None
    y_cv = df_normal['human_score'].values.astype(float)

    cv_pearsons, cv_maes, cv_f1s = [], [], []
    folds = GroupKFold(n_splits=CV_N_SPLITS).split(df_normal, groups=groups_cv)
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f'  Fold {fold_idx+1}/{CV_N_SPLITS} — train={len(tr_idx)}, val={len(val_idx)}')
        f_tr, f_va = subset_features(cv, tr_idx), subset_features(cv, val_idx)
        y_f_tr, y_f_va = y_cv[tr_idx], y_cv[val_idx]
        g_f_tr = groups_cv[tr_idx]
        if syn is not None:
            keep = np.where(syn_df['question'].isin(set(df_normal['question'].iloc[tr_idx])))[0]
            f_tr = concat_features(f_tr, subset_features(syn, keep))
            y_f_tr = np.concatenate([y_f_tr, syn_df['human_score'].values[keep]])
            g_f_tr = np.concatenate([g_f_tr, split_groups(syn_df.iloc[keep], args.group_by)])

        # Fresh GNN every fold: the saved weights were trained on rows that
        # sit in this fold's validation set, so warm-starting from them leaks.
        gnn_cv, sc_cv, _ = fit_gnn_and_scorer(f_tr, y_f_tr, g_f_tr, args.epochs,
                                              seed=RANDOM_STATE + 100 * (fold_idx + 1))
        _, _, y_f_va_pred, _ = predict(gnn_cv, sc_cv, f_va)

        pc, _ = pearsonr(y_f_va, y_f_va_pred)
        mae   = mean_absolute_error(y_f_va, y_f_va_pred)
        f1    = f1_score((y_f_va >= PASS_THRESHOLD).astype(int),
                         (y_f_va_pred >= PASS_THRESHOLD).astype(int), zero_division=0) * 100
        cv_pearsons.append(pc); cv_maes.append(mae); cv_f1s.append(f1)
        print(f'    Pearson={pc:.4f}  MAE={mae:.2f}  F1={f1:.1f}%\n')
        saver.add_cv_fold(fold_idx + 1, pc, mae, f1)

    print(f'\n{"="*75}')
    print(f'  CV SUMMARY (mean ± std)')
    print(f'{"="*75}')
    print(f'  Pearson:  {np.mean(cv_pearsons):.4f} ± {np.std(cv_pearsons):.4f}')
    print(f'  MAE:      {np.mean(cv_maes):.2f} ± {np.std(cv_maes):.2f}')
    print(f'  F1:       {np.mean(cv_f1s):.1f}% ± {np.std(cv_f1s):.1f}%')
    print(f'{"="*75}')
    saver.add_cv_summary(cv_pearsons, cv_maes, cv_f1s)


def main():
    ensure_utf8_stdio()
    args = parse_args()
    set_seed(RANDOM_STATE)

    # ── Offline mode: use only cached HuggingFace models ────────────────────
    if args.offline:
        os.environ['HF_HUB_OFFLINE']       = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print('[INFO] Offline mode — using locally cached models only.')

    # ── Results saver (starts log capture immediately) ────────────────────────
    saver = ResultsSaver(base_dir=args.output_results, label='evaluate')
    saver.start_log_capture()

    # ── Dataset ──────────────────────────────────────────────────────────────
    original_csv, src_stem, augmented_csv, df_augmented, train_df, test_df = \
        load_split(args.dataset, args.group_by)

    # ── Dataset-specific model paths ─────────────────────────────────────────
    gnn_savepath    = args.gnn_model or os.path.join(args.models_dir, gnn_model_filename(src_stem))
    scorer_savepath = args.scorer_model or os.path.join(args.models_dir, scorer_model_filename(src_stem))
    if not (args.gnn_model or args.scorer_model):
        _ensure_model_downloaded(src_stem, models_dir=args.models_dir)

    # ── Components ────────────────────────────────────────────────────────────
    comps = build_components()

    # ── Load GNN + scorer (a scorer is only valid with the GNN it was fit on) ─
    gnn, scorer, loaded = SiameseGNN(), AnswerScorer(), False
    if os.path.exists(gnn_savepath) and os.path.exists(scorer_savepath):
        try:
            gnn.load_state_dict(torch.load(gnn_savepath, map_location='cpu', weights_only=True))
            scorer.load_model(scorer_savepath)
            loaded = True
            print(f'[OK] Models loaded: {gnn_savepath}, {scorer_savepath}')
        except Exception as e:
            print(f'[WARN] Could not load models ({e}). Will train.')
    else:
        print(f'[WARN] No trained models at {gnn_savepath} / {scorer_savepath}. Will train.')

    # ── Train split features (always needed: they calibrate the baselines) ───
    print('\nExtracting training features...')
    train_fit = with_synthetic(train_df, comps['dep_parser'])
    tr        = create_graphs_and_features(train_fit, comps)
    y_train   = train_fit['human_score'].values.astype(float)

    if loaded:
        gsims_tr = get_gnn_similarities(gnn, tr['ref_pyg'], tr['stu_pyg'])
    else:
        gnn, scorer, gsims_tr = fit_gnn_and_scorer(
            tr, y_train, split_groups(train_fit, args.group_by), args.epochs)
        for path in (gnn_savepath, scorer_savepath):
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(gnn.state_dict(), gnn_savepath)
        scorer.save_model(scorer_savepath)
        print(f'[INFO] Models saved to {gnn_savepath}, {scorer_savepath}')

    print('\nExtracting test features...')
    te       = create_graphs_and_features(test_df, comps)
    y_test   = test_df['human_score'].values.astype(float)
    gsims_te, y_pred_raw, y_pred_proposed, override_log = predict(gnn, scorer, te)

    # ── BanglaBERT cosine similarity baseline ────────────────────────────────
    print('\n[INFO] Computing BanglaBERT cosine similarity baseline...')
    em = comps['embedder']
    bb_tr = np.array([em.banglabert_similarity(r, s) for r, s in
                      zip(train_fit['reference_answer'].astype(str), train_fit['student_answer'].astype(str))])
    bb_te = np.array([em.banglabert_similarity(r, s) for r, s in
                      zip(test_df['reference_answer'].astype(str), test_df['student_answer'].astype(str))])

    # ── Baselines (each similarity calibrated on the train split) ───────────
    # GNN-only uses in-sample similarities of a loaded GNN when no OOF values
    # exist; it is a 1-D linear calibration, so the effect is small.
    predictions = [
        ('Baseline: LaBSE Cosine Sim',      calibrated([tr['base_sim']], y_train, [te['base_sim']])),
        ('Baseline: ROUGE-L Only',          calibrated([tr['rouge_l']], y_train, [te['rouge_l']])),
        ('Baseline: LaBSE + ROUGE-L',       calibrated([tr['base_sim'], tr['rouge_l']], y_train,
                                                       [te['base_sim'], te['rouge_l']])),
        ('Baseline: BanglaBERT Cosine Sim', calibrated([bb_tr], y_train, [bb_te])),
        ('Ablation: GNN Only',              calibrated([gsims_tr], y_train, [gsims_te])),
        ('Proposed: Full Ensemble (no override)', y_pred_raw),
        ('Proposed: Full + Overrides (ours)',     y_pred_proposed),
    ]
    phrases = template_phrases(normal_rows(df_augmented))
    report(saver, predictions, y_test, test_df, phrases, proposed='Proposed: Full + Overrides (ours)')

    # ── Register test results with saver ──────────────────────────────────────
    saver.set_config({
        'script':          'evaluate.py',
        'dataset':         augmented_csv,
        'gnn_model':       gnn_savepath,
        'scorer_model':    scorer_savepath,
        'models_loaded':   loaded,
        'epochs':          args.epochs,
        'random_state':    RANDOM_STATE,
        'test_size':       TEST_SIZE,
        'num_adversarial': NUM_ADVERSARIAL,
        'split_group_by':  args.group_by,
        'n_test':          len(test_df),
        'ensemble_weights': list(scorer.ensemble_weights),
        'template_phrases': phrases,
        'skip_cv':         args.skip_cv,
    })
    saver.add_predictions(test_df, y_test, y_pred_raw, y_pred_proposed, override_log)
    saver.add_feature_importances(scorer)

    if not args.skip_cv:
        _run_cv(df_augmented, comps, args, saver)

    saver.save_all(num_adversarial=NUM_ADVERSARIAL)


if __name__ == '__main__':
    main()
