"""
metrics.py
==========
Evaluation metrics for the Bengali Answer Evaluation System.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)


def compute_full_metrics(y_true, y_pred, name: str) -> dict:
    """
    Compute regression and classification metrics for a predictions array.

    Parameters
    ----------
    y_true, y_pred : array-like of float scores in [0, 100]
    name : str  — label used in display tables

    Returns
    -------
    dict with keys: name, pearson, spearman, mae, rmse, r2,
                    within_10, within_15, within_20,
                    cls_acc, cls_prec, cls_rec, cls_f1,
                    y_true_cls, y_pred_cls
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    mae        = mean_absolute_error(y_true, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_true, y_pred))
    r2         = r2_score(y_true, y_pred)

    within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    within_15 = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100

    y_true_cls = (y_true >= 50).astype(int)
    y_pred_cls = (y_pred >= 50).astype(int)

    return {
        'name': name,
        'pearson':  p_corr, 'spearman': s_corr,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'within_10': within_10, 'within_15': within_15, 'within_20': within_20,
        'cls_acc':  accuracy_score(y_true_cls, y_pred_cls) * 100,
        'cls_prec': precision_score(y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_rec':  recall_score   (y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_f1':   f1_score       (y_true_cls, y_pred_cls, zero_division=0) * 100,
        'y_true_cls': y_true_cls, 'y_pred_cls': y_pred_cls,
    }


def print_regression_table(all_metrics: list):
    """Print formatted tables for all metrics dicts in *all_metrics*."""
    print('\n' + '=' * 100)
    print('  REGRESSION METRICS (Test Set)')
    print('=' * 100)
    print(f"  {'Model':<38} | {'Pearson':>8} | {'Spearman':>8} | {'MAE':>6} | {'RMSE':>6} | {'R^2':>6}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['pearson']:>8.4f} | {m['spearman']:>8.4f} "
              f"| {m['mae']:>6.2f} | {m['rmse']:>6.2f} | {m['r2']:>6.4f}")
    print('=' * 100)

    print('\n' + '=' * 78)
    print('  TOLERANCE-BASED ACCURACY')
    print('=' * 78)
    print(f"  {'Model':<38} | {'+/-10':>8} | {'+/-15':>8} | {'+/-20':>8}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['within_10']:>7.1f}% | {m['within_15']:>7.1f}% | {m['within_20']:>7.1f}%")
    print('=' * 78)

    print('\n' + '=' * 85)
    print('  CLASSIFICATION METRICS (>=50 correct, <50 incorrect)')
    print('=' * 85)
    print(f"  {'Model':<38} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['cls_acc']:>7.1f}% | {m['cls_prec']:>8.1f}% "
              f"| {m['cls_rec']:>7.1f}% | {m['cls_f1']:>7.1f}%")
    print('=' * 85)
