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

from src.config import BOOTSTRAP_ROUNDS, PASS_THRESHOLD, RANDOM_STATE


def compute_full_metrics(y_true, y_pred, name: str) -> dict:
    """
    Compute regression and classification metrics for a predictions array.

    Parameters
    ----------
    y_true, y_pred : array-like of float scores in [0, 100]
    name : str  — label used in display tables

    Returns
    -------
    dict with keys: name, n, pearson, spearman, mae, rmse, r2,
                    within_10, within_15, within_20  (% of samples within ±N points),
                    cls_acc, cls_prec, cls_rec,
                    cls_f1        (binary F1 of the PASS class, score >= PASS_THRESHOLD),
                    cls_f1_macro  (macro F1 over PASS / FAIL),
                    y_true_cls, y_pred_cls
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Pearson / Spearman are undefined for a constant prediction
    constant = np.ptp(y_pred) == 0 or np.ptp(y_true) == 0
    p_corr = np.nan if constant else pearsonr(y_true, y_pred)[0]
    s_corr = np.nan if constant else spearmanr(y_true, y_pred)[0]
    mae        = mean_absolute_error(y_true, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_true, y_pred))
    r2         = r2_score(y_true, y_pred)

    within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    within_15 = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100

    y_true_cls = (y_true >= PASS_THRESHOLD).astype(int)
    y_pred_cls = (y_pred >= PASS_THRESHOLD).astype(int)

    return {
        'name': name, 'n': len(y_true),
        'pearson':  p_corr, 'spearman': s_corr,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'within_10': within_10, 'within_15': within_15, 'within_20': within_20,
        'cls_acc':  accuracy_score(y_true_cls, y_pred_cls) * 100,
        'cls_prec': precision_score(y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_rec':  recall_score   (y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_f1':   f1_score       (y_true_cls, y_pred_cls, zero_division=0) * 100,
        'cls_f1_macro': f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0) * 100,
        'y_true_cls': y_true_cls, 'y_pred_cls': y_pred_cls,
    }


def _pearson(y, p):
    return np.nan if np.ptp(p) == 0 or np.ptp(y) == 0 else pearsonr(y, p)[0]


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def paired_bootstrap(y_true, pred_a, pred_b, n_rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = RANDOM_STATE) -> dict:
    """
    Bootstrap 95% confidence intervals for model A, and for A − B, on the
    same resampled rows.

    Returns a dict with, for 'pearson' and 'mae': A's estimate and CI, the
    difference A − B with its CI, and a two-sided p-value for "no
    difference" (share of resamples whose difference has the other sign).
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_rounds, len(y)))
    out = {}
    for name, fn in (('pearson', _pearson), ('mae', _mae)):
        va = np.array([fn(y[i], a[i]) for i in idx])
        vd = va - np.array([fn(y[i], b[i]) for i in idx])
        va, vd = va[~np.isnan(va)], vd[~np.isnan(vd)]
        diff = fn(y, a) - fn(y, b)
        p = 2 * min(np.mean(vd <= 0), np.mean(vd >= 0)) if len(vd) else np.nan
        out[name] = {
            'value': fn(y, a), 'ci_low': np.percentile(va, 2.5), 'ci_high': np.percentile(va, 97.5),
            'diff': diff, 'diff_ci_low': np.percentile(vd, 2.5), 'diff_ci_high': np.percentile(vd, 97.5),
            'p_value': min(1.0, p),
        }
    return out


def _fmt(v, spec):
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def print_regression_table(all_metrics: list, title: str = 'Test Set'):
    """Print formatted tables for all metrics dicts in *all_metrics*."""
    n = all_metrics[0].get('n', '?') if all_metrics else '?'
    print('\n' + '=' * 100)
    print(f'  REGRESSION METRICS ({title}, n={n})')
    print('=' * 100)
    print(f"  {'Model':<38} | {'Pearson':>8} | {'Spearman':>8} | {'MAE':>6} | {'RMSE':>6} | {'R^2':>6}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {_fmt(m['pearson'], '>8.4f')} | {_fmt(m['spearman'], '>8.4f')} "
              f"| {m['mae']:>6.2f} | {m['rmse']:>6.2f} | {m['r2']:>6.4f}")
    print('=' * 100)

    print('\n' + '=' * 78)
    print('  TOLERANCE-BASED ACCURACY (% of samples within ±N score points)')
    print('=' * 78)
    print(f"  {'Model':<38} | {'±10':>8} | {'±15':>8} | {'±20':>8}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['within_10']:>7.1f}% | {m['within_15']:>7.1f}% | {m['within_20']:>7.1f}%")
    print('=' * 78)

    print('\n' + '=' * 98)
    print(f'  CLASSIFICATION METRICS (>={PASS_THRESHOLD} pass, <{PASS_THRESHOLD} fail; '
          f'F1 = PASS class, macro-F1 = both classes)')
    print('=' * 98)
    print(f"  {'Model':<38} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'Macro F1':>8}")
    print(f"  {'-'*38}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<38} | {m['cls_acc']:>7.1f}% | {m['cls_prec']:>8.1f}% "
              f"| {m['cls_rec']:>7.1f}% | {m['cls_f1']:>7.1f}% | {m['cls_f1_macro']:>7.1f}%")
    print('=' * 98)
