"""
evaluation.py - Comprehensive evaluation: metrics tables, visualisation,
cross-validation, per-subject / per-category breakdown.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from src.config import (
    CV_SPLITS, RANDOM_STATE,
    PLOT_CONFUSION, PLOT_ERROR_DIST, PLOT_SCORE_DIST,
    PLOT_ABLATION, PLOT_SCATTER, PLOT_SUBJECT, PLOT_INCREMENTAL,
)
from src.gnn_model import SiameseGNN
from src.scorer import AnswerScorer
from src.pipeline import (
    create_graphs_and_features, train_gnn, get_gnn_similarities,
)

# ── per-sample metrics ────────────────────────────────────────────────────────

def compute_full_metrics(y_true, y_pred, name: str) -> dict:
    """Compute regression + classification metrics and return as a dict."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    within_15 = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100

    y_true_cls = (y_true >= 50).astype(int)
    y_pred_cls = (y_pred >= 50).astype(int)
    cls_acc  = accuracy_score(y_true_cls, y_pred_cls) * 100
    cls_prec = precision_score(y_true_cls, y_pred_cls, zero_division=0) * 100
    cls_rec  = recall_score(y_true_cls, y_pred_cls, zero_division=0) * 100
    cls_f1   = f1_score(y_true_cls, y_pred_cls, zero_division=0) * 100

    return {
        "name": name,
        "pearson": p_corr, "spearman": s_corr,
        "mae": mae, "rmse": rmse, "r2": r2,
        "within_10": within_10, "within_15": within_15, "within_20": within_20,
        "cls_acc": cls_acc, "cls_prec": cls_prec, "cls_rec": cls_rec, "cls_f1": cls_f1,
        "y_true_cls": y_true_cls, "y_pred_cls": y_pred_cls,
    }

# ── tables ────────────────────────────────────────────────────────────────────

def print_all_metric_tables(all_metrics: list) -> None:
    """Print regression, tolerance, classification and ablation tables."""

    # Regression metrics
    print("\n" + "=" * 95)
    print("  REGRESSION METRICS (Test Set)")
    print("=" * 95)
    print(f"  {'Model':<35} | {'Pearson':>8} | {'Spearman':>8} | {'MAE':>6} | {'RMSE':>6} | {'R²':>6}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for m in all_metrics:
        print(f"  {m['name']:<35} | {m['pearson']:>8.4f} | {m['spearman']:>8.4f} | {m['mae']:>6.2f} | {m['rmse']:>6.2f} | {m['r2']:>6.4f}")
    print("=" * 95)

    # Tolerance-based accuracy
    print("\n" + "=" * 75)
    print("  TOLERANCE-BASED ACCURACY (% within ±N of human score)")
    print("=" * 75)
    print(f"  {'Model':<35} | {'±10 pts':>8} | {'±15 pts':>8} | {'±20 pts':>8}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<35} | {m['within_10']:>7.1f}% | {m['within_15']:>7.1f}% | {m['within_20']:>7.1f}%")
    print("=" * 75)

    # Classification metrics
    print("\n" + "=" * 80)
    print("  CLASSIFICATION METRICS (Correct ≥50 vs Incorrect <50)")
    print("=" * 80)
    print(f"  {'Model':<35} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")
    for m in all_metrics:
        print(f"  {m['name']:<35} | {m['cls_acc']:>7.1f}% | {m['cls_prec']:>8.1f}% | {m['cls_rec']:>7.1f}% | {m['cls_f1']:>7.1f}%")
    print("=" * 80)

    # Incremental ablation
    print("\n" + "=" * 95)
    print("  INCREMENTAL ABLATION: COMPONENT CONTRIBUTION")
    print("=" * 95)
    print(f"  {'Step':<4} | {'Configuration':<35} | {'Pearson':>8} | {'MAE':>6} | {'F1':>7} | {'ΔPearson':>9} | {'ΔMAE':>7} | {'ΔF1':>7}")
    print(f"  {'-'*4}-+-{'-'*35}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}")
    prev_p = prev_mae = prev_f1 = None
    steps = ["B", "+E", "+G", "+K", "+X"]
    for i, m in enumerate(all_metrics):
        dp  = f"{m['pearson']-prev_p:+.4f}"  if prev_p   is not None else "   —    "
        dm  = f"{m['mae']-prev_mae:+.2f}"    if prev_mae is not None else "  —   "
        df1 = f"{m['cls_f1']-prev_f1:+.1f}%" if prev_f1  is not None else "  —    "
        print(f"  {steps[i]:<4} | {m['name']:<35} | {m['pearson']:>8.4f} | {m['mae']:>6.2f} | {m['cls_f1']:>6.1f}% | {dp:>9} | {dm:>7} | {df1:>7}")
        prev_p, prev_mae, prev_f1 = m["pearson"], m["mae"], m["cls_f1"]

    print("=" * 95)
    print("  Key: B=Baseline(LaBSE), +E=Entity, +G=GAT-GNN, +K=Karak, +X=XGBoost+MLP")

    # Improvement summary
    b, p = all_metrics[0], all_metrics[-1]
    print("\n" + "=" * 60)
    print("  IMPROVEMENT OF PROPOSED MODEL OVER BASELINE")
    print("=" * 60)
    for label, bv, pv in [
        ("Pearson",   b["pearson"],  p["pearson"]),
        ("Spearman",  b["spearman"], p["spearman"]),
        ("MAE",       b["mae"],      p["mae"]),
        ("RMSE",      b["rmse"],     p["rmse"]),
        ("R²",        b["r2"],       p["r2"]),
    ]:
        denom = max(abs(bv), 0.001)
        print(f"  {label:<10}: {bv:.4f} → {pv:.4f}  ({(pv-bv)/denom*100:+.1f}%)")
    print(f"  {'F1':10}: {b['cls_f1']:.1f}% → {p['cls_f1']:.1f}%  ({p['cls_f1']-b['cls_f1']:+.1f} pp)")
    print(f"  {'Acc±10':10}: {b['within_10']:.1f}% → {p['within_10']:.1f}%  ({p['within_10']-b['within_10']:+.1f} pp)")
    print("=" * 60)

# ── cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(
    df_normal,
    text_processor,
    dep_parser, triple_extractor, kg_constructor,
    embedder, validator,
    coref_resolver=None, ner_pipeline_obj=None,
    n_splits: int = CV_SPLITS,
    gnn_epochs: int = 3,
):
    """5-fold cross-validation using only non-adversarial samples."""
    print("\n" + "=" * 75)
    print(f"  {n_splits}-FOLD CROSS-VALIDATION (non-adversarial samples, proposed model)")
    print("=" * 75)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_df = df_normal.reset_index(drop=True)
    cv_pearson, cv_spearman, cv_mae, cv_f1 = [], [], [], []

    for fold_idx, (tr_idx, vl_idx) in enumerate(kf.split(cv_df), 1):
        print(f"\n  Fold {fold_idx}/{n_splits} — train={len(tr_idx)}, val={len(vl_idx)}")
        fold_train = cv_df.iloc[tr_idx].reset_index(drop=True)
        fold_val   = cv_df.iloc[vl_idx].reset_index(drop=True)

        bsims_tr, rpyg_tr, spyg_tr, pen_tr, emm_tr = create_graphs_and_features(
            fold_train, text_processor, dep_parser, triple_extractor,
            kg_constructor, embedder, validator,
            coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipeline_obj,
        )
        bsims_vl, rpyg_vl, spyg_vl, pen_vl, emm_vl = create_graphs_and_features(
            fold_val, text_processor, dep_parser, triple_extractor,
            kg_constructor, embedder, validator,
            coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipeline_obj,
        )

        y_tr = fold_train["human_score"].values.astype(float)
        y_vl = fold_val["human_score"].values.astype(float)

        fold_gnn = SiameseGNN()
        fold_gnn = train_gnn(fold_gnn, rpyg_tr, spyg_tr, y_tr, epochs=gnn_epochs)
        gsims_tr = get_gnn_similarities(fold_gnn, rpyg_tr, spyg_tr)
        gsims_vl = get_gnn_similarities(fold_gnn, rpyg_vl, spyg_vl)

        fold_scorer = AnswerScorer()
        fold_scorer.train(
            np.column_stack((bsims_tr, gsims_tr, pen_tr, emm_tr)), y_tr
        )
        y_pred_vl = fold_scorer.predict_batch(
            np.column_stack((bsims_vl, gsims_vl, pen_vl, emm_vl))
        )

        pc, _ = pearsonr(y_vl, y_pred_vl)
        sc, _ = spearmanr(y_vl, y_pred_vl)
        mae_v = mean_absolute_error(y_vl, y_pred_vl)
        f1_v  = f1_score((y_vl >= 50).astype(int),
                         (y_pred_vl >= 50).astype(int), zero_division=0) * 100

        cv_pearson.append(pc); cv_spearman.append(sc)
        cv_mae.append(mae_v);  cv_f1.append(f1_v)
        print(f"    Pearson={pc:.4f}  Spearman={sc:.4f}  MAE={mae_v:.2f}  F1={f1_v:.1f}%")

    print(f"\n  {'─'*65}")
    print(f"  {'Metric':<12} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print(f"  {'─'*12}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")
    for label, vals in [
        ("Pearson", cv_pearson), ("Spearman", cv_spearman),
        ("MAE", cv_mae), ("F1 (%)", cv_f1),
    ]:
        print(f"  {label:<12} | {np.mean(vals):>8.4f} | {np.std(vals):>8.4f} | {np.min(vals):>8.4f} | {np.max(vals):>8.4f}")
    print(f"  {'═'*65}")
    print(f"  CV Pearson: {np.mean(cv_pearson):.4f} ± {np.std(cv_pearson):.4f}")
    print(f"  CV MAE:     {np.mean(cv_mae):.2f} ± {np.std(cv_mae):.2f}")
    print("=" * 75)

# ── per-subject breakdown ─────────────────────────────────────────────────────

def per_subject_breakdown(test_df, y_test, y_pred_baseline, y_pred_proposed,
                           adv_start: int):
    """Print per-subject Pearson, MAE, F1 table."""
    test_subj_df   = test_df.iloc[:adv_start].copy()
    y_test_s       = y_test[:adv_start]
    y_base_s       = y_pred_baseline[:adv_start]
    y_prop_s       = y_pred_proposed[:adv_start]

    print("\n" + "=" * 85)
    print("  PER-SUBJECT PERFORMANCE (Proposed vs Baseline)")
    print("=" * 85)
    print(f"  {'Subject':<16} | {'n':>4} | {'Base P':>7} | {'Prop P':>7} | {'Base MAE':>8} | {'Prop MAE':>8} | {'F1 Base':>8} | {'F1 Prop':>8}")
    print(f"  {'-'*16}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    subject_rows = []
    for subj in test_subj_df["subject"].value_counts().index:
        mask = (test_subj_df["subject"] == subj).values
        n = mask.sum()
        if n < 3:
            continue
        yt, yb, yp = y_test_s[mask], y_base_s[mask], y_prop_s[mask]
        try:
            pb, _ = pearsonr(yt, yb)
        except Exception:
            pb = float("nan")
        try:
            pp, _ = pearsonr(yt, yp)
        except Exception:
            pp = float("nan")
        mae_b = mean_absolute_error(yt, yb)
        mae_p = mean_absolute_error(yt, yp)
        f1_b  = f1_score((yt>=50).astype(int), (yb>=50).astype(int), zero_division=0)*100
        f1_p  = f1_score((yt>=50).astype(int), (yp>=50).astype(int), zero_division=0)*100

        print(f"  {subj[:14]:<16} | {n:>4} | {pb:>7.4f} | {pp:>7.4f} | {mae_b:>8.2f} | {mae_p:>8.2f} | {f1_b:>7.1f}% | {f1_p:>7.1f}%")
        subject_rows.append(dict(subject=subj, n=n, pearson_base=pb, pearson_prop=pp,
                                 mae_base=mae_b, mae_prop=mae_p, f1_base=f1_b, f1_prop=f1_p))
    print("=" * 85)
    return subject_rows

# ── per-category breakdown ────────────────────────────────────────────────────

def per_category_breakdown(y_test, y_pred_proposed, y_pred_baseline, adv_start: int):
    """Print normal vs adversarial performance, with adversarial type breakdown."""
    y_test_n = y_test[:adv_start];   y_test_a = y_test[adv_start:]
    y_prop_n = y_pred_proposed[:adv_start]; y_prop_a = y_pred_proposed[adv_start:]
    y_base_n = y_pred_baseline[:adv_start]; y_base_a = y_pred_baseline[adv_start:]

    print("\n" + "=" * 75)
    print("  PER-CATEGORY PERFORMANCE")
    print("=" * 75)

    if len(y_test_n) > 2:
        p_norm_b, _ = pearsonr(y_test_n, y_base_n)
        p_norm_p, _ = pearsonr(y_test_n, y_prop_n)
        print(f"  Normal (n={len(y_test_n)}):")
        print(f"    Baseline  → Pearson: {p_norm_b:.4f}, MAE: {mean_absolute_error(y_test_n, y_base_n):.2f}")
        print(f"    Proposed  → Pearson: {p_norm_p:.4f}, MAE: {mean_absolute_error(y_test_n, y_prop_n):.2f}")

    if len(y_test_a) > 2:
        det_b = np.mean(y_base_a < 20)*100; det_p = np.mean(y_prop_a < 20)*100
        print(f"\n  Adversarial (n={len(y_test_a)}, all human_score=0):")
        print(f"    Baseline → MAE: {mean_absolute_error(y_test_a, y_base_a):.2f}, Detected: {det_b:.1f}%")
        print(f"    Proposed → MAE: {mean_absolute_error(y_test_a, y_prop_a):.2f}, Detected: {det_p:.1f}%")

        if len(y_test_a) >= 50:
            types = [("Karak Reversal", 0, 17), ("Negation", 17, 33), ("Voice Change", 33, 50)]
            print("\n  Adversarial Breakdown by Type:")
            print(f"    {'Type':<20} | {'Base MAE':>8} | {'Prop MAE':>8} | {'Base Det%':>9} | {'Prop Det%':>9}")
            print(f"    {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}")
            for tname, s, e in types:
                yt = y_test_a[s:e]; yb = y_base_a[s:e]; yp = y_prop_a[s:e]
                print(
                    f"    {tname:<20} | {mean_absolute_error(yt,yb):>8.2f} | {mean_absolute_error(yt,yp):>8.2f}"
                    f" | {np.mean(yb<20)*100:>8.1f}% | {np.mean(yp<20)*100:>8.1f}%"
                )
    print("=" * 75)

# ── visualisation ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_kg, y_pred_proposed):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = [
        ("Baseline (LaBSE only)", y_pred_baseline),
        ("LaBSE + KG + GAT-GNN", y_pred_kg),
        ("Proposed (Full Ensemble)", y_pred_proposed),
    ]
    y_true_cls = (np.array(y_test) >= 50).astype(int)
    for ax, (name, y_pred) in zip(axes, models):
        y_pred_cls = (np.array(y_pred) >= 50).astype(int)
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred\nIncorrect", "Pred\nCorrect"],
                    yticklabels=["Actually\nIncorrect", "Actually\nCorrect"])
        ax.set_title(name, fontsize=11)
    plt.suptitle("Confusion Matrices: Correct (≥50) vs Incorrect (<50)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_CONFUSION, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_CONFUSION}")

    print("\nClassification Report (Proposed Full Ensemble):")
    y_pred_cls = (np.array(y_pred_proposed) >= 50).astype(int)
    print(classification_report(y_true_cls, y_pred_cls,
                                target_names=["Incorrect (<50)", "Correct (≥50)"], digits=3))

def plot_error_distribution(y_test, y_pred_baseline, y_pred_kg, y_pred_proposed):
    fig, ax = plt.subplots(figsize=(10, 5))
    for preds, label, color in [
        (y_pred_baseline, f"Baseline (MAE={mean_absolute_error(y_test, y_pred_baseline):.1f})", "red"),
        (y_pred_kg,       f"LaBSE+KG+GAT (MAE={mean_absolute_error(y_test, y_pred_kg):.1f})",   "blue"),
        (y_pred_proposed, f"Full Ensemble (MAE={mean_absolute_error(y_test, y_pred_proposed):.1f})", "green"),
    ]:
        ax.hist(np.array(y_test) - np.array(preds), bins=30, alpha=0.5,
                label=label, color=color, edgecolor="dark"+color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Residual (Human − Predicted)"); ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution: Residuals per Model", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_ERROR_DIST, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_ERROR_DIST}")

def plot_score_distributions(y_test, y_pred_baseline, y_pred_proposed):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y_test, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].axvline(np.mean(y_test), color="red", linestyle="--", label=f"Mean={np.mean(y_test):.1f}")
    axes[0].set_title("Human Score Distribution"); axes[0].set_xlabel("Score"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].hist(y_pred_proposed, bins=20, color="coral", edgecolor="black", alpha=0.8, label="Proposed")
    axes[1].hist(y_pred_baseline, bins=20, color="lightblue", edgecolor="black", alpha=0.5, label="Baseline")
    axes[1].axvline(np.mean(y_pred_proposed), color="red", linestyle="--", label=f"Ensemble Mean={np.mean(y_pred_proposed):.1f}")
    axes[1].set_title("Predicted Score Distribution"); axes[1].set_xlabel("Score"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SCORE_DIST, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_SCORE_DIST}")

def plot_ablation_bars(all_metrics: list):
    names    = [m["name"] for m in all_metrics]
    pearsons = [m["pearson"] for m in all_metrics]
    maes     = [m["mae"]     for m in all_metrics]
    f1s      = [m["cls_f1"] for m in all_metrics]
    r2s      = [m["r2"]     for m in all_metrics]
    colors   = ["#FF6B6B", "#FFA07A", "#87CEEB", "#4ECDC4", "#2ECC71"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(names))

    for ax, vals, title, fmt in [
        (axes[0,0], pearsons, "Pearson (↑)", ".4f"),
        (axes[0,1], maes,     "MAE (↓)",      ".2f"),
        (axes[1,0], f1s,      "F1 % (↑)",     ".1f"),
        (axes[1,1], r2s,      "R² (↑)",        ".4f"),
    ]:
        bars = ax.barh(names, vals, color=colors, edgecolor="black")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(max(bar.get_width(), 0) + max(vals)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    format(v, fmt), va="center", fontsize=9)

    plt.suptitle("Ablation Study: Model Comparison", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_ABLATION, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_ABLATION}")

def plot_scatter_comparison(y_test, y_pred_baseline, y_pred_proposed, adv_start: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, preds, title in [
        (axes[0], y_pred_baseline, "Baseline (LaBSE only)"),
        (axes[1], y_pred_proposed, "Proposed (Full Ensemble)"),
    ]:
        ax.scatter(y_test[:adv_start], preds[:adv_start], alpha=0.5, c="blue", label="Normal", s=30)
        ax.scatter(y_test[adv_start:], preds[adv_start:], alpha=0.7, c="red", marker="x", label="Adversarial", s=50)
        ax.plot([0,100],[0,100],"k--",alpha=0.5, label="Perfect")
        ax.fill_between([0,100],[-15,85],[15,115], alpha=0.1, color="green", label="±15")
        ax.set_xlabel("Human Score"); ax.set_ylabel("Predicted Score")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(-5,105); ax.set_ylim(-5,105)
    plt.suptitle("Predicted vs Human Scores: Normal vs Adversarial", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_SCATTER}")

def plot_per_subject(subject_rows: list):
    if len(subject_rows) < 2:
        print("  Skipping per-subject chart (fewer than 2 subjects).")
        return
    subj_df = pd.DataFrame(subject_rows).sort_values("n", ascending=False)
    labels  = subj_df["subject"].tolist()
    x = np.arange(len(labels)); w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(labels)*2), 5))
    for ax, metric_b, metric_p, ylabel, title in [
        (axes[0], "mae_base",  "mae_prop",  "MAE (↓)",       "MAE per Subject"),
        (axes[1], "f1_base",   "f1_prop",   "F1 Score % (↑)", "F1 per Subject"),
    ]:
        b1 = ax.bar(x-w/2, subj_df[metric_b], w, label="Baseline", color="#FF6B6B", edgecolor="black")
        b2 = ax.bar(x+w/2, subj_df[metric_p], w, label="Proposed", color="#2ECC71", edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Per-Subject: Baseline vs Proposed", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_SUBJECT, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_SUBJECT}")

def plot_incremental_ablation(all_metrics: list):
    names    = [m["name"]    for m in all_metrics]
    pearsons = [m["pearson"] for m in all_metrics]
    maes     = [m["mae"]     for m in all_metrics]
    f1s      = [m["cls_f1"] for m in all_metrics]
    x = np.arange(len(names))
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#428bca", "#5cb85c"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vals, title, fmt, up in [
        (axes[0], pearsons, "Pearson (↑)", ".4f", True),
        (axes[1], maes,     "MAE (↓)",     ".2f", False),
        (axes[2], f1s,      "F1 % (↑)",    ".1f", True),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="black", width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(["B","+E","+G","+K","+X"], fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    format(v, fmt), ha="center", va="bottom", fontsize=9, fontweight="bold")
        for i in range(1, len(vals)):
            delta = vals[i]-vals[i-1]
            col = "#2ECC71" if (delta>0)==up else "#E74C3C"
            ax.annotate(f"{delta:+.3f}" if up else f"{delta:+.2f}",
                        xy=(i, vals[i]), xytext=(i-0.5, max(vals)*0.5),
                        fontsize=7, color=col, ha="center",
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    legend_labels = ["B: Baseline", "+E: Entity", "+G: GAT-GNN", "+K: Karak", "+X: XGBoost+MLP"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    plt.suptitle("Incremental Ablation: Component Contributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_INCREMENTAL, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_INCREMENTAL}")
