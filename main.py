"""
main.py - Main training and evaluation entry point for the Bengali Answer Evaluation System.

Usage:
    python main.py                   # train + evaluate
    python main.py --skip-cv         # skip 5-fold CV (faster)
    python main.py --data path/to/dataset.csv
"""

import argparse
import os
import warnings
import numpy as np
import torch
import stanza
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from transformers import pipeline as hf_pipeline

warnings.filterwarnings("ignore")

# Project modules
from src.config import (
    AUGMENTED_CSV, GNN_MODEL_PATH, SCORER_MODEL_PATH,
    NUM_ADVERSARIAL, TEST_SIZE, RANDOM_STATE, GNN_EPOCHS,
    PLOT_CORRELATION,
)
from src.data_processing import get_augmented_dataset, setup_bengali_font
from src.preprocessing import TextPreprocessor
from src.coreference import BengaliCoreferenceResolver
from src.dependency_parser import BengaliDependencyParser
from src.triple_extractor import TripleExtractor
from src.knowledge_graph import KnowledgeGraphConstructor
from src.embeddings import BanglaBERTEmbedder
from src.gnn_model import SiameseGNN
from src.karak_validator import KarakValidator
from src.scorer import AnswerScorer
from src.pipeline import (
    create_graphs_and_features, train_gnn,
    get_gnn_similarities, plot_correlation,
)
from src.evaluation import (
    compute_full_metrics, print_all_metric_tables,
    run_cross_validation,
    per_subject_breakdown, per_category_breakdown,
    plot_confusion_matrices, plot_error_distribution,
    plot_score_distributions, plot_ablation_bars,
    plot_scatter_comparison, plot_per_subject,
    plot_incremental_ablation,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Bengali Answer Evaluation System")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the original CSV dataset (optional).")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip 5-fold cross-validation to save time.")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 55)
    print("  Bengali Answer Evaluation System")
    print("  GAT + LaBSE + XGBoost + Coref + Soft NER")
    print("=" * 55)

    # 0. Bengali font
    bengali_fp = setup_bengali_font()

    # 1. Stanza pipeline
    print("\n[1/8] Initialising Stanza NLP pipeline...")
    nlp = None
    try:
        stanza.download("bn", verbose=False)
        nlp = stanza.Pipeline("bn", use_gpu=False, verbose=False)
        print("  Stanza Bengali pipeline loaded.")
    except Exception as e:
        print(f"  Stanza failed: {e}. Using SOV fallback parser.")

    # 2. NER pipeline
    print("\n[2/8] Loading NER pipeline...")
    ner_pipe = None
    try:
        ner_pipe = hf_pipeline(
            "ner", model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple", device=-1,
        )
        print("  NER pipeline loaded.")
    except Exception as e:
        print(f"  ⚠ NER pipeline failed: {e}.")

    # 3. Initialise components
    print("\n[3/8] Initialising components...")
    text_processor  = TextPreprocessor(nlp_pipeline=nlp)
    dep_parser      = BengaliDependencyParser(nlp_pipeline=nlp)
    coref_resolver  = BengaliCoreferenceResolver(nlp_pipeline=nlp)
    triple_extractor = TripleExtractor()
    kg_constructor  = KnowledgeGraphConstructor()
    embedder        = BanglaBERTEmbedder()
    validator       = KarakValidator()

    # 4. Load / augment dataset
    print("\n[4/8] Loading dataset...")
    df = get_augmented_dataset()

    # Split: the last NUM_ADVERSARIAL rows go to test; the rest are split normally.
    df_adv    = df.tail(NUM_ADVERSARIAL).copy()
    df_normal = df.iloc[:-NUM_ADVERSARIAL].copy()

    train_df, test_normal_df = train_test_split(
        df_normal, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    test_df  = df_adv.copy() if len(test_normal_df) == 0 else \
               __import__("pandas").concat([test_normal_df, df_adv]).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    adv_start = len(test_normal_df)

    print(f"  Train: {len(train_df)} | Test: {len(test_df)} ({NUM_ADVERSARIAL} adversarial)")

    # 5. Feature extraction — training set
    print("\n[5/8] Processing TRAINING data...")
    bsims_tr, rpyg_tr, spyg_tr, pen_tr, emm_tr = create_graphs_and_features(
        train_df, text_processor, dep_parser, triple_extractor,
        kg_constructor, embedder, validator,
        coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipe,
    )
    y_train = train_df["human_score"].values.astype(float)

    # 6. Train GAT-GNN
    print("\n[6/8] Training Siamese GAT-GNN...")
    gnn = SiameseGNN()
    gnn = train_gnn(gnn, rpyg_tr, spyg_tr, y_train, epochs=GNN_EPOCHS)
    gsims_tr = get_gnn_similarities(gnn, rpyg_tr, spyg_tr)

    # 7. Train ensemble scorer
    print("\n[7/8] Training XGBoost + MLP Ensemble Scorer...")
    X_train = np.column_stack((bsims_tr, gsims_tr, pen_tr, emm_tr))
    scorer  = AnswerScorer()
    scorer.train(X_train, y_train)

    # 8. Feature extraction — test set
    print("\n[8/8] Processing TEST data and evaluating...")
    bsims_te, rpyg_te, spyg_te, pen_te, emm_te = create_graphs_and_features(
        test_df, text_processor, dep_parser, triple_extractor,
        kg_constructor, embedder, validator,
        coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipe,
    )
    y_test = test_df["human_score"].values.astype(float)
    gsims_te = get_gnn_similarities(gnn, rpyg_te, spyg_te)

    # ── ablation models ───────────────────────────────────────────────────────
    # Baseline: LaBSE only
    base_lr = LinearRegression().fit(np.array(bsims_tr).reshape(-1, 1), y_train)
    y_pred_baseline = base_lr.predict(np.array(bsims_te).reshape(-1, 1)).clip(0, 100)

    # +Entity
    em_lr = LinearRegression().fit(np.column_stack((bsims_tr, emm_tr)), y_train)
    y_pred_em = em_lr.predict(np.column_stack((bsims_te, emm_te))).clip(0, 100)

    # +GAT-GNN
    kg_lr = LinearRegression().fit(np.column_stack((bsims_tr, gsims_tr, emm_tr)), y_train)
    y_pred_kg = kg_lr.predict(np.column_stack((bsims_te, gsims_te, emm_te))).clip(0, 100)

    # +Karak
    noe_lr = LinearRegression().fit(np.column_stack((bsims_tr, gsims_tr, pen_tr)), y_train)
    y_pred_noe = noe_lr.predict(np.column_stack((bsims_te, gsims_te, pen_te))).clip(0, 100)

    # Proposed: full ensemble
    X_test_full = np.column_stack((bsims_te, gsims_te, pen_te, emm_te))
    y_pred_proposed = scorer.predict_batch(X_test_full)

    # ── metrics ───────────────────────────────────────────────────────────────
    all_metrics = [
        compute_full_metrics(y_test, y_pred_baseline, "Baseline (LaBSE only)"),
        compute_full_metrics(y_test, y_pred_em,       "LaBSE + Soft Entity Match"),
        compute_full_metrics(y_test, y_pred_kg,       "LaBSE + KG + GAT-GNN"),
        compute_full_metrics(y_test, y_pred_noe,      "LaBSE + KG + GAT + Karak"),
        compute_full_metrics(y_test, y_pred_proposed, "Proposed (Full Ensemble)"),
    ]

    print_all_metric_tables(all_metrics)

    per_subject_breakdown(test_df, y_test, y_pred_baseline, y_pred_proposed, adv_start)
    per_category_breakdown(y_test, y_pred_proposed, y_pred_baseline, adv_start)

    # ── cross-validation ──────────────────────────────────────────────────────
    if not args.skip_cv:
        run_cross_validation(
            df_normal, text_processor, dep_parser, triple_extractor,
            kg_constructor, embedder, validator,
            coref_resolver=coref_resolver, ner_pipeline_obj=ner_pipe,
        )

    # ── visualisations ────────────────────────────────────────────────────────
    plot_correlation(
        y_test,
        {
            "Baseline (LaBSE only)": y_pred_baseline,
            "LaBSE + KG + GAT-GNN": y_pred_kg,
            "Proposed (Full Ensemble)": y_pred_proposed,
        },
        "AI vs Human Score Correlation",
        PLOT_CORRELATION,
    )
    plot_confusion_matrices(y_test, y_pred_baseline, y_pred_kg, y_pred_proposed)
    plot_error_distribution(y_test, y_pred_baseline, y_pred_kg, y_pred_proposed)
    plot_score_distributions(y_test, y_pred_baseline, y_pred_proposed)
    plot_ablation_bars(all_metrics)
    plot_scatter_comparison(y_test, y_pred_baseline, y_pred_proposed, adv_start)
    plot_incremental_ablation(all_metrics)

    # Per-subject chart
    subject_rows = per_subject_breakdown(test_df, y_test, y_pred_baseline, y_pred_proposed, adv_start)
    plot_per_subject(subject_rows)

    # ── explainability demo ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EXPLAINABILITY DEMO (random adversarial sample)")
    print("=" * 50)
    adv_idx = np.random.randint(adv_start, len(test_df))
    row = test_df.iloc[adv_idx]
    print(f"  Question:    {row['question']}")
    print(f"  Reference:   {row['reference_answer']}")
    print(f"  Student:     {row['student_answer']}")
    print(f"  Human Score: {row['human_score']}")

    ref_p = dep_parser.parse(text_processor.normalize(row["reference_answer"]))
    stu_p = dep_parser.parse(text_processor.normalize(row["student_answer"]))
    _, expls = validator.validate(ref_p, stu_p)

    print(
        scorer.generate_explanation(
            y_pred_proposed[adv_idx],
            bsims_te[adv_idx],
            gsims_te[adv_idx],
            pen_te[adv_idx],
            expls,
            emm_te[adv_idx],
        )
    )

    # ── save models ───────────────────────────────────────────────────────────
    print("Saving trained models...")
    torch.save(gnn.state_dict(), GNN_MODEL_PATH)
    scorer.save_model(SCORER_MODEL_PATH)

    print("\n✅ Enhanced Bengali Answer Evaluation System — training complete!")
    print(f"   Models saved: {GNN_MODEL_PATH}, {SCORER_MODEL_PATH}")
    print("   Run  python app.py  to launch the Gradio web interface.")

if __name__ == "__main__":
    main()
