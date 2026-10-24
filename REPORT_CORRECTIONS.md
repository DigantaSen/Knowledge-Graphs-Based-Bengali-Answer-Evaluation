# Report corrections — Knowledge-Graph-Based ASAG for Bengali

What has to change in `report.pdf` after the code review and fixes (runs of
2026-10-23). All numbers come from `results/<dataset>/evaluate_20261023_*`
(unseen-answer split) and `results/<dataset>_question/evaluate_*`
(unseen-question split). Each folder's `run_summary.md` has paste-ready tables.

---

## 1. What changed in the system (describe these in §4–§5)

| Area | Before | Now |
|---|---|---|
| POS mapping | Penn tags (`NN`, `VB`); BNLP's BIS tags (`NC`, `VM`, …) matched nothing, so 92% of tokens were `X` and no subject/object was ever found | BIS tagset mapped; case suffixes stripped for lemmas (`রামকে` → `রাম`) |
| Negation | Whitespace tokens, so `না।` and fused `হয়নি` were missed (0/17 adversarial negations) | Punctuation-aware, fused `-নি` forms; compared per aligned sentence pair (17/17) |
| Karak validator | Never fired (0/17); once enabled it fired on 74/191 correct medium answers | Swaps counted only for explicit objects (`-কে`, passive subject), per aligned sentence pair; passive agents (`দ্বারা`/`কর্তৃক`) handled. Karak reversal 15/17, voice change 9/10, false alarms on score-≥90 answers 0 |
| Coreference | Substring replacement corrupted 11–22% of answers (`সেকেন্ডে` → `শব্দেরকেন্ডে`); seeded from the question | Whole-token replacement, seeded from the reference, noun candidates from the POS tagger; applied to reference and student |
| Triple extraction / KG | Subject = first word, verb = last word, objects after the verb only (none in SOV); star graphs; verb and negation stored only as an edge label the GNN never saw | Predicate-centred graph: `subject —agent→ PREDICATE —object/location/time/count→ argument`; predicate node carries the verb, compound-verb noun and negation (`ওঠে না`); possessives/adjectives attach to their noun; proper-noun runs and number+unit merged; passives |
| GNN input | 768-d BanglaBERT node vectors (every word pair had cosine ≥ 0.82); edge weight only | LaBSE node vectors + node-type one-hot + negated flag (775-d); edge features = weight + edge-type one-hot + reverse flag (9-d); reverse edges added |
| Entity matching | BanglaBERT synonym check at 0.75 always matched (any added word counted as a synonym) | LaBSE similarity ≥ 0.85 (measured: synonyms 0.84–0.98, different entities 0.31–0.83); numbers must match exactly |
| ROUGE-L | `rouge_score` tokenizer deleted all Bengali text, so ROUGE-L = 0 for every pair | Bengali tokenizer |
| Wrong-number flag | Compared whole tokens (`২০৬টি` ≠ `২০৬`) | Digit runs only; also a scorer feature (`wrong_num`) |
| Hard caps | Applied to every answer | Single-sentence references only (humans give multi-fact answers partial credit); caps are fixed a-priori rules, not tuned |
| Scorer | 6 features, fixed 0.6/0.4 XGBoost/MLP weights, pickled | 7 features (+ `wrong_num`); XGBoost/MLP weight chosen on out-of-fold training predictions; saved as plain `.npz` |
| Training data | Adversarial error types never seen in training | Synthetic negated / role-swapped / wrong-number answers (score 0) + unchanged copies added to the **training split only** (≤ 30% of real rows); adversarial test sentences excluded |
| Data cleaning | Duplicate (reference, student) pairs with conflicting scores | Merged and scores averaged |
| Split / CV | Random row split: 47% (single) / 73% (medium) of test rows had an identical pair in train; CV warm-started from a GNN trained on the CV validation rows | Grouped split and grouped CV; fresh GNN per fold; scorer trained on out-of-fold GNN similarities |
| Baselines | Raw `cosine × 100` against a trained model | Each similarity linearly calibrated on the training split |
| Statistics | Single run, no intervals; no seeds | Seeded; bootstrap 95% CIs and paired tests against the best baseline |

---

## 2. Section-by-section corrections

### Abstract
- Replace all numbers (see §3 below). Suggested headline: on normal test answers the proposed system reaches Pearson 0.77 / 0.94 / 0.89 (single / medium / long) against 0.63 / 0.92 / 0.79 for the best calibrated baseline, with significant MAE reductions on single and medium. CV Pearson 0.76 / 0.94 / 0.89.
- Remove "Five-fold cross-validation yields a mean Pearson of 0.87 ± 0.02 … confirming generalisation" — that number was inflated by duplicate pairs and the warm-started GNN.
- Remove "negation and Karak reversal detection remain open challenges" — both were bugs. Replace with the remaining real limitations (wrong location, passive paraphrases, templated data; §4 below).

### §1 Contributions
- Item 1: the dataset has **51** adversarial samples (17 Karak, 17 negation, 4 wrong number, 2 wrong location, 1 partial cover, **10** voice change), not 50 / 9.
- Item 4 (Karak validator): describe the explicit-object rule and sentence alignment; "correctly suppressing false positives from free word order" still holds (identical inflected words in another order are exempt).
- Item 5 (overrides): add "single-sentence references only".
- Add: synthetic error augmentation of the training split; grouped evaluation with bootstrap CIs.

### §2.2 Related work — tools
- Keep "neither Stanza nor spaCy provides a production Bengali dependency parser". Add: the only Universal Dependencies Bengali treebank (BRU) has under 1,000 units; Trankit lists Bengali only as trainable (no pretrained pipeline); mREBEL's 18 languages exclude Bengali; IndIE was evaluated on Hindi; BanglaAutoKG (LREC-COLING 2024) builds corpus-level graphs with an LLM and untyped edges. This justifies the rule-based extractor.

### §3 Dataset
- **Table 1 caption is wrong.** Actual score bands:
  | | correct | partially_correct | incorrect |
  |---|---|---|---|
  | Single | 85–100 | 35–80 | 0–40 |
  | Medium | 85–100 | 40–75 | 5–35 |
  | Long | 78–100 | 25–77 | 0–26 |
  The bands overlap (single: incorrect up to 40, partial from 35). Say so.
- Add the data-quality facts:
  - Only 50 distinct questions per dataset (16 for long); one reference per question.
  - Duplicate pairs: 331 (single) and 508 (medium) rows repeat an earlier (reference, student) pair; 193 of 200 repeated groups have different scores. Some verbatim copies of the reference are scored 45–79.
  - Templated student answers: 53/1000 single, **695/1000 medium**, 113/320 long start or end with a recurring phrase that never occurs in a reference (e.g. `এই প্রশ্নের উত্তরে`, `আরও তথ্য প্রয়োজন।`, `আজ রবিবার।`). Templates largely determine the score.
- New sizes after deduplication and split:
  | Dataset | Raw rows | After dedup | Train | + synthetic | Test normal | Test adversarial |
  |---|---|---|---|---|---|---|
  | Single | 1000 | 669 | 568 | 111 | 101 | 51 |
  | Medium | 1000 | 492 | 418 | 125 | 74 | 51 |
  | Long | 320 | 291 | 247 | 74 | 44 | 51 |

### §4 Methodology
- §4.1: coreference is applied to both answers and seeded from the reference (not the question).
- §4.2: LaBSE is used for sentence similarity, word similarity **and node embeddings**; BanglaBERT only for its baseline (with the official `csebuetnlp` normaliser).
- §4.3: replace the description with the predicate-centred extraction (table in §1).
- §4.4: node features 775-d (LaBSE 768 + 6 node types + negated flag); edge features 9-d; weights: agent/object 1.0, location/time/count 0.8, possessive 0.7, quality 0.6. Remove "agent/object 1.0, modifier 0.5, others 0.75".
- §4.5: GAT layers use edge features of dimension 9; graphs include reverse edges.
- §4.6: describe `nsubj`, `-কে` object, bare object, passive agent / patient; swap penalty 0.5 per explicit swap within aligned sentences; negation penalty uses aligned sentence pairs.
- §4.7: the NER model actually used is **`Davlan/bert-base-multilingual-cased-ner-hrl`**, not `sagorsarker/bangla-bert-base-ner`; synonym threshold 0.85 on LaBSE.
- §4.8: caps apply only to single-sentence references; they are fixed rules, not tuned.
- §4.9: **seven** base features + ten engineered features; the hyperparameters in §4.9 (200 trees, depth 4, lr 0.1, MLP 64-32-16) do not match the code. Actual: XGBoost 300 trees, depth 5, lr 0.08, subsample 0.8, colsample 0.8; MLP 128-64-32, ReLU, Adam lr 1e-3, early stopping. Ensemble weight chosen on out-of-fold training predictions: XGBoost/MLP = 0.9/0.1 (single), 0.5/0.5 (medium), 1.0/0.0 (long).

### §5 Experimental setup
- GNN: lr **0.005** (report says 1e-3), weight decay 1e-4, cosine schedule, 5 epochs, batch 32, dropout 0.2, seed 42. CPU, not a T4 (the re-run took 25 min for all three datasets).
- Split: 85/15 grouped by identical (reference, student) pair ("unseen answers"); a second setting groups by question ("unseen questions"). Adversarial rows test-only.
- Baselines are linearly calibrated on the training split (the old raw `100 × cos` made LaBSE look like MAE 50.5).
- Metrics: F1 is the **binary F1 of the PASS class** (score ≥ 50), not macro-F1; macro-F1 is now reported too. "±10%" is ±10 **points**.
- Significance: paired bootstrap (2,000 resamples) of the proposed system vs the best baseline (lowest MAE) on the same rows.

### Figure 1
- The image is a predicted-vs-true scatter plot, not the architecture diagram. Replace it with the architecture figure (or recaption it).

---

## 3. Replacement results

### Table 2 — single-sentence, full test set (n = 152, incl. 51 adversarial)
| Model | Pearson | MAE | ±10 pts | F1 | Macro-F1 |
|---|---|---|---|---|---|
| LaBSE cosine (calibrated) | 0.306 | 30.26 | 20.4% | 55.0 | 65.0 |
| ROUGE-L (calibrated) | 0.514 | 26.14 | 19.1% | 54.9 | 67.8 |
| LaBSE + ROUGE-L | 0.436 | 28.35 | 21.1% | 61.7 | 72.3 |
| BanglaBERT cosine | 0.241 | 32.00 | 14.5% | 48.4 | 55.5 |
| GNN only | 0.444 | 27.05 | 27.0% | 53.9 | 57.0 |
| **Proposed** | **0.779** | **14.29** | **54.6%** | **72.7** | **80.8** |

### Table 3 — cross-dataset, full test set
| Model | Single P / MAE / F1 | Medium P / MAE / F1 | Long P / MAE / F1 |
|---|---|---|---|
| LaBSE cosine | 0.306 / 30.3 / 55.0 | 0.028 / 32.7 / 50.8 | 0.108 / 35.6 / 37.4 |
| ROUGE-L | 0.514 / 26.1 / 54.9 | 0.279 / 27.7 / 54.7 | 0.235 / 31.8 / 46.2 |
| BanglaBERT | 0.241 / 32.0 / 48.4 | 0.225 / 30.1 / 54.2 | 0.296 / 32.7 / 56.7 |
| GNN only | 0.444 / 27.1 / 53.9 | 0.308 / 28.5 / 63.7 | 0.347 / 32.2 / 48.8 |
| **Proposed** | **0.779 / 14.3 / 72.7** | **0.852 / 9.5 / 84.3** | **0.782 / 13.4 / 75.0** |

### New table — normal test rows with 95% CIs (lead with this)
| Dataset (n) | Best baseline | Baseline P / MAE | Proposed Pearson [95% CI] | Proposed MAE [95% CI] | ΔPearson (p) | ΔMAE (p) |
|---|---|---|---|---|---|---|
| Single (101) | LaBSE + ROUGE-L | 0.626 / 21.21 | 0.772 [0.673, 0.848] | 15.15 [12.45, 18.12] | +0.146 (0.003) | −6.06 (<0.001) |
| Medium (74) | ROUGE-L | 0.922 / 11.52 | 0.942 [0.918, 0.962] | 7.84 [6.67, 9.02] | +0.020 (0.05) | −3.68 (0.001) |
| Long (44) | LaBSE + ROUGE-L | 0.793 / 13.46 | 0.895 [0.812, 0.950] | 9.08 [6.39, 11.93] | +0.101 (0.025) | −4.37 (0.059) |

Non-templated normal rows: single (n = 92) 0.720 vs 0.511, ΔMAE −6.47 (p = 0.001);
medium (n = 18) 0.841 vs 0.651 (p = 0.056), ΔMAE −3.84 (p = 0.18); long (n = 29)
0.868 vs 0.782 (p = 0.27), ΔMAE −1.46 (p = 0.57). The medium/long subsets are
too small for firm conclusions.

### New table — unseen questions (normal test rows)
| Dataset (n) | Best baseline P | Proposed P | ΔMAE (p) |
|---|---|---|---|
| Single (100) | 0.607 (LaBSE) | 0.780 | −6.00 (<0.001) |
| Medium (82) | 0.896 (ROUGE-L) | 0.917 | −4.31 (0.003) |
| Long (54) | 0.863 (LaBSE + ROUGE-L) | 0.887 | −4.70 (0.006) |

### Table 4 — 5-fold grouped CV (normal rows)
| Dataset | Pearson | MAE | F1 |
|---|---|---|---|
| Single | 0.763 ± 0.023 | 15.61 ± 0.86 | 78.4 ± 2.7 |
| Medium | 0.938 ± 0.013 | 7.78 ± 0.73 | 91.8 ± 1.5 |
| Long | 0.890 ± 0.042 | 9.17 ± 1.48 | 82.7 ± 3.4 |

CV and held-out test now agree; the old "CV ≫ test gap" explanation in §6.2 must go.

### Table 5 — feature importances
Top XGBoost features: `max_pen` (= max(Karak penalty, entity mismatch)) dominates
(0.59 single, 0.62 medium); on long the Karak `penalty` itself (0.50). `wrong_num`
is second on medium (0.13). `graph_sim` is no longer a top feature.

### Table 6 — adversarial breakdown (mean predicted score; true score 0, partial cover 50)
| Category (n) | Single | Medium | Long | Report (single) |
|---|---|---|---|---|
| Karak reversal (17) | 16.0 | 16.5 | 14.1 | 54.8 |
| Negation (17) | 1.7 | 0.1 | 6.2 | 45.2 |
| Wrong number (4) | 9.1 | 1.5 | 3.3 | 30.8 |
| Wrong location (2) | 27.9 | 63.3 | 77.6 | 35.0 |
| Partial cover (1, true 50) | 14.7 | 43.4 | 74.7 | 50.2 |
| Voice change (10) | 21.4 | 18.7 | 33.6 | 47.0 |
| Overrides fired | 0 | 0 | 1 | — |

The old Table 6 was also shifted by one row (the code assumed 50 adversarial rows).

---

## 4. Claims to remove or rewrite (§6–§7)

- "The Karak validator does detect role swaps in many cases" / Case 2 "the validator fires": untrue for the old code (it never fired). True now.
- "Negation remains largely undetected … LaBSE similarity dominates": the cause was the tokenisation bug. Now negated answers score ≈ 0–6.
- "graph_sim ranks second, validating our choice to invest in KG construction": not supported by the new runs; GNN-only is the weakest signal on single and is not a top feature. Describe the KG's contribution through the Karak/role features and the predicate-centred graph instead.
- "NER-backed soft entity matching prevents penalising paraphrases": with BanglaBERT the synonym check accepted everything; describe the LaBSE version.
- "The substantial gap between CV Pearson and held-out test Pearson is attributable to the adversarial samples": the gap came from leakage; it is gone.
- ROUGE-L "—" Pearson and F1 0.0 in Tables 2/3 were the tokenizer bug.

## 5. Limitations to state

- **Synthetic, templated data.** 695/1000 medium answers are templated; duplicate pairs carry conflicting scores; label bands overlap. Medium results partly measure template recognition (calibrated ROUGE-L alone reaches 0.92 on normal rows).
- **Adversarial set was seen during development** and is small (51, two wrong-location items). A clean estimate needs a new hand-written set from someone who has not seen the rules.
- **Synthetic training errors** use the same perturbation types as the adversarial test (negation, swap, number), so adversarial gains partly reflect matched training data. Wrong location (not synthesised) is still scored 28–78.
- **Correct passive paraphrases also score low** (e.g. `রামের দ্বারা রাবণ নিহত হয়েছেন` ≈ 39): the voice-change gain is partly a surface-difference effect.
- **Rule-based extraction** depends on BNLP POS tags and case endings; tagger errors (e.g. `সূর্য` tagged as an adjective) still pick the wrong subject. No Bengali dependency parser is available.
- **Small test sets** (44–101 normal rows); medium and long differences on the non-templated subset are not significant.

---

## 6. Update — answer-slot features and clean synthetic data

**This section supersedes the numbers in §3 for the unseen-answer split.**
Two additions were made after §3 was written; describe both in §3–§5.

### 6.1 Method additions

- **Answer-slot features** (`src/features/answer_slot.py`). Words that also
  occur in the question are removed from both answers; the rest is the
  answer content. Three features are added to the scorer (now 10 base + 12
  engineered features):
  `ans_recall` (share of the reference's answer content present),
  `ans_precision` (share of the student's answer content supported by the
  reference) and `key_match` (whether the key answer item is present — the
  number for কবে/কত, the place/direction for কোথায়, the item named by
  "কোন X", else the first proper noun, skipping titles such as সম্রাট).
  Motivation: shared question words inflated the similarity of wrong answers
  and added nothing for correct brief ones.
- **Clean synthetic dataset** (`datasets/clean_synthetic/`): 514 rows, 54 new
  questions (30 / 16 / 8), written without templates and scored by a fixed
  rubric (not human grades) — see that folder's README for the rubric.
  `data/dataset_<tier>_plus_clean.csv` combines it with the original data.

### 6.2 Replacement results — original datasets, unseen-answer split

| Dataset | Full test P / MAE / F1 | Normal rows P [95% CI] | Normal MAE [95% CI] | vs best baseline (p) | CV Pearson |
|---|---|---|---|---|---|
| Single | 0.830 / 12.1 / 80.4 | 0.829 [0.743, 0.895] | 12.71 [10.30, 15.35] | ΔP +0.203, ΔMAE −8.49 (both p < 0.001) | 0.801 ± 0.030 |
| Medium | 0.855 / 9.5 / 86.0 | 0.951 [0.930, 0.967] | 6.79 [5.58, 8.04] | ΔP +0.029 (p = 0.013), ΔMAE −4.72 (p < 0.001) | 0.944 ± 0.012 |
| Long | 0.788 / 12.2 / 72.3 | 0.907 [0.822, 0.959] | 8.42 [6.06, 11.15] | ΔP +0.114 (p = 0.019), ΔMAE −5.04 (p = 0.02) | 0.916 ± 0.036 |

Baselines are unchanged from §3 (Table 3). All three comparisons with the
best baseline are now significant on both Pearson and MAE.

Adversarial breakdown (mean predicted score, true 0; partial cover 50):

| Category | Single | Medium | Long |
|---|---|---|---|
| Karak reversal | 14.9 | 18.4 | 15.4 |
| Negation | 0.7 | 0.3 | 5.8 |
| Wrong number | 4.5 | 6.4 | 4.2 |
| Wrong location | 32.4 | 62.8 | 45.4 |
| Partial cover (true 50) | 34.2 | 54.2 | 52.3 |
| Voice change | 19.1 | 21.2 | 31.8 |

### 6.3 New experiment — does the system generalise to non-templated answers?

Trained on original + clean data, tested on **unseen questions**; the
"clean" rows are clean-dataset answers to questions the model never saw.

| Tier | Clean test rows | Best baseline P / MAE | Original data only | + clean data | + clean + answer-slot features |
|---|---|---|---|---|---|
| Single | 66 | 0.41 / 34.2 | 0.57 / 27.4 | 0.65 / 25.0 | **0.74 / 20.5** |
| Medium | 24 | 0.74 / 17.4 | 0.20 / 28.6 | 0.78 / 15.2 | **0.86 / 12.4** |
| Long | 14 | 0.91 / 15.4 | 0.60 / 26.3 | 0.87 / 13.9 | **0.95 / 9.0** |

CV Pearson on original + clean (unseen questions): 0.749 ± 0.024 (single),
0.920 ± 0.020 (medium), 0.926 ± 0.026 (long).

Key finding for the discussion: trained on the original data alone, the
model does *worse than a simple baseline* on natural, non-templated medium
answers (0.20 vs 0.74) — evidence that it had partly learned the templates.

Single-sentence clean test rows by answer type (mean predicted vs rubric):

| Type | Rubric | Original + clean | + answer-slot features |
|---|---|---|---|
| Paraphrase | 95 | 54 | 67 |
| Brief ("শাহজাহান।") | 88 | 45 | 62 |
| Correct passive | 92 | 58 | 63 |
| Wrong entity | 10 | 42 | 32 |
| Wrong detail | 8 | 43 | 38 |

Each type has only 6 test rows. Correctly worded but differently phrased
answers are still under-scored and wrong-entity answers over-scored; state
this as a limitation.

### 6.4 Tried and rejected (worth one sentence in the report)

- Up-weighting the clean rows ×3 in training: clean-row Pearson fell from 0.76 to 0.72 and the original rows got worse.
- Rule-generating brief / wrong-entity training answers from the original references: about two-thirds came out mislabelled (BNLP tags many common words as proper nouns), so it was dropped.

### 6.5 Unseen questions (regenerated with the current features)

Supersedes the §3 "unseen questions" table.

| Dataset (n) | Best baseline P / MAE | Proposed P / MAE | ΔPearson (p) | ΔMAE (p) |
|---|---|---|---|---|
| Single (100) | 0.607 / 21.49 (LaBSE) | 0.829 / 13.76 | +0.222 (<0.001) | −7.73 (<0.001) |
| Medium (82) | 0.896 / 13.47 (ROUGE-L) | 0.924 / 8.72 | +0.028 (0.028) | −4.75 (<0.001) |
| Long (54) | 0.863 / 14.34 (LaBSE + ROUGE-L) | 0.916 / 8.57 | +0.053 (0.16) | −5.77 (<0.001) |

### 6.6 Revised report

`report/report.tex` (compiled to `report/report.pdf` with XeLaTeX) is a
rebuilt version of the report with all of the above applied, a component
ablation (Appendix A) and regenerated case studies.
