# Clean synthetic dataset

A second, cleaner source of training and test data, built because the
original datasets are heavily templated (695 / 1000 medium student answers
share a template phrase), repeat (reference, student) pairs with conflicting
scores, and use overlapping label bands.

**Provenance.** Every question, reference and student answer here was written
with an LLM (Claude) and checked against well-established textbook facts.
Scores come from the fixed rubric below — **they are not human grades**. The
column is still called `human_score` so the pipeline can read the files; the
`source` column (`clean_synthetic`) and `answer_type` mark these rows. Report
them as rubric-scored synthetic data. A native Bengali speaker should review
the text before it is used for anything beyond this project.

## Contents

| Tier | Questions | Rows | Answers per question | Subjects |
|---|---|---|---|---|
| Single sentence | 30 | 330 | 11 | ইতিহাস, বিজ্ঞান, ভূগোল, সাহিত্য, সাধারণ (6 each) |
| Medium (3 key facts) | 16 | 128 | 8 | বিজ্ঞান, ভূগোল, ইতিহাস, সাহিত্য, সাধারণ |
| Long (5 key facts) | 8 | 56 | 7 | বিজ্ঞান, ইতিহাস, ভূগোল, সাহিত্য, রাষ্ট্রবিজ্ঞান |

No question appears in the original datasets, no reference sentence is one
of the hand-written adversarial test samples, no answer uses a template
phrase of the original data, and there are no duplicate answers.

## Rubric

Labels use non-overlapping bands: **correct ≥ 85**, **partially correct 35–84**, **incorrect < 35**.

**Single sentence** — one score per answer type:

| Type | Score | Meaning | Example (তাজমহল কে নির্মাণ করেছিলেন?) |
|---|---|---|---|
| `copy` | 100 | the reference | সম্রাট শাহজাহান … তাজমহল নির্মাণ করেছিলেন। |
| `paraphrase` | 95 | same facts, other words / order | মুঘল সম্রাট শাহজাহান … তাজমহল তৈরি করেন। |
| `alt_construction` | 92 | same facts, other construction (often passive) | তাজমহল সম্রাট শাহজাহান কর্তৃক … নির্মিত হয়েছিল। |
| `extra_info` | 92 | all facts plus correct extra detail | … আগ্রায় যমুনা নদীর তীরে … |
| `brief` | 88 | only the key information | শাহজাহান। |
| `partial` | 55 | vague / incomplete | একজন মুঘল সম্রাট … |
| `partial_error` | 40 | key fact right, a secondary detail wrong | … তাঁর মায়ের স্মৃতিতে … |
| `wrong_entity` | 10 | wrong key entity | সম্রাট আকবর … |
| `wrong_detail` | 8 | wrong number / date / direction / place | (e.g. দক্ষিণ instead of উত্তর) |
| `swap` | 0 | agent and patient (or cause and effect) reversed | মমতাজ মহল … শাহজাহানের স্মৃতিতে … |
| `negation` | 0 | the reference negated | … নির্মাণ করেননি। |
| `off_topic` / `dont_know` | 0 | unrelated / no answer | |

**Medium and long** — each of the k key facts (k = 3 medium, 5 long, listed
in the JSON) is worth 95 / k points; a fact stated wrongly (wrong value,
negated, roles swapped) earns nothing and costs a further 10:

| Type | Medium (k = 3) | Long (k = 5) |
|---|---|---|
| `full` / `full_extra` | 95 | 95 |
| `missing1` (one fact left out) | 63 | 76 |
| `missing2` | 32 | 57 |
| `missing3` | — | 38 |
| `missing4` | — | 19 |
| `wrong_value` / `negated` / `swap` / `contradicted` (one fact wrong) | 53 | 66 |
| `off_topic` | 0 | 0 |

## Building

```bash
python datasets/clean_synthetic/build.py
```

Reads `single_part1.json`, `single_part2.json`, `medium.json`, `long.json`,
validates them, and writes to `data/`:

- `dataset_clean_<tier>.csv` — the clean rows only
- `dataset_<tier>_plus_clean.csv` — original rows (`source = original`) + clean rows

The `_plus_clean` files run through the normal pipeline; results then include
`metrics_table_original.csv` and `metrics_table_clean.csv`.

## Limitations

- Rubric scores are idealised: real graders disagree and give in-between scores.
- One author wrote every answer, so the variety of errors is limited to the listed types.
- Facts were chosen to be uncontroversial textbook facts but have not been reviewed by a subject expert.
- Small: 54 questions in total; the long tier has only 8.

## Native-speaker review

`review_sheet.csv` lists all 514 rows (opens in Excel with Bengali intact).
For each row, fill in `natural_bengali (y/n)`, `fact_and_type_correct (y/n)`,
and, where needed, `corrected_student_answer` / `comment`. Then:

```bash
python datasets/clean_synthetic/apply_review.py            # applies the sheet, rebuilds the CSVs
python run_all_datasets.py --datasets data/dataset_single_sentence_plus_clean.csv \
       data/dataset_medium_sentences_plus_clean.csv data/dataset_long_sentences_plus_clean.csv
```

A correction replaces the answer text; a row marked `n` with no correction is
removed. The author's own check (facts verified against standard textbook
knowledge; dates such as 1905/1911 বঙ্গভঙ্গ, 1914 প্রথম বিশ্বযুদ্ধ, 1526 পানিপথ,
1907 চর্যাপদ, 2013 রানা প্লাজা) found no factual errors, but naturalness of the
phrasing has not been reviewed by a native speaker.
