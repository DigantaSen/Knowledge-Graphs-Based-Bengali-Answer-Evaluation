"""
build.py
========
Build the clean synthetic datasets from the JSON sources in this folder.

    python datasets/clean_synthetic/build.py

Writes, into data/:
    dataset_clean_<tier>.csv          clean synthetic rows only
    dataset_<tier>_plus_clean.csv     original rows + clean synthetic rows

Every row is written by hand (LLM-assisted) with a declared answer type, and
its score comes from the fixed rubric below, NOT from human graders.  The
'human_score' column keeps its name only for pipeline compatibility; the
'source' column ('original' / 'clean_synthetic') and 'answer_type' say where
each row came from.

The build refuses to write anything if a check fails (unknown answer type,
duplicate answer, question already in the original data, reference sentence
taken from the adversarial test samples, template phrase from the original
data, missing sentence terminator).
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from src.data.dataset import _ADVERSARIAL_SAMPLES, template_phrases  # noqa: E402

# ─── Rubric ───────────────────────────────────────────────────────────────────
# Single-sentence answers: one score per answer type.
SINGLE_SCORES = {
    'copy':             100,   # the reference itself
    'paraphrase':        95,   # same facts, different words / order
    'alt_construction':  92,   # same facts, different construction (passive, nominal)
    'extra_info':        92,   # all facts plus correct extra detail
    'brief':             88,   # just the key information ('শাহজাহান।')
    'partial':           55,   # vague / incomplete key information
    'partial_error':     40,   # key fact right, a secondary detail wrong
    'wrong_entity':      10,   # wrong person / thing / place as the key fact
    'wrong_detail':       8,   # wrong number / date / direction / attribute
    'swap':               0,   # agent and patient (or cause and effect) reversed
    'negation':           0,   # the reference negated
    'off_topic':          0,   # unrelated statement
    'dont_know':          0,   # no answer
}

# Multi-fact answers: every reference fact is worth 95 / k points; a fact that
# is stated wrongly (wrong value, negated, roles swapped) earns nothing and
# costs 10 more.  type -> (correct facts, contradicted facts), k inferred.
def _fact_types(k: int) -> dict:
    types = {'full': (k, 0), 'full_extra': (k, 0), 'off_topic': (0, 0)}
    for m in range(1, k):
        types[f'missing{m}'] = (k - m, 0)
    for t in ('wrong_value', 'negated', 'swap', 'contradicted'):
        types[t] = (k - 1, 1)
    return types


def fact_score(correct: int, contradicted: int, k: int) -> int:
    return max(0, round(95 * correct / k - 10 * contradicted))


def label_for(score: float) -> str:
    # Non-overlapping bands (the original data's bands overlap)
    if score >= 85:
        return 'correct'
    if score >= 35:
        return 'partially_correct'
    return 'incorrect'


TIERS = {
    'single_sentence':  {'files': ['single_part1.json', 'single_part2.json'], 'k': None, 'id0': 100001},
    'medium_sentences': {'files': ['medium.json'], 'k': 3, 'id0': 200001},
    'long_sentences':   {'files': ['long.json'], 'k': 5, 'id0': 300001},
}


def _norm(text: str) -> str:
    return ' '.join(str(text).replace('।', ' ').split())


def build_tier(tier: str, cfg: dict, original: pd.DataFrame) -> pd.DataFrame:
    items = []
    for f in cfg['files']:
        with open(os.path.join(HERE, f), encoding='utf-8') as fh:
            items += json.load(fh)

    k = cfg['k']
    fact_types = _fact_types(k) if k else {}
    orig_questions = set(original['question'].map(_norm))
    adv_refs = {_norm(a['reference_answer']) for a in _ADVERSARIAL_SAMPLES}
    phrases = template_phrases(original)

    rows, seen, errors = [], set(), []
    for item in items:
        q, ref = item['question'], item['reference']
        if _norm(q) in orig_questions:
            errors.append(f'question already in original data: {q}')
        if _norm(ref) in adv_refs:
            errors.append(f'reference is an adversarial test sentence: {ref}')
        if k and len(item.get('facts', [])) != k:
            errors.append(f'{q}: expected {k} facts, got {len(item.get("facts", []))}')
        for atype, text in item['answers']:
            if k:
                if atype not in fact_types:
                    errors.append(f'{q}: unknown answer type {atype!r}')
                    continue
                score = fact_score(*fact_types[atype], k)
            else:
                if atype not in SINGLE_SCORES:
                    errors.append(f'{q}: unknown answer type {atype!r}')
                    continue
                score = SINGLE_SCORES[atype]
            if (q, text) in seen:
                errors.append(f'{q}: duplicate answer {text}')
            seen.add((q, text))
            if not text.strip().endswith(('।', '?', '!')):
                errors.append(f'{q}: answer lacks a sentence terminator: {text}')
            if any(text.startswith(p) or text.endswith(p) for p in phrases):
                errors.append(f'{q}: answer uses an original template phrase: {text}')
            rows.append({'question': q, 'reference_answer': ref, 'student_answer': text,
                         'label': label_for(score), 'human_score': score,
                         'subject': item['subject'], 'source': 'clean_synthetic',
                         'answer_type': atype})

    if errors:
        raise SystemExit(f'[{tier}] {len(errors)} problem(s):\n  ' + '\n  '.join(errors))

    df = pd.DataFrame(rows)
    df.insert(0, 'id', range(cfg['id0'], cfg['id0'] + len(df)))
    return df


def main():
    data_dir = os.path.join(ROOT, 'data')
    for tier, cfg in TIERS.items():
        original = pd.read_csv(os.path.join(data_dir, f'dataset_{tier}.csv'))
        clean = build_tier(tier, cfg, original)

        clean_path = os.path.join(data_dir, f'dataset_clean_{tier}.csv')
        clean.to_csv(clean_path, index=False)

        combined = pd.concat([original.assign(source='original', answer_type=''), clean],
                             ignore_index=True)
        combined_path = os.path.join(data_dir, f'dataset_{tier}_plus_clean.csv')
        combined.to_csv(combined_path, index=False)

        print(f'[{tier}] {clean["question"].nunique()} questions, {len(clean)} rows '
              f'-> {os.path.relpath(clean_path, ROOT)}; combined {len(combined)} rows')
        print('   labels:', clean['label'].value_counts().to_dict())
        print('   scores by type:',
              clean.groupby('answer_type')['human_score'].first().sort_values(ascending=False).to_dict())


if __name__ == '__main__':
    main()
