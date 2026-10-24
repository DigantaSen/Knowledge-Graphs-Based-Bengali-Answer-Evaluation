"""
synthetic.py
============
Rule-generated wrong answers for the training split.

The hand-written adversarial samples are kept for testing only, so without
these rows the scorer never sees a negated, role-swapped or wrong-number
answer during training and cannot learn that such answers score 0.

Each synthetic row pairs one reference sentence (as the reference) with a
perturbed copy (score 0).  The unchanged copy is added too (scored like the
dataset's verbatim copies) so that short pairs are not all wrong.
"""

import random
import re

import pandas as pd

from src.data.dataset import _ADVERSARIAL_SAMPLES
from src.utils.helpers import has_negation, normalize_digits, split_sentences, tokenize_bengali

_BN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
_TOKEN_PUNCT = ',;:'


def _key(text: str) -> str:
    return ' '.join(tokenize_bengali(text))


def _replace_token(tokens: list, old: str, new: str) -> bool:
    for i, tok in enumerate(tokens):
        if tok.strip(_TOKEN_PUNCT) == old:
            tokens[i] = tok.replace(old, new, 1)
            return True
    return False


def _negate(sentence: str, parsed: list):
    """Add 'না' after a sentence-final verb, or 'নয়' after a nominal predicate."""
    tokens = sentence.rstrip('। ').split()
    root = next((w for w in parsed if w['deprel'] == 'root'), None)
    if not tokens or root is None or has_negation(sentence):
        return None
    if tokens[-1].strip(_TOKEN_PUNCT) != root['text']:
        return None
    suffix = 'না' if root['upos'] == 'VERB' else 'নয়'
    return ' '.join(tokens) + f' {suffix}।'


def _swap(sentence: str, parsed: list):
    """Swap the subject with the object ('রাম রাবণকে …' -> 'রাবণ রামকে …')."""
    subj = next((w for w in parsed if w['deprel'] == 'nsubj'), None)
    if subj is None:
        return None
    root_id = next((w['id'] for w in parsed if w['deprel'] == 'root'), None)
    obj = next((w for w in parsed if w['deprel'] == 'obj' and w['text'].endswith('কে')), None)
    if obj is not None:
        obj_stem = obj['text'][:-len('কে')]
    else:
        # An unmarked object, but not the noun of a compound verb (বধ করেছিলেন)
        obj = next((w for w in parsed if w['deprel'] == 'obj:bare'
                    and root_id is not None and w['id'] != root_id - 1), None)
        obj_stem = obj['text'] if obj is not None else ''
    if obj is None or len(obj_stem) < 2:
        return None
    tokens = sentence.split()
    if not (_replace_token(tokens, subj['text'], '\0SUBJ\0')
            and _replace_token(tokens, obj['text'], subj['text'] + 'কে')):
        return None
    return ' '.join(tokens).replace('\0SUBJ\0', obj_stem)


def _wrong_number(sentence: str, parsed: list = None):
    """Change the first number in the sentence by one (keeps its script)."""
    m = re.search(r'\d+', sentence)
    if not m:
        return None
    digits = m.group()
    new = str(int(normalize_digits(digits)) + 1)
    if digits[0] in '০১২৩৪৫৬৭৮৯':
        new = new.translate(_BN_DIGITS)
    return sentence[:m.start()] + new + sentence[m.end():]


_PERTURBATIONS = (('negation', _negate), ('swap', _swap), ('number', _wrong_number))


def _copy_score(df: pd.DataFrame) -> float:
    same = df[df['reference_answer'].astype(str).str.strip() == df['student_answer'].astype(str).str.strip()]
    return float(same['human_score'].mean()) if len(same) else 95.0


def make_synthetic_examples(df: pd.DataFrame, dep_parser, max_rows: int = 150,
                            random_state: int = 42, max_total: int = None) -> pd.DataFrame:
    """
    Build synthetic training rows from the reference sentences of *df*.

    At most *max_rows* perturbed rows are produced, balanced across the
    perturbation types, plus one unchanged copy per source sentence used;
    *max_total* caps all synthetic rows together, so they cannot outweigh
    the real data.  Sentences of the hand-written adversarial test samples
    are excluded, and so is any perturbation that reproduces an adversarial
    student answer.
    """
    adv_refs = {_key(s) for a in _ADVERSARIAL_SAMPLES for s in split_sentences(a['reference_answer'])}
    adv_stus = {_key(a['student_answer']) for a in _ADVERSARIAL_SAMPLES}

    sources, seen = [], set()
    for row in df.drop_duplicates('reference_answer').itertuples():
        for sent in split_sentences(row.reference_answer):
            k = _key(sent)
            if k in seen or k in adv_refs or len(k.split()) < 3:
                continue
            seen.add(k)
            sources.append((row.question, row.subject, sent + '।'))

    by_kind = {kind: [] for kind, _ in _PERTURBATIONS}
    for question, subject, sent in sources:
        parsed = dep_parser.parse(sent)[0]
        for kind, fn in _PERTURBATIONS:
            out = fn(sent, parsed)
            if out and _key(out) != _key(sent) and _key(out) not in adv_stus:
                by_kind[kind].append((question, subject, sent, out, kind))

    # Round-robin over the types so no single type dominates
    rng = random.Random(random_state)
    for cands in by_kind.values():
        rng.shuffle(cands)
    chosen = []
    while len(chosen) < max_rows and any(by_kind.values()):
        for cands in by_kind.values():
            if cands and len(chosen) < max_rows:
                chosen.append(cands.pop())

    copy_score = _copy_score(df)
    rows, copied = [], set()
    for question, subject, sent, out, kind in chosen:
        if max_total is not None and len(rows) + (1 if sent in copied else 2) > max_total:
            break
        rows.append({'question': question, 'reference_answer': sent, 'student_answer': out,
                     'label': 'incorrect', 'human_score': 0.0, 'subject': subject,
                     'synthetic': kind})
        if sent not in copied:
            copied.add(sent)
            rows.append({'question': question, 'reference_answer': sent, 'student_answer': sent,
                         'label': 'correct', 'human_score': copy_score, 'subject': subject,
                         'synthetic': 'copy'})
    out = pd.DataFrame(rows, columns=['question', 'reference_answer', 'student_answer',
                                      'label', 'human_score', 'subject', 'synthetic'])
    out.insert(0, 'id', range(-1, -len(out) - 1, -1))
    return out
