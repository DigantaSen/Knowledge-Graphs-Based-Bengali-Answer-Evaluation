"""
overrides.py
============
Hard post-processing overrides applied after ensemble prediction.
"""

import numpy as np

from src.config import HARD_NEG_CAP, HARD_NUM_CAP


def apply_hard_overrides(scores, neg_flags, wrong_num_flags, multi_sentence_flags=None):
    """
    Apply hard post-processing caps after ensemble prediction.

    Rules (fixed a-priori, not tuned; single-sentence references only)
    -----
    1. Negation mismatch  → cap score at HARD_NEG_CAP (30)
    2. Wrong number value → cap score at HARD_NUM_CAP (25)

    For multi-sentence references the caps are skipped: one negated or wrong
    fact among several gets partial credit from human graders, and the
    learned scorer sees the same flags as features.

    Parameters
    ----------
    scores               : array-like of predicted scores
    neg_flags            : array-like (1 = negation mismatch)
    wrong_num_flags      : array-like (1 = wrong numeric value)
    multi_sentence_flags : array-like of bool (True = multi-sentence reference)

    Returns
    -------
    adjusted : np.ndarray of adjusted scores
    override_log : list of (list of str) — reasons per sample
    """
    adjusted     = np.array(scores, dtype=float)
    if multi_sentence_flags is None:
        multi_sentence_flags = np.zeros(len(adjusted), dtype=bool)
    override_log = []

    for i in range(len(adjusted)):
        reasons = []
        if not multi_sentence_flags[i]:
            if neg_flags[i] and adjusted[i] > HARD_NEG_CAP:
                adjusted[i] = HARD_NEG_CAP
                reasons.append(f'Negation mismatch — score capped at {HARD_NEG_CAP:.0f}')
            if wrong_num_flags[i] and adjusted[i] > HARD_NUM_CAP:
                adjusted[i] = HARD_NUM_CAP
                reasons.append(f'Wrong numeric value — score capped at {HARD_NUM_CAP:.0f}')
        override_log.append(reasons)

    return adjusted, override_log
