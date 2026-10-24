"""
helpers.py
==========
Shared utility functions for the Bengali Answer Evaluation System.
Includes Bengali digit normalisation, tokenisation, sentence alignment,
negation and numeric-mismatch detection, sentence coverage, ROUGE-L,
seeding, console encoding, and font setup.
"""

import os
import random
import re
import sys
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ─── Bengali digit utilities ──────────────────────────────────────────────────
BENGALI_DIGIT_MAP = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')


def is_bengali_numeral(word: str) -> bool:
    """Return True if *word* contains at least one decimal-digit character."""
    return any(unicodedata.category(c) == 'Nd' for c in word)


def normalize_digits(word: str) -> str:
    """Translate Bengali/Devanagari digits to ASCII digits."""
    return word.translate(BENGALI_DIGIT_MAP)


def _nfc(text: str) -> str:
    # 'য়' has a precomposed (U+09DF) and a decomposed (য + ়) spelling;
    # NFC maps both to the same sequence so string comparisons agree.
    return unicodedata.normalize('NFC', text)


# ─── Tokenisation ─────────────────────────────────────────────────────────────
_PUNCT_TABLE = str.maketrans({c: ' ' for c in '।॥,.?!;:"\'()[]{}‘’“”'})


def tokenize_bengali(text: str) -> list:
    """
    NFC-normalise *text*, replace punctuation with spaces, and split.

    Raw Bengali text attaches the dari to the last word ('না।'), so a plain
    ``str.split()`` never yields the bare word.
    """
    return _nfc(str(text)).translate(_PUNCT_TABLE).split()


# ─── Case-suffix stripping ────────────────────────────────────────────────────
# Case / plural markers removed so inflected nominals compare equal
# ('রামকে' ~ 'রাম', 'সূর্যের' ~ 'সূর্য').  A bare 'র' is deliberately not
# stripped: it would also truncate stems such as 'শিকার'.
_CASE_SUFFIXES = tuple(_nfc(s) for s in ('দেরকে', 'কেই', 'কে', 'য়ের', 'দের', 'ের', 'এর'))


def strip_case_suffix(word: str) -> str:
    """Return *word* without its case / plural suffix (keeps at least 2 chars)."""
    w = _nfc(word)
    for suf in _CASE_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            return w[:-len(suf)]
    return w


# ─── Numbers ──────────────────────────────────────────────────────────────────
_DIGIT_RUN = re.compile(r'\d+')      # \d also matches Bengali digits


def extract_numbers(text: str) -> set:
    """
    Return the set of numbers in *text* as ASCII digit strings.

    Only the digit run is kept, so '২০৬টি', '২০৬' and '206' all compare equal;
    comparing whole tokens flagged a classifier change as a wrong number.
    """
    return {normalize_digits(m) for m in _DIGIT_RUN.findall(_nfc(str(text)))}


def wrong_number(ref_text: str, stu_text: str) -> bool:
    """True when the reference states numbers and the student gives one it doesn't contain."""
    ref_nums = extract_numbers(ref_text)
    return bool(ref_nums) and bool(extract_numbers(stu_text) - ref_nums)


# ─── Sentences ────────────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r'[।॥?!\n]+')


def split_sentences(text: str) -> list:
    """Split *text* on Bengali / Latin sentence terminators; drop empty parts."""
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s.strip()]


def is_multi_sentence(text: str) -> bool:
    return len(split_sentences(text)) > 1


def align_token_sets(ref_sets: list, stu_sets: list, min_overlap: float = 0.5) -> list:
    """
    Pair up sentences (given as token sets) that talk about the same thing.

    One sentence on each side is always paired.  Otherwise every sentence is
    paired with its most similar counterpart on the other side when their
    Jaccard overlap is at least *min_overlap*; unmatched sentences (extra
    remarks, omitted facts) are left out.  Returns sorted (ref_i, stu_j) pairs.
    """
    if not ref_sets or not stu_sets:
        return []
    if len(ref_sets) == 1 and len(stu_sets) == 1:
        return [(0, 0)]

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0

    pairs = set()
    for i, a in enumerate(ref_sets):
        j = max(range(len(stu_sets)), key=lambda k: jaccard(a, stu_sets[k]))
        if jaccard(a, stu_sets[j]) >= min_overlap:
            pairs.add((i, j))
    for j, b in enumerate(stu_sets):
        i = max(range(len(ref_sets)), key=lambda k: jaccard(ref_sets[k], b))
        if jaccard(ref_sets[i], b) >= min_overlap:
            pairs.add((i, j))
    return sorted(pairs)


def align_sentences(ref_text: str, stu_text: str, min_overlap: float = 0.5) -> list:
    """Return aligned (reference_sentence, student_sentence) pairs (see align_token_sets)."""
    ref_s, stu_s = split_sentences(ref_text), split_sentences(stu_text)
    # Negation words are ignored so 'X ওঠে' and 'X ওঠে না' still align
    as_set = lambda s: {t for t in tokenize_bengali(s) if not is_negation_token(t)}
    pairs = align_token_sets([as_set(s) for s in ref_s], [as_set(s) for s in stu_s], min_overlap)
    return [(ref_s[i], stu_s[j]) for i, j in pairs]


# ─── Sentence coverage ────────────────────────────────────────────────────────
def sentence_coverage_score(ref_text: str, stu_text: str, embedder) -> float:
    """
    Compute sentence-level coverage score.

    For each sentence in *ref_text* find the maximum cosine similarity with
    any sentence in *stu_text*.  Return the average maximum similarity.
    """
    try:
        ref_sents = split_sentences(ref_text)
        stu_sents = split_sentences(stu_text)
        if not ref_sents or not stu_sents:
            return 0.0

        ref_embs = np.asarray(embedder.encode_sentences(ref_sents), dtype=float)
        stu_embs = np.asarray(embedder.encode_sentences(stu_sents), dtype=float)
        ref_embs /= np.clip(np.linalg.norm(ref_embs, axis=1, keepdims=True), 1e-9, None)
        stu_embs /= np.clip(np.linalg.norm(stu_embs, axis=1, keepdims=True), 1e-9, None)
        return float((ref_embs @ stu_embs.T).max(axis=1).mean())
    except Exception:
        return 0.0


# ─── Negation mismatch ────────────────────────────────────────────────────────
NEGATION_WORDS = {_nfc(w) for w in ('না', 'নয়', 'নি', 'নে', 'নেই', 'নাই', 'না-ই', 'নই', 'নহে')}

# Negation fused onto a finite verb: হয়নি, যায়নি, করেনি, করেননি, যাইনি
_FUSED_NEG_SUFFIXES = tuple(_nfc(s) for s in ('য়নি', 'েনি', 'ননি', 'ইনি'))
_FUSED_NEG_EXCEPTIONS = {_nfc('ফেনি')}   # district name, not a negated verb


def is_negation_token(token: str) -> bool:
    """Return True if *token* is a negation particle or a verb with fused '-নি'."""
    tok = _nfc(token)
    if tok in NEGATION_WORDS:
        return True
    if tok in _FUSED_NEG_EXCEPTIONS:
        return False
    return any(tok.endswith(s) and len(tok) > len(s) for s in _FUSED_NEG_SUFFIXES)


def has_negation(text: str) -> bool:
    """Return True if *text* contains any negation token."""
    return any(is_negation_token(t) for t in tokenize_bengali(text))


def negation_mismatch(ref_text: str, stu_text: str) -> bool:
    """
    Return True when an aligned reference / student sentence pair disagrees
    on negation.

    Comparing whole answers flagged correct paragraphs whenever the reference
    had a negation in a sentence the student simply did not repeat.
    """
    return any(has_negation(r) != has_negation(s) for r, s in align_sentences(ref_text, stu_text))


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────
class _BengaliRougeTokenizer:
    """rouge_score's default tokenizer keeps only [a-z0-9], erasing Bengali text."""

    def tokenize(self, text):
        return tokenize_bengali(text)


_ROUGE_SCORER = None


def compute_rouge_l(ref_text: str, stu_text: str) -> float:
    """Return ROUGE-L F-measure between *ref_text* and *stu_text*."""
    global _ROUGE_SCORER
    try:
        if _ROUGE_SCORER is None:
            from rouge_score import rouge_scorer as rouge_lib
            _ROUGE_SCORER = rouge_lib.RougeScorer(
                ['rougeL'], use_stemmer=False, tokenizer=_BengaliRougeTokenizer())
        scores = _ROUGE_SCORER.score(ref_text, stu_text)
        return float(scores['rougeL'].fmeasure)
    except Exception:
        return 0.0


# ─── Reproducibility / console ────────────────────────────────────────────────
def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch RNGs."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_utf8_stdio():
    """
    Switch stdout / stderr to UTF-8.  When output is redirected on Windows,
    Python uses the ANSI code page, and printing Bengali text or symbols such
    as '↓' raised UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except (AttributeError, ValueError):
            pass


# ─── Bengali font setup ───────────────────────────────────────────────────────
def setup_bengali_font():
    """
    Attempt to configure matplotlib to use a Bengali-capable font.
    Returns a FontProperties object if successful, else None.
    """
    # findSystemFonts() scans the disk directly and addfont() registers the
    # match, so matplotlib's shared font cache does not need to be flushed.
    bengali_font = None
    for f in fm.findSystemFonts():
        fname = os.path.basename(f).lower()
        if 'notosansbengali' in fname or ('noto' in fname and 'bengali' in fname):
            bengali_font = f
            break
        if 'lohit' in fname and 'bengali' in fname:
            bengali_font = f

    if bengali_font is None:
        for candidate in [
            os.path.expanduser('~/.local/share/fonts/NotoSansBengali-Regular.ttf'),
            '/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf',
            '/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf',
        ]:
            if os.path.exists(candidate):
                bengali_font = candidate
                break

    if bengali_font:
        try:
            fm.fontManager.addfont(bengali_font)
            bengali_fp = fm.FontProperties(fname=bengali_font)
            plt.rcParams['font.family'] = bengali_fp.get_name()
            print(f'[OK] Bengali font configured: {os.path.basename(bengali_font)}')
            return bengali_fp
        except Exception as e:
            print(f'[WARN] Error configuring Bengali font: {e}')
            return None
    else:
        print('[WARN] No Bengali font found.')
        return None
