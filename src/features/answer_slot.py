"""
answer_slot.py
==============
Answer-slot features: compare only the part of each answer that answers the
question.

Words that also occur in the question ('তাজমহল', 'নির্মাণ' for "তাজমহল কে
নির্মাণ করেছিলেন?") are shared by right and wrong answers alike, so they
inflate the similarity of a wrong answer and add nothing for a correct brief
one.  Removing them leaves the answer content:

    reference  "সম্রাট শাহজাহান … মমতাজ মহলের স্মৃতিতে"
    brief      "শাহজাহান"          → precise, contains the key
    wrong      "সম্রাট আকবর …"      → 'আকবর' is unsupported, key missing

Features (all in [0, 1]):
    ans_recall    – share of the reference's answer content found in the student answer
    ans_precision – share of the student's answer content supported by the reference
    key_match     – whether the key answer item is present (1) or not (0)

Two words match when their case-stripped lemmas are equal, one contains the
other, or (non-numbers only) their LaBSE similarity reaches SYNONYM_THRESHOLD.
Numbers match only on their digits.
"""

import unicodedata

from src.config import SYNONYM_THRESHOLD
from src.utils.helpers import extract_numbers, is_bengali_numeral, strip_case_suffix

_CONTENT_UPOS = {'NOUN', 'PROPN', 'ADJ', 'NUM'}
_DIRECTIONS = tuple(unicodedata.normalize('NFC', s) for s in ('উত্তর', 'দক্ষিণ', 'পূর্ব', 'পশ্চিম'))

# Titles and role nouns are never the key answer ('সম্রাট শাহজাহান' → শাহজাহান)
_TITLES = {_t for _t in (unicodedata.normalize('NFC', s) for s in (
    'সম্রাট', 'রাজা', 'রানী', 'মহারাজা', 'নবাব', 'সুলতান', 'বাদশাহ', 'লর্ড', 'স্যার',
    'বিজ্ঞানী', 'কবি', 'লেখক', 'ঔপন্যাসিক', 'সাহিত্যিক', 'কথাসাহিত্যিক', 'বঙ্গবন্ধু',
    'মহাত্মা', 'ড.', 'ডঃ', 'জনাব', 'শ্রী', 'বেগম', 'মুঘল', 'ইংরেজ', 'বিখ্যাত'))}

# Question word → which reference item is the key answer
_WH_NUMBER   = ('কবে', 'কখন', 'কত', 'কয়টি', 'কয়জন', 'কতজন', 'সালে')
_WH_PLACE    = ('কোথায়', 'কোন দিকে', 'কোন দিক', 'তীরে')
_WH_WHICH    = 'কোন'


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', str(text))


def _tokens(text: str, comps: dict) -> list:
    """Parsed tokens [(lemma, upos, text)] of *text*, punctuation dropped."""
    parsed = comps['dep_parser'].parse(comps['text_processor'].normalize(text))
    out = []
    for sent in parsed:
        for w in sent:
            if w['deprel'] == 'punct':
                continue
            out.append((_nfc(strip_case_suffix(w['text'])), w['upos'], _nfc(w['text'])))
    return out


def _is_content(lemma: str, upos: str, text: str) -> bool:
    return (upos in _CONTENT_UPOS or is_bengali_numeral(text)
            or any(d in text for d in _DIRECTIONS))


def _numbers(text: str) -> set:
    return extract_numbers(text)


def _match(a: tuple, b: tuple, embedder) -> bool:
    (la, _, ta), (lb, _, tb) = a, b
    na, nb = _numbers(ta), _numbers(tb)
    if na or nb:
        return bool(na & nb)
    if la == lb or (len(la) >= 2 and len(lb) >= 2 and (la in lb or lb in la)):
        return True
    if embedder is None:
        return False
    try:
        return embedder.word_cosine_similarity(la, lb) >= SYNONYM_THRESHOLD
    except Exception:
        return False


def _key(question: str, q_tokens: list, ref_novel: list):
    """Pick the reference item that answers the question (None if unclear)."""
    q = _nfc(question)
    ref_novel = [t for t in ref_novel if t[0] not in _TITLES and t[2] not in _TITLES] or ref_novel
    if not ref_novel:
        return None
    if any(w in q for w in _WH_NUMBER):
        num = next((t for t in ref_novel if _numbers(t[2])), None)
        if num:
            return num
    if any(w in q for w in _WH_PLACE):
        place = next((t for t in ref_novel if any(d in t[2] for d in _DIRECTIONS)
                      or t[1] == 'PROPN'), None)
        if place:
            return place
    if _WH_WHICH in q:
        # 'কোন বিশ্ববিদ্যালয়' → the reference word just before / fused with বিশ্ববিদ্যালয়
        q_words = [t[2] for t in q_tokens]
        if _WH_WHICH in q_words:
            i = q_words.index(_WH_WHICH)
            if i + 1 < len(q_tokens):
                head = q_tokens[i + 1][0]
                for t in ref_novel:
                    if head and head != t[0] and head in t[2]:
                        return t                      # fused: 'বঙ্গোপসাগরের' ⊃ 'সাগর'
    propn = next((t for t in ref_novel if t[1] == 'PROPN'), None)
    return propn or ref_novel[0]


def answer_slot_features(question: str, ref_text: str, stu_text: str, comps: dict,
                         debug: bool = False) -> dict:
    """Return {'ans_recall', 'ans_precision', 'key_match'} for one answer pair."""
    embedder = comps.get('embedder')
    q_tokens = _tokens(question, comps)
    q_lemmas = {t[0] for t in q_tokens} | {t[2] for t in q_tokens}

    def novel(text):
        return [t for t in _tokens(text, comps)
                if _is_content(*t) and t[0] not in q_lemmas and t[2] not in q_lemmas]

    ref_novel, stu_novel = novel(ref_text), novel(stu_text)
    key = _key(question, q_tokens, ref_novel)

    if not ref_novel:
        out = {'ans_recall': 1.0 if not stu_novel else 0.5,
               'ans_precision': 1.0 if not stu_novel else 0.5, 'key_match': 1.0}
    else:
        recall = sum(any(_match(r, s, embedder) for s in stu_novel) for r in ref_novel) / len(ref_novel)
        precision = (sum(any(_match(s, r, embedder) for r in ref_novel) for s in stu_novel)
                     / len(stu_novel)) if stu_novel else 0.0
        key_hit = key is not None and any(_match(key, s, embedder) for s in stu_novel)
        out = {'ans_recall': recall, 'ans_precision': precision, 'key_match': float(key_hit)}
    if debug:
        out['_key'] = key[2] if key else None
        out['_ref_novel'] = [t[2] for t in ref_novel]
        out['_stu_novel'] = [t[2] for t in stu_novel]
    return out
