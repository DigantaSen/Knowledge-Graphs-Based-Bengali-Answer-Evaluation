"""
triple_extractor.py
===================
Relation triple extraction from Bengali text using BNLP POS tags.

Every clause becomes a small predicate-centred graph:

    subject --agent--> PREDICATE --object/location/time/count--> argument
    genitive --possessive--> the noun it modifies
    noun --quality--> adjective

  - The predicate (verb, with its negation and the noun of a compound verb
    such as 'বধ করেছিলেন') is a node of its own, so the GNN sees the verb and
    whether it is negated.
  - Consecutive proper nouns ('রবীন্দ্রনাথ ঠাকুর') and number + unit/time word
    ('১৯৭১ সালে') are merged into one node.
  - Each argument is a separate node; possessives and adjectives attach to
    the noun they modify, not to the subject.
  - Compound sentences are split on coordinating conjunctions (BIS 'CCD').

Triples are (head_id, relation, tail_id, triple_type).  Predicate node ids are
'PRED::<text>::<n>' so identical verbs in different clauses stay separate;
entity ids are their text, so an entity mentioned twice is one node.
"""

import re as _re
import unicodedata
from typing import List, Tuple

from src.utils.helpers import is_bengali_numeral, is_negation_token, strip_case_suffix

try:
    from bnlp import BengaliPOS
except ImportError:
    BengaliPOS = None


Triple = Tuple[str, str, str, str]
PRED_PREFIX = 'PRED::'


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def predicate_id(label: str, n: int) -> str:
    return f'{PRED_PREFIX}{label}::{n}'


def node_label(node_id: str) -> str:
    """Display / embedding text of a node id."""
    return node_id.split('::')[1] if node_id.startswith(PRED_PREFIX) else node_id


class TripleExtractor:
    """Extract predicate-centred (head, relation, tail, type) triples from Bengali text."""

    CC_TAGS     = {'CCD', 'CC'}
    VERB_TAGS   = {'VM', 'VAUX', 'VBZ', 'VBD', 'VBP', 'VBN', 'VBG', 'VB'}
    PROPN_TAGS  = {'NP', 'NNP'}
    NOUN_TAGS   = {'NC', 'NV', 'PPR', 'PRF', 'PRC', 'PRL', 'PWH',
                   'NNS', 'NN', 'PRP', 'WP'} | PROPN_TAGS
    ADJ_TAGS    = {'JJ'}
    PUNCT_TAGS  = {'PU', 'RDS'}
    SKIP_TAGS   = {'PP', 'NST', 'CSB', 'CX', 'DAB', 'DRL', 'DWH', 'JQ', 'AMN', 'ALC',
                   'CIN', 'IN', 'DT', 'RB', 'RP'}

    COPULAS     = {_nfc(s) for s in ('হয়', 'হয়েছে', 'হয়েছিল', 'হয়েছিলেন', 'হলো', 'হল',
                                      'হন', 'ছিল', 'ছিলেন', 'আছে', 'রয়েছে', 'থাকে')}
    # Verb stems that form compound verbs with a preceding noun/adjective
    LIGHT_VERB_STEMS = tuple(_nfc(s) for s in ('কর', 'হ', 'দি', 'দে', 'নি', 'পে', 'পা'))
    TIME_WORDS  = tuple(_nfc(s) for s in ('সাল', 'বছর', 'শতাব্দী', 'শতক', 'খ্রিস্টাব্দ',
                                          'খ্রিস্টপূর্ব', 'মাস', 'দিন', 'সময়', 'যুগ'))
    UNIT_WORDS  = TIME_WORDS + tuple(_nfc(s) for s in ('মিটার', 'কিলোমিটার', 'কেজি', 'শতাংশ',
                                                       'টাকা', 'জন', 'টি'))
    OBJ_SUFFIXES  = tuple(_nfc(s) for s in ('কে', 'কেই'))
    POSS_SUFFIXES = tuple(_nfc(s) for s in ('ের', 'য়ের', 'এর', 'দের', 'টির', 'টার'))
    LOC_SUFFIXES  = tuple(_nfc(s) for s in ('তে', 'তেই', 'ায়'))
    PASSIVE_MARKERS = {_nfc(s) for s in ('দ্বারা', 'কর্তৃক')}

    def __init__(self):
        print('Loading BNLP POS tagger for TripleExtractor...')
        self._cache = {}
        if BengaliPOS is not None:
            try:
                self.pos_tagger = BengaliPOS()
                print('  [OK] BNLP POS tagger loaded.')
            except Exception as e:
                print(f'  [WARN] BNLP POS tagger unavailable: {e}')
                self.pos_tagger = None
        else:
            print('  [WARN] bnlp package not available. Fallback to simple split.')
            self.pos_tagger = None

    # ─── Tagging ──────────────────────────────────────────────────────────────

    def _tag_words(self, sentence: str) -> List[Tuple[str, str]]:
        """Return [(word, bnlp_tag), …] for *sentence* (punctuation removed)."""
        words = sentence.replace('।', '').replace('?', '').split()
        if not words:
            return []
        tagged = None
        if self.pos_tagger is not None:
            try:
                tagged = self.pos_tagger.tag(sentence)
            except Exception:
                tagged = None
        if tagged is None:
            tagged = [(w, 'UNK') for w in words]
        out = []
        for w, t in tagged:
            w = _nfc(w.strip(',;:।?!\'"'))
            if not w:
                continue
            if is_negation_token(w) and t not in self.VERB_TAGS:
                t = 'CX'                  # the tagger sometimes calls 'নয়' punctuation
            elif w in self.COPULAS:
                t = 'VM'
            if t not in self.PUNCT_TAGS:
                out.append((w, t))
        return out

    def _merge_units(self, tagged):
        """Merge proper-noun runs and number + unit/time word into single tokens."""
        merged = []
        for w, t in tagged:
            if merged:
                pw, pt = merged[-1]
                # A case ending marks a new entity ('রাম রাবণকে' is two
                # people), so only an unmarked token extends the name.
                if (t in self.PROPN_TAGS and pt in self.PROPN_TAGS
                        and not pw.endswith(self.POSS_SUFFIXES)
                        and not w.endswith(self.OBJ_SUFFIXES + self.POSS_SUFFIXES + self.LOC_SUFFIXES)):
                    merged[-1] = (f'{pw} {w}', pt)
                    continue
                if pt == 'NUMBER' and w.startswith(self.UNIT_WORDS):
                    merged[-1] = (f'{pw} {w}', 'NUMBER')
                    continue
            merged.append((w, 'NUMBER' if is_bengali_numeral(w) else t))
        return merged

    def _split_on_conjunctions(self, tagged):
        clauses, current = [], []
        for tok in tagged:
            if tok[1] in self.CC_TAGS and current:
                clauses.append(current)
                current = []
            elif tok[1] not in self.CC_TAGS:
                current.append(tok)
        if current:
            clauses.append(current)
        return clauses

    # ─── Clause → triples ─────────────────────────────────────────────────────

    def _is_light_verb(self, verb: str) -> bool:
        return verb.startswith(self.LIGHT_VERB_STEMS)

    def _arg_type(self, word: str, tag: str):
        """Return the relation linking the predicate to argument *word*, or None."""
        if tag == 'NUMBER':
            return 'time' if any(tw in word for tw in self.TIME_WORDS) else 'count'
        if tag not in self.NOUN_TAGS:
            return None
        if word.endswith(self.OBJ_SUFFIXES):           # before '-ে': 'কে' ends in it too
            return 'object'
        if any(tw in word for tw in self.TIME_WORDS):
            return 'time'
        if word.endswith(self.LOC_SUFFIXES) or (tag in self.PROPN_TAGS and word.endswith('ে')):
            return 'location'
        return 'object'

    def _extract_from_clause(self, clause, pred_n: int) -> List[Triple]:
        if not clause:
            return []
        neg_word = next((w for w, t in clause if is_negation_token(w) and t not in self.VERB_TAGS), None)
        toks = [(w, t) for w, t in clause if not (is_negation_token(w) and t not in self.VERB_TAGS)]
        if not toks:
            return []

        # ── Predicate: rightmost verb (+ compound-verb noun), else last token ─
        verb_idx = next((i for i in range(len(toks) - 1, -1, -1)
                         if toks[i][1] in self.VERB_TAGS), None)
        if verb_idx is None:
            verb_idx = len(toks) - 1                     # nominal predicate ('রং লাল')
            start = verb_idx
        else:
            start = verb_idx
            if (verb_idx > 0 and self._is_light_verb(toks[verb_idx][0])
                    and toks[verb_idx - 1][1] in (self.NOUN_TAGS | self.ADJ_TAGS) - self.PROPN_TAGS):
                start = verb_idx - 1
        pred_text = ' '.join(w for w, _ in toks[start:verb_idx + 1])
        if neg_word:                                     # 'ওঠে না', 'লাল নয়'
            pred_text += f' {neg_word}'
        pred = predicate_id(pred_text, pred_n)

        args = [(i, w, t) for i, (w, t) in enumerate(toks) if not (start <= i <= verb_idx)]

        # ── Passive: the noun before 'দ্বারা' / 'কর্তৃক' is the agent ─────────
        passive_agent = next((i - 1 for i, (w, _) in enumerate(toks)
                              if w in self.PASSIVE_MARKERS and i > 0
                              and toks[i - 1][1] in self.NOUN_TAGS), None)

        # ── Subject: first unmarked nominal ──────────────────────────────────
        subject = None
        for i, w, t in args:
            if (i != passive_agent and t in self.NOUN_TAGS
                    and not w.endswith(self.OBJ_SUFFIXES + self.POSS_SUFFIXES)
                    and self._arg_type(w, t) == 'object'):
                subject = i
                break
        if subject is None and passive_agent is None:
            subject = next((i for i, w, t in args if t in self.NOUN_TAGS), None)

        triples: List[Triple] = []
        if passive_agent is not None:
            # 'রাবণের দ্বারা রাম নিহত হয়েছেন': রাবণ acts, রাম is the patient
            triples.append((strip_case_suffix(toks[passive_agent][0]), 'agent', pred, 'agent'))
            if subject is not None:
                triples.append((pred, 'object', toks[subject][0], 'object'))
        elif subject is not None:
            triples.append((toks[subject][0], 'agent', pred, 'agent'))

        def next_noun(after):
            return next((w for i, w, t in args if i > after and t in self.NOUN_TAGS
                         and not w.endswith(self.POSS_SUFFIXES)), None)

        for i, w, t in args:
            if i in (subject, passive_agent) or t in self.SKIP_TAGS:
                continue
            if t in self.ADJ_TAGS:
                head = next_noun(i)
                triples.append((head, 'quality', w, 'quality') if head
                               else (pred, 'quality', w, 'quality'))
                continue
            if t in self.NOUN_TAGS and w.endswith(self.POSS_SUFFIXES):
                head = next_noun(i)
                triples.append((w, 'possessive', head, 'possessive') if head
                               else (pred, 'object', w, 'object'))
                continue
            rel = self._arg_type(w, t)
            if rel:
                triples.append((pred, rel, w, rel))

        if not triples:
            # A lone predicate: 'none' triples only add the node
            triples.append((pred, 'none', pred, 'none'))
        return triples

    # ─── Public API ───────────────────────────────────────────────────────────

    def extract_triples(self, parsed_sentences=None, raw_text: str = '') -> List[Triple]:
        """
        Extract triples from *raw_text*.

        Returns
        -------
        list of (head_id, relation, tail_id, triple_type) tuples
        """
        text = raw_text if raw_text else ''
        if not text:
            return []
        if text in self._cache:
            return self._cache[text]

        all_triples: List[Triple] = []
        pred_n = 0
        sentences = [s.strip() for s in _re.split(r'[।?!\n]+', text) if s.strip()]
        for sent in sentences:
            tagged = self._merge_units(self._tag_words(sent))
            for clause in self._split_on_conjunctions(tagged):
                for t in self._extract_from_clause(clause, pred_n):
                    if t not in all_triples:
                        all_triples.append(t)
                pred_n += 1

        self._cache[text] = all_triples
        return all_triples
