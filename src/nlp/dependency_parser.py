"""
dependency_parser.py
====================
BNLP-based Bengali dependency parser.
"""

import unicodedata

from src.utils.helpers import NEGATION_WORDS, is_bengali_numeral, strip_case_suffix

try:
    from bnlp import BengaliPOS
except ImportError:
    BengaliPOS = None


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


class BengaliDependencyParser:
    """
    Lightweight dependency parser for Bengali using BNLP POS tags.

    Falls back to a position-based SOV heuristic when the BNLP tagger
    is unavailable.

    Relations produced
    ------------------
    nsubj       first unmarked nominal (the subject)
    nsubj:pass  subject of a passive clause (the logical patient)
    obj         '-কে'-marked nominal (an explicit object)
    obj:bare    other unmarked nominal after the subject (object or part of
                a compound; only weak evidence of objecthood)
    obl:agent   nominal before 'দ্বারা' / 'কর্তৃক' (the logical agent of a passive)
    obl:loc     locative nominal
    nmod:poss   genitive nominal
    amod, nummod, advmod (negation), case, root, dep
    """

    NEGATIONS     = NEGATION_WORDS
    OBJ_SUFFIXES  = tuple(_nfc(s) for s in ('কে', 'কেই'))
    # 'য়' alone also ends বিষয় / সময়, and 'য়ে' ends মেয়ে, so only the
    # unambiguous locative endings are used; '-এ' is accepted on proper nouns
    # only (রাজস্থানে), because common nouns such as ছেলে end in it too.
    LOC_SUFFIXES       = tuple(_nfc(s) for s in ('তে', 'তেই', 'ায়'))
    PROPN_LOC_SUFFIXES = tuple(_nfc(s) for s in ('ে',))
    # Explicit genitive endings only: a bare 'র' also ends ঠাকুর / সরকার / উপর.
    POSS_SUFFIXES = tuple(_nfc(s) for s in ('ের', 'য়ের', 'এর', 'দের', 'টির', 'টার'))
    PASSIVE_MARKERS = {_nfc(s) for s in ('দ্বারা', 'কর্তৃক')}
    # Copula / light verbs that the tagger often labels as nouns
    COPULAS = {_nfc(s) for s in ('হয়', 'হয়েছে', 'হয়েছিল', 'হয়েছিলেন', 'হলো', 'হল',
                                  'হন', 'ছিল', 'ছিলেন', 'আছে', 'রয়েছে', 'থাকে')}
    NOMINAL_UPOS  = ('NOUN', 'PROPN', 'PRON')

    # bnlp.BengaliPOS emits the BIS / LDC-IL tagset (NC, NP, VM, PPR, ...).
    BNLP_TO_UPOS = {
        'NC':  'NOUN',  'NP':  'PROPN', 'NV':  'NOUN',  'CCL': 'NOUN',
        'NST': 'ADP',                                   # মধ্যে, উপর, কাছে
        'VM':  'VERB',  'VAUX': 'VERB', 'LC':  'VERB',  # LC = conditional (করলে)
        'PPR': 'PRON',  'PRF': 'PRON',  'PRC': 'PRON',  'PRL': 'PRON', 'PWH': 'PRON',
        'JJ':  'ADJ',
        # JQ covers একটি / প্রথম / আরও; tagging it NUM would flag those as
        # numeric errors.  Digit numerals are caught by is_bengali_numeral().
        'JQ':  'DET',
        'DAB': 'DET',   'DRL': 'DET',   'DWH': 'DET',
        'AMN': 'ADV',   'ALC': 'ADV',
        'PP':  'ADP',   'CCD': 'CCONJ', 'CSB': 'SCONJ',
        'CX':  'PART',  'CIN': 'INTJ',
        'PU':  'PUNCT', 'RDS': 'SYM',   'RDF': 'X',     'RDX': 'X',
        # Penn-style tags, for taggers that emit those instead
        'NNP': 'PROPN', 'NNS': 'NOUN', 'NN': 'NOUN',
        'VBZ': 'VERB',  'VBD': 'VERB', 'VBN': 'VERB',
        'VB':  'VERB',  'VBP': 'VERB', 'VBG': 'VERB',
        'RB':  'ADV',
        'QF':  'NUM',   'QT_QTF': 'NUM', 'QT_QTC': 'NUM', 'Q': 'NUM',
        'CC':  'CCONJ', 'IN': 'ADP',   'DT': 'DET',
        'PRP': 'PRON',  'WP': 'PRON',
        'NEG': 'PART',  'RP': 'PART',
        'UNK': 'X',
    }

    def __init__(self):
        print('Loading BNLP POS tagger for dependency parsing...')
        self._cache = {}
        if BengaliPOS is not None:
            try:
                self.pos_tagger = BengaliPOS()
                self.use_bnlp = True
                print('  [OK] BNLP POS tagger loaded.')
            except Exception as e:
                print(f'  [WARN] BNLP POS tagger failed: {e}. SOV fallback active.')
                self.pos_tagger = None
                self.use_bnlp = False
        else:
            print('  [WARN] bnlp package not available. SOV fallback active.')
            self.pos_tagger = None
            self.use_bnlp = False

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _bnlp_tag(self, text: str):
        try:
            return self.pos_tagger.tag(text)
        except Exception:
            return [(w, 'UNK') for w in text.split()]

    def _tag_to_upos(self, word: str, bnlp_tag: str) -> str:
        if _nfc(word) in self.COPULAS:
            return 'VERB'
        return self.BNLP_TO_UPOS.get(bnlp_tag, 'X')

    def _assign_deprel(self, word, upos, position, verb_idx, subj_assigned, before_passive_marker):
        w = _nfc(word)
        if is_bengali_numeral(word) or upos == 'NUM':
            return 'nummod'
        if w in self.NEGATIONS:
            return 'advmod'
        if position == verb_idx:
            return 'root'
        if upos in self.NOMINAL_UPOS:
            if before_passive_marker:
                return 'obl:agent'
            if w.endswith(self.OBJ_SUFFIXES):
                return 'obj'
            if w.endswith(self.POSS_SUFFIXES):
                return 'nmod:poss'
            if w.endswith(self.LOC_SUFFIXES) or (upos == 'PROPN' and w.endswith(self.PROPN_LOC_SUFFIXES)):
                return 'obl:loc'
            if not subj_assigned:
                return 'nsubj'
            return 'obj:bare'
        if upos == 'ADJ':
            return 'amod'
        if upos in ('ADP', 'PART'):
            return 'case'
        return 'dep'

    # ─── Public API ───────────────────────────────────────────────────────────

    def parse(self, text: str) -> list:
        """
        Parse *text* and return a list of sentence lists (CoNLL-U style dicts).

        Each token dict has keys: id, text, lemma, upos, head, deprel, bnlp_tag.
        Results are cached per text; callers must not mutate them.
        """
        if text in self._cache:
            return self._cache[text]

        # Split on Bengali sentence boundaries FIRST.  Commas are kept for the
        # tagger — they help it identify phrase boundaries.
        sentences_raw = [s.strip() for s in text.replace('?', '\n').replace('!', '\n')
                         .replace('।', '\n').split('\n') if s.strip()]
        if not sentences_raw:
            sentences_raw = [text.replace('।', '').replace('?', '').strip()]

        parsed_doc = []
        for sent_text in sentences_raw:
            words_raw = sent_text.split()
            if not words_raw:
                continue
            tagged_raw = self._bnlp_tag(sent_text) if self.use_bnlp else [(w, 'UNK') for w in words_raw]
            # Drop pure-comma / pure-punctuation tokens after tagging
            tagged = [(w.strip(','), t) for w, t in tagged_raw
                      if w.strip(',.?।') != '']
            if not tagged:
                continue

            # Find rightmost verb
            verb_idx = len(tagged) - 1
            for i in range(len(tagged) - 1, -1, -1):
                if self._tag_to_upos(*tagged[i]) == 'VERB':
                    verb_idx = i
                    break

            agent_marked = {i - 1 for i, (w, _) in enumerate(tagged)
                            if _nfc(w) in self.PASSIVE_MARKERS}

            parsed_sent = []
            subj_assigned = False
            for i, (word, bnlp_tag) in enumerate(tagged):
                upos = self._tag_to_upos(word, bnlp_tag)
                # Comma/punctuation tokens: mark as PUNCT with 'punct' deprel
                if word in (',', '।', '.', '!', '?', ';', ':') or upos == 'PUNCT':
                    parsed_sent.append({
                        'id': i + 1, 'text': word, 'lemma': word,
                        'upos': 'PUNCT', 'head': verb_idx + 1, 'deprel': 'punct',
                        'bnlp_tag': bnlp_tag,
                    })
                    continue

                deprel = self._assign_deprel(word, upos, i, verb_idx, subj_assigned,
                                             i in agent_marked)
                if deprel == 'nsubj':
                    subj_assigned = True
                head = 0 if deprel == 'root' else verb_idx + 1
                # Strip case markers so 'রামকে' (object) and 'রাম' (agent)
                # are recognised as the same entity by the Karak validator.
                lemma = strip_case_suffix(word) if upos in self.NOMINAL_UPOS else word
                parsed_sent.append({
                    'id': i + 1, 'text': word, 'lemma': lemma,
                    'upos': upos, 'head': head, 'deprel': deprel, 'bnlp_tag': bnlp_tag,
                })

            # In a passive clause the grammatical subject is the patient
            if any(w['deprel'] == 'obl:agent' for w in parsed_sent):
                for w in parsed_sent:
                    if w['deprel'] == 'nsubj':
                        w['deprel'] = 'nsubj:pass'
            parsed_doc.append(parsed_sent)

        result = parsed_doc if parsed_doc else [[]]
        self._cache[text] = result
        return result
