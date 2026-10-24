"""
karak_validator.py
==================
Semantic-role (karak) validation for Bengali answer scoring.
"""

from src.utils.helpers import (
    NEGATION_WORDS, align_token_sets, extract_numbers, has_negation as _has_negation,
    is_bengali_numeral, negation_mismatch,
)

_ROLE_KEYS = ('agent', 'object', 'marked_object', 'location_time', 'numeric', 'modifier')


class KarakValidator:
    """
    Validate student answers by checking semantic role alignment.

    Roles checked: agent, object, location/time, numeric, modifier.
    Applies role-swap penalty, missing-location penalty, numeric mismatch penalty,
    and negation mismatch hard penalty.
    """

    NEGATIONS = NEGATION_WORDS

    def __init__(self):
        pass

    def has_negation(self, text: str) -> bool:
        return _has_negation(text)

    def extract_roles(self, parsed_sentence: list) -> dict:
        """
        Extract semantic roles from a single parsed sentence (list of token dicts).

        ``marked_object`` holds only explicit objects — '-কে'-marked nominals and
        passive subjects — which are the reliable evidence for a role swap.
        ``object`` also includes unmarked nominals after the subject.

        Returns
        -------
        dict with keys: agent, object, marked_object, location_time, numeric, modifier
        """
        roles = {k: set() for k in _ROLE_KEYS}
        for word in parsed_sentence:
            deprel = word['deprel']
            # Skip punctuation tokens added by the updated parser
            if deprel == 'punct' or word.get('upos') == 'PUNCT':
                continue
            lemma  = word.get('lemma') or word['text']
            raw    = word['text']
            if is_bengali_numeral(raw) or deprel == 'nummod':
                roles['numeric'] |= extract_numbers(raw) or {raw}
            elif deprel in ('nsubj', 'obl:agent'):
                roles['agent'].add(lemma)
            elif deprel in ('obj', 'nsubj:pass'):
                roles['object'].add(lemma)
                roles['marked_object'].add(lemma)
            elif deprel == 'obj:bare':
                roles['object'].add(lemma)
            elif deprel == 'obl:loc':
                roles['location_time'].add(lemma)
            elif deprel in ('amod', 'nmod', 'nmod:poss'):
                roles['modifier'].add(lemma)
        return roles

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _surface_words(parsed_sentence: list) -> set:
        return {w['text'] for w in parsed_sentence if w['deprel'] != 'punct'}

    @staticmethod
    def _lemmas(parsed_sentence: list) -> set:
        return {w.get('lemma') or w['text'] for w in parsed_sentence if w['deprel'] != 'punct'}

    def _swap_penalty(self, ref_roles: dict, stu_roles: dict):
        """
        Role swap within one aligned sentence pair.  An entity only counts as
        moved into the object role when the student marks it explicitly
        ('-কে' or passive subject); unmarked nominals after the subject are
        too ambiguous (a template prefix shifts every one of them).
        """
        penalty = 0.0
        explanations = []
        for agent in ref_roles['agent']:
            if agent in stu_roles['marked_object'] and agent not in stu_roles['agent']:
                penalty += 0.5
                explanations.append(f"Role swap: '{agent}' was AGENT, now OBJECT.")
        subject_changed = not (ref_roles['agent'] & stu_roles['agent'])
        for obj in ref_roles['object']:
            if (subject_changed and obj in stu_roles['agent']
                    and obj not in stu_roles['object']):
                penalty += 0.5
                explanations.append(f"Role swap: '{obj}' was OBJECT, now AGENT.")
        return min(penalty, 1.0), explanations

    def _merge_roles(self, parsed: list) -> dict:
        merged = {k: set() for k in _ROLE_KEYS}
        for sent in parsed:
            for k, v in self.extract_roles(sent).items():
                merged[k] |= v
        return merged

    # ─── Public API ───────────────────────────────────────────────────────────

    def validate(self, ref_parsed: list, stu_parsed: list,
                 ref_raw: str = '', stu_raw: str = ''):
        """
        Compute the total penalty for a student answer.

        Parameters
        ----------
        ref_parsed, stu_parsed : list of sentence lists (from BengaliDependencyParser)
        ref_raw, stu_raw       : raw text strings for negation checking

        Returns
        -------
        (penalty: float, explanations: list of str)
        """
        ref_sents = [s for s in ref_parsed if s]
        stu_sents = [s for s in stu_parsed if s]

        # Role swaps are judged per aligned sentence pair: pooling roles over a
        # whole paragraph mixes up subjects of different sentences.
        penalty, explanations = 0.0, []
        pairs = align_token_sets([self._lemmas(s) for s in ref_sents],
                                 [self._lemmas(s) for s in stu_sents])
        for i, j in pairs:
            # The same inflected words in another order is a legal Bengali
            # reordering (SOV <-> OSV), not a role swap.  This must compare
            # surface forms: the role sets hold case-stripped lemmas, which are
            # identical in a genuine swap ('রাম রাবণকে' vs 'রাবণ রামকে').
            if self._surface_words(ref_sents[i]) == self._surface_words(stu_sents[j]):
                continue
            p, e = self._swap_penalty(self.extract_roles(ref_sents[i]),
                                      self.extract_roles(stu_sents[j]))
            penalty += p
            explanations += e
        penalty = min(penalty, 1.0)

        # Content omissions are judged over the whole answer
        ref_roles = self._merge_roles(ref_sents)
        stu_roles = self._merge_roles(stu_sents)

        missing_loc = ref_roles['location_time'] - stu_roles['location_time']
        if missing_loc:
            penalty += 0.1 * len(missing_loc)
            explanations.append(f'Missing location/time: {missing_loc}')

        missing_num = ref_roles['numeric'] - stu_roles['numeric']
        wrong_num   = stu_roles['numeric'] - ref_roles['numeric']
        if missing_num or wrong_num:
            penalty += 0.25 * len(missing_num) + 0.3 * len(wrong_num)
            if missing_num:
                explanations.append(f'Missing numeric: {missing_num}')
            if wrong_num:
                explanations.append(f'Wrong numeric: {wrong_num}')

        missing_mod = ref_roles['modifier'] - stu_roles['modifier']
        if missing_mod:
            penalty += 0.05 * len(missing_mod)

        if ref_raw and stu_raw and negation_mismatch(ref_raw, stu_raw):
            penalty += 0.6
            explanations.append('Negation mismatch detected (hard penalty).')

        return min(penalty, 1.0), explanations
