"""
coreference.py
==============
Rule-based coreference resolution for Bengali.

Strategy
--------
Process an answer sentence by sentence.  For each sentence:

  1. Resolve PERSON pronouns (তিনি, সে, তার …) to the most recently
     mentioned PERSON-type entity.
  2. Resolve THING pronouns  (এটি, সেটি, এটা …) to the most recently
     mentioned THING-type entity.
  3. After resolving, scan the *original* sentence for new entity
     candidates and update the recency focus so that subsequent
     sentences pick the right antecedent.

Seeding
-------
The initial focus is seeded from the *reference* first and the *question*
second.  The question holds the interrogative, not the answer ('কে গীতাঞ্জলি
লিখেছিলেন?' contains no person), so seeding from it first resolved 'তিনি' to
'গীতাঞ্জলি'.

Candidates
----------
Antecedent candidates are nouns from the BNLP POS tagger, with case
suffixes stripped: PERSON = first proper noun, THING = first common noun of
the sentence.  Without a tagger the first / last content word is used.
"""

import re as _re
from typing import Dict, List, Optional

from src.utils.helpers import strip_case_suffix

try:
    from bnlp import BengaliPOS
except ImportError:
    BengaliPOS = None


class BengaliCoreferenceResolver:
    """
    Sentence-level, recency-aware Bengali coreference resolver.

      • Works sentence-by-sentence — later sentences use the most recently
        mentioned entity, not always the first word in the question.
      • Separates PERSON and THING pronoun resolution.
      • Replaces only the first occurrence of each pronoun per sentence to
        avoid over-substitution.
      • Updates the "focus" entity after processing each sentence.
    """

    # Each pronoun maps to its semantic type: PERSON or THING.  Relative and
    # demonstrative forms (যে, যিনি, সেই) are left alone: they are usually a
    # conjunction / determiner, and replacing them breaks the sentence.
    PRONOUNS: Dict[str, str] = {
        'তিনি':  'PERSON', 'তাঁর':   'PERSON', 'তাঁকে':  'PERSON',
        'তাঁদের':'PERSON', 'সে':     'PERSON', 'তার':    'PERSON',
        'তাকে':  'PERSON', 'তাদের':  'PERSON', 'তারা':   'PERSON',
        'তাঁরা': 'PERSON', 'তিনিই':  'PERSON', 'ইনি':    'PERSON',
        'উনি':   'PERSON', 'ওনার':   'PERSON',
        'এটি':   'THING',  'এটা':    'THING',  'সেটি':   'THING',
        'সেটা':  'THING',  'এগুলো':  'THING',  'সেগুলো': 'THING',
        'ওটা':   'THING',  'ওটি':    'THING',
    }

    # Words that are never useful resolution targets (tagger-less fallback)
    STOP: set = {
        'কে', 'কী', 'কোথায়', 'কবে', 'কিভাবে', 'কেন', 'এবং', 'ও', 'আর',
        'করে', 'করেছে', 'করেছিল', 'করেছিলেন', 'হয়', 'ছিল', 'ছিলেন',
        'থেকে', 'যে', 'তখন', 'এখন', 'সেখানে', 'এখানে',
    }
    PROPN_TAGS = {'NP', 'NNP'}
    NOUN_TAGS  = {'NC', 'NNS', 'NN'}

    _SENT_SEP = _re.compile(r'[।?!\n]+')
    _TOKEN_PUNCT = ',?!;:"\'()'

    def __init__(self, pos_tagger=None):
        self.pos_tagger = pos_tagger
        if self.pos_tagger is None and BengaliPOS is not None:
            try:
                self.pos_tagger = BengaliPOS()
            except Exception:
                self.pos_tagger = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """Split *text* on sentence terminators; return non-empty stripped segments."""
        return [s.strip() for s in self._SENT_SEP.split(text) if s.strip()]

    def _candidates(self, text: str) -> Dict[str, Optional[str]]:
        """Return {'PERSON': …, 'THING': …} antecedent candidates of one sentence."""
        clean = text.replace('।', '').replace(',', '').replace('?', '')
        if self.pos_tagger is not None:
            try:
                tagged = self.pos_tagger.tag(clean)
            except Exception:
                tagged = None
            if tagged:
                tokens = [(w.strip(self._TOKEN_PUNCT), t) for w, t in tagged]
                tokens = [(w, t) for w, t in tokens if w and w not in self.PRONOUNS]
                propn = next((w for w, t in tokens if t in self.PROPN_TAGS), None)
                noun  = next((w for w, t in tokens if t in self.NOUN_TAGS), None)
                first = propn or noun
                return {'PERSON': strip_case_suffix(first) if first else None,
                        'THING':  strip_case_suffix(noun or first) if (noun or first) else None}
        words = [w for w in clean.split()
                 if w not in self.PRONOUNS and w not in self.STOP and len(w) > 1]
        if not words:
            return {'PERSON': None, 'THING': None}
        return {'PERSON': words[0], 'THING': words[-1] if len(words) > 1 else words[0]}

    def _update_focus(self, focus: dict, sentence: str):
        """Make the entities of *sentence* the current antecedents."""
        for k, v in self._candidates(sentence).items():
            if v:
                focus[k] = v

    def _resolve_sentence(self, sentence: str, focus: dict) -> str:
        """
        Replace whole-token pronouns in *sentence* with their antecedents.

        Matching is per token, never per substring: substring replacement
        rewrote 'সে' inside 'সেকেন্ডে' and 'তার' inside 'তারা' / 'স্বাধীনতার'.
        Only the first occurrence of each pronoun is replaced, so repeated
        pronouns referring to different entities aren't all collapsed.
        """
        tokens   = sentence.split()
        replaced = set()
        for i, tok in enumerate(tokens):
            core  = tok.strip(self._TOKEN_PUNCT)
            ptype = self.PRONOUNS.get(core)
            if ptype is None or core in replaced:
                continue
            antecedent = focus.get(ptype)
            if antecedent and antecedent != core:
                tokens[i] = tok.replace(core, antecedent, 1)
                replaced.add(core)
        return ' '.join(tokens)

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, text: str,
                question_text: str = '',
                reference_text: str = '') -> str:
        """
        Return *text* with recognized pronouns replaced by their most likely
        antecedents.

        Parameters
        ----------
        text            : the answer to resolve (student or reference)
        question_text   : used to seed the initial entity focus (secondary)
        reference_text  : used to seed the initial entity focus (primary)
        """
        # Quick exit: no pronoun token present — nothing to do
        if not any(t.strip(self._TOKEN_PUNCT + '।') in self.PRONOUNS
                   for t in text.split()):
            return text

        # ── Seed focus from context: reference first, then question ─────────
        focus: Dict[str, Optional[str]] = {'PERSON': None, 'THING': None}
        for ctx in (reference_text, question_text):
            if not ctx:
                continue
            for sent in self._split_sentences(ctx):
                for k, v in self._candidates(sent).items():
                    if focus[k] is None and v:
                        focus[k] = v
            if focus['PERSON'] and focus['THING']:
                break   # both slots filled — stop scanning context

        # ── Sentence-level resolution ────────────────────────────────────────
        sentences = self._split_sentences(text)
        if not sentences:
            return text

        resolved_parts = []
        changed = False
        for sent in sentences:
            # Resolve pronouns using CURRENT focus (i.e. most recent entity)
            resolved_sent = self._resolve_sentence(sent, focus)
            changed |= resolved_sent != ' '.join(sent.split())
            resolved_parts.append(resolved_sent)

            # Update focus from ORIGINAL sentence (before pronoun substitution)
            # so substituted words don't pollute the candidate pool.
            self._update_focus(focus, sent)

        if not changed:
            return text
        resolved = '। '.join(resolved_parts)
        return resolved + '।' if text.rstrip().endswith('।') else resolved
