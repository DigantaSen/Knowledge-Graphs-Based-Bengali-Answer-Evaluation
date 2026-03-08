"""
coreference.py - Rule-based Bengali coreference resolution.

Resolves pronouns like তিনি (he/she), সে (he/she), তারা (they), এটি (it)
to the most likely antecedent noun found in the question or reference answer.
Applied BEFORE knowledge-graph construction to improve entity linking.
"""


class BengaliCoreferenceResolver:
    """Rule-based Bengali coreference resolver.

    Parameters
    ----------
    nlp_pipeline : stanza.Pipeline or None
        Shared Stanza pipeline for UPOS tagging.  Falls back to a
        heuristic word-filter when None.
    """

    # Bengali pronouns → referent type
    PRONOUNS = {
        "তিনি": "PERSON", "তাঁর": "PERSON", "তাঁকে": "PERSON", "তাঁদের": "PERSON",
        "সে": "PERSON", "তার": "PERSON", "তাকে": "PERSON", "তাদের": "PERSON",
        "তারা": "PERSON", "তাঁরা": "PERSON",
        "এটি": "THING", "এটা": "THING", "সেটি": "THING", "সেটা": "THING",
        "এগুলো": "THING", "সেগুলো": "THING", "ওটা": "THING", "ওটি": "THING",
        "তিনিই": "PERSON", "সেই": "PERSON",
        "এই": "THING", "ওই": "THING",
        "যিনি": "PERSON", "যে": "PERSON",
        "ইনি": "PERSON", "উনি": "PERSON", "ওনার": "PERSON",
    }

    _STOP = {
        "কে", "কী", "কোথায়", "কবে", "কিভাবে", "কেন", "এবং", "ও", "আর",
        "করে", "করেছে", "করেছিল", "করেছিলেন", "হয়", "ছিল", "ছিলেন", "থেকে",
    }

    def __init__(self, nlp_pipeline=None):
        self.nlp = nlp_pipeline

    # ── private ───────────────────────────────────────────────────────────────

    def _extract_candidate_antecedents(self, text: str) -> list:
        candidates = []
        if self.nlp:
            try:
                doc = self.nlp(text)
                for sent in doc.sentences:
                    for word in sent.words:
                        if word.upos in ("NOUN", "PROPN") and word.text not in self.PRONOUNS:
                            candidates.append(word.text)
            except Exception:
                pass

        if not candidates:
            for word in text.replace("।", "").replace(",", "").replace("?", "").split():
                if word not in self.PRONOUNS and word not in self._STOP and len(word) > 1:
                    candidates.append(word)
        return candidates

    # ── public API ────────────────────────────────────────────────────────────

    def resolve(self, student_text: str, question_text: str = "", reference_text: str = "") -> str:
        """Replace pronouns in *student_text* with their most likely antecedent."""
        resolved = student_text

        antecedent_pool = []
        if question_text:
            antecedent_pool.extend(self._extract_candidate_antecedents(question_text))
        if reference_text:
            antecedent_pool.extend(self._extract_candidate_antecedents(reference_text))

        if not antecedent_pool:
            return resolved

        for pronoun in self.PRONOUNS:
            if pronoun in resolved:
                best = antecedent_pool[0]
                if best and best != pronoun:
                    resolved = resolved.replace(pronoun, best)

        return resolved
