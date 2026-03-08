"""
dependency_parser.py - Stanza-based Bengali dependency parser with SOV fallback.
"""

class BengaliDependencyParser:
    """Parses Bengali sentences into dependency trees.

    Parameters
    ----------
    nlp_pipeline : stanza.Pipeline or None
        Shared Stanza pipeline.  When None, a simple Subject-Object-Verb
        heuristic is used as fallback.
    """

    def __init__(self, nlp_pipeline=None):
        self.nlp = nlp_pipeline

    # ── public API ────────────────────────────────────────────────────────────

    def parse(self, text: str) -> list:
        """Return a list-of-sentences, each sentence being a list of word dicts.

        Each word dict has keys:
            id, text, lemma, upos, head, deprel
        """
        if not self.nlp:
            return self._sov_fallback(text)

        doc = self.nlp(text)
        parsed = []
        for sentence in doc.sentences:
            words = []
            for word in sentence.words:
                words.append({
                    "id": word.id,
                    "text": word.text,
                    "lemma": word.lemma if word.lemma else word.text,
                    "upos": word.upos if word.upos else "X",
                    "head": word.head,
                    "deprel": word.deprel if word.deprel else "dep",
                })
            parsed.append(words)
        return parsed

    def print_tree(self, parsed_sentences: list) -> None:
        """Pretty-print a dependency tree to stdout."""
        for i, sentence in enumerate(parsed_sentences):
            print(f"--- Sentence {i + 1} ---")
            for word in sentence:
                head_id = word["head"]
                head_text = (
                    sentence[head_id - 1]["text"]
                    if 0 < head_id <= len(sentence)
                    else "ROOT"
                )
                print(f"  {word['text']} ({word['deprel']}) <-- {head_text}")

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sov_fallback(text: str) -> list:
        """Minimal SOV heuristic parser used when Stanza is unavailable."""
        words_raw = text.split()
        if not words_raw:
            return [[]]
        parsed_words = []
        n = len(words_raw)
        for i, w in enumerate(words_raw):
            if i == 0:
                deprel, head = "nsubj", n
            elif i == n - 1:
                deprel, head = "root", 0
            else:
                deprel, head = "obj", n
            parsed_words.append({
                "id": i + 1,
                "text": w,
                "lemma": w,
                "upos": "NOUN",
                "head": head,
                "deprel": deprel,
            })
        return [parsed_words]
