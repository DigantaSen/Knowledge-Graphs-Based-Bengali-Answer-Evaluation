"""
preprocessing.py - Bengali text normalization, tokenization and lemmatisation.
"""

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

class TextPreprocessor:
    """Normalizes and tokenizes Bengali text.

    Parameters
    ----------
    nlp_pipeline : stanza.Pipeline or None
        A pre-initialized Stanza pipeline shared across components.
        When None the preprocessor falls back to whitespace tokenization.
    """

    STOPWORDS = {
        "কী", "কেন", "কোথায়", "কখন", "কিভাবে", "যে", "ও", "এবং", "আর",
        "কিন্তু", "অথবা", "নয়", "না", "নি", "করি", "করে", "করা", "হয়",
        "হয়েছে", "ছিল", "ছিলেন", "হবে", "হবেন",
    }

    def __init__(self, nlp_pipeline=None):
        factory = IndicNormalizerFactory()
        self.normalizer = factory.get_normalizer("bn")
        self.nlp = nlp_pipeline

    # ── private helpers ───────────────────────────────────────────────────────

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        return self.normalizer.normalize(text)

    def remove_punctuation(self, text: str) -> str:
        for p in ["।", ",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "[", "]", "{", "}"]:
            text = text.replace(p, " ")
        return " ".join(text.split())

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, text: str, remove_stops: bool = False) -> list:
        """Return a list of lemmas (or tokens) from *text*."""
        text = self.normalize(text)
        text = self.remove_punctuation(text)

        if self.nlp is None:
            tokens = text.split()
            if remove_stops:
                tokens = [t for t in tokens if t not in self.STOPWORDS]
            return tokens

        doc = self.nlp(text)
        clean_tokens = []
        for sentence in doc.sentences:
            for word in sentence.words:
                lemma = word.lemma if word.lemma else word.text
                if remove_stops and lemma in self.STOPWORDS:
                    continue
                clean_tokens.append(lemma)
        return clean_tokens
