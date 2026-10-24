"""
text_preprocessor.py
=====================
Bengali text normalisation using the Indic NLP Library.
"""

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


class TextPreprocessor:
    """Normalise Bengali text."""

    def __init__(self):
        factory = IndicNormalizerFactory()
        self.normalizer = factory.get_normalizer('bn')

    def normalize(self, text: str) -> str:
        """Apply Indic normalisation to *text*."""
        if not isinstance(text, str):
            text = str(text)
        return self.normalizer.normalize(text)
